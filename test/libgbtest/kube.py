# SPDX-License-Identifier: Apache-2.0

"""Skip gates for build tests that launch into Kubernetes via SkyPilot.

Three gates share one probe: :func:`kind_cluster_reachable` for the local kind
cluster (skypilot/kubernetes/local), :func:`vela_cluster_reachable` for the Vela
OpenShift cluster (skypilot/kubernetes/vela), and :func:`cpu_cluster_reachable`
for the gb CPU cluster (skypilot/kubernetes/cpu). The kind gate is described
first; the remote ones differ only in context, namespace, claim and set-up.

The Kubernetes counterpart of the slurm gate (``_slurm_cluster_reachable``) and
of ``aws_credentials_present``: a kube build test runs only when ``make
kube-setup`` (``scripts/kube/setup-kind.sh``) has brought the cluster and its
shared volume up.

The tests use ``space://environments/skypilot/kubernetes/local``, which pins
SkyPilot to the kind context through ``kubernetes.allowed_contexts`` in its
``cloud_config``. So the gate does not require kind to be the CURRENT kube
context — the env, not the ambient context, decides where pods go. It checks
that the pinned context exists and answers, and that the env's shared-volume
claim exists, so a missing ``kube-setup`` skips cleanly instead of leaving pods
``Pending`` on an absent claim.
"""

import functools
import os
import shutil
import subprocess

from libgbtest.xdist_groups import SKYPILOT_XDIST_GROUP

#: Context ``sky local up`` creates with its default cluster name; the one
#: skypilot/kubernetes/local pins in ``allowed_contexts``.
DEFAULT_KIND_CONTEXT = "kind-skypilot"

#: PersistentVolumeClaim skypilot/kubernetes/local mounts at /shared; created by
#: ``make kube-setup`` in the ``default`` namespace.
SHARED_CLAIM = "gb-shared"

#: Every kube build test class carries this, so they run one at a time. Each pod
#: requests 2 CPUs and a CI runner's kind node has ~3 left after the control
#: plane, so concurrent xdist workers would leave pods unschedulable; with the
#: Makefile's ``--dist=loadgroup`` a group runs serially on one worker. Shared with
#: every skypilot_integration test, so kube never overlaps slurm either.
KUBE_XDIST_GROUP = SKYPILOT_XDIST_GROUP


def expected_kind_context() -> str:
    """The kube context the tests expect (``GBTEST_SKY_KUBE_CONTEXT``).

    Overriding it only changes what the gate checks; skypilot/kubernetes/local
    itself pins ``kind-skypilot``.
    """
    return os.environ.get("GBTEST_SKY_KUBE_CONTEXT") or DEFAULT_KIND_CONTEXT


def kind_cluster_reachable() -> bool:
    """True when the kind context exists, answers, and has the shared claim.

    Evaluated by ``skipif`` at collection time in every kube test module, on
    every xdist worker, so the probe is cached per (kubectl, context) for the
    life of the process.

    :returns: False when kubectl is missing, the context is not in the
        kubeconfig, its API server does not report ready within a few seconds,
        or the ``gb-shared`` claim does not exist.
    """
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        return False
    return _probe(kubectl, expected_kind_context())


@functools.lru_cache(maxsize=None)
def _probe(
    kubectl: str,
    context: str,
    namespace: str = "default",
    claim: str = SHARED_CLAIM,
) -> bool:
    """Uncached check behind :func:`kind_cluster_reachable` and
    :func:`vela_cluster_reachable`: ``context`` exists, answers ``/readyz``,
    and has the ``claim`` volume claim in ``namespace``."""

    def ok(*args: str) -> bool:
        proc = subprocess.run(
            [kubectl, *args], capture_output=True, text=True, timeout=10, check=False
        )
        return proc.returncode == 0

    try:
        names = subprocess.run(
            [kubectl, "config", "get-contexts", "-o", "name"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if names.returncode != 0 or context not in names.stdout.split():
            return False
        return ok(
            "--context", context, "get", "--raw", "/readyz", "--request-timeout=3s"
        ) and ok(
            "--context",
            context,
            "-n",
            namespace,
            "get",
            "pvc",
            claim,
            "--request-timeout=3s",
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def kind_skip_reason() -> str:
    """Skip message naming what the gate wanted."""
    return (
        f"kind cluster context '{expected_kind_context()}' not reachable, or its "
        f"'{SHARED_CLAIM}' volume claim is missing (run: make kube-setup)"
    )


#: Remote (non-kind) clusters: env var overriding the context, the alias the
#: env pins in ``allowed_contexts``, the namespace, and the shared claim. OpenShift
#: names the context a login creates after the user, so each user adds the alias
#: once (the command is in each env file's header).
DEFAULT_VELA_CONTEXT = "gb-vela"
VELA_NAMESPACE = "granite-build"
#: The gb chart's ReadWriteMany claim (k8s/chart/templates/pvc-afm-read-write.yaml),
#: mounted by skypilot/kubernetes/vela with a subPath.
VELA_CLAIM = "gb-afm-read-write"
VELA_ENV_FILE = (
    "configurations/assets/environments/skypilot/kubernetes/vela/environment.yaml"
)

DEFAULT_CPU_CONTEXT = "gb-cpu"
#: The CPU cluster's namespace and shared claim are site-specific and kept out of
#: the repo (as is its environment.yaml), so they come from the environment. The
#: gate skips until both are set.
ENV_VAR_CPU_NAMESPACE = "GBTEST_SKY_CPU_NAMESPACE"
ENV_VAR_CPU_CLAIM = "GBTEST_SKY_CPU_CLAIM"
CPU_ENV_FILE = (
    "configurations/assets/environments/skypilot/kubernetes/cpu/environment.yaml"
)


def _remote_reachable(context: str, namespace: str, claim: str) -> bool:
    """The kind probe, pointed at a remote cluster's context, namespace and claim.

    False off the IBM network, with an expired login, or without the alias, so
    those skip rather than fail.
    """
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        return False
    return _probe(kubectl, context, namespace, claim)


def _remote_skip_reason(context: str, namespace: str, claim: str, env: str) -> str:
    return (
        f"context '{context}' not reachable (network/VPN, expired login, or alias "
        f"not created), or its '{claim}' volume claim is missing in namespace "
        f"'{namespace}' (see {env})"
    )


def expected_vela_context() -> str:
    """The Vela kube context the tests expect (``GBTEST_SKY_VELA_CONTEXT``)."""
    return os.environ.get("GBTEST_SKY_VELA_CONTEXT") or DEFAULT_VELA_CONTEXT


def vela_cluster_reachable() -> bool:
    """True when the Vela alias answers and has the shared claim in granite-build."""
    return _remote_reachable(expected_vela_context(), VELA_NAMESPACE, VELA_CLAIM)


def vela_skip_reason() -> str:
    """Skip message naming what the Vela gate wanted."""
    return "Vela " + _remote_skip_reason(
        expected_vela_context(), VELA_NAMESPACE, VELA_CLAIM, VELA_ENV_FILE
    )


def expected_cpu_context() -> str:
    """The CPU-cluster kube context the tests expect (``GBTEST_SKY_CPU_CONTEXT``)."""
    return os.environ.get("GBTEST_SKY_CPU_CONTEXT") or DEFAULT_CPU_CONTEXT


def cpu_namespace() -> str:
    """The CPU cluster's namespace (``GBTEST_SKY_CPU_NAMESPACE``), or ``""``."""
    return os.environ.get(ENV_VAR_CPU_NAMESPACE, "").strip()


def cpu_claim() -> str:
    """The CPU cluster's shared volume claim (``GBTEST_SKY_CPU_CLAIM``), or ``""``."""
    return os.environ.get(ENV_VAR_CPU_CLAIM, "").strip()


def cpu_cluster_reachable() -> bool:
    """True when the namespace and claim are set and the alias answers with the claim."""
    if not (cpu_namespace() and cpu_claim()):
        return False
    return _remote_reachable(expected_cpu_context(), cpu_namespace(), cpu_claim())


def cpu_skip_reason() -> str:
    """Skip message naming what the CPU-cluster gate wanted."""
    if not (cpu_namespace() and cpu_claim()):
        return (
            f"CPU cluster: set {ENV_VAR_CPU_NAMESPACE} and {ENV_VAR_CPU_CLAIM} to its "
            f"namespace and shared claim (and provide {CPU_ENV_FILE}, kept out of "
            "the repo)"
        )
    return "CPU cluster " + _remote_skip_reason(
        expected_cpu_context(), cpu_namespace(), cpu_claim(), CPU_ENV_FILE
    )
