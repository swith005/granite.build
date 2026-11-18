# SPDX-License-Identifier: Apache-2.0

"""Unit tests for libgbtest.kube.kind_cluster_reachable, the skypilot/kubernetes skip gate.

Runs the gate against a stub ``kubectl`` on PATH, so no cluster is needed. The gate
admits a run only when the pinned kind context exists, answers, and has the
shared-volume claim; it no longer cares which context is CURRENT, because
skypilot/kubernetes/local pins SkyPilot to the kind context itself.
"""

import pathlib
import shutil
import stat

import pytest
from libgbtest import kube

_BASH = shutil.which("bash") or "/bin/bash"


def _stub_kubectl(
    tmp_path: pathlib.Path,
    contexts: tuple[str, ...] = ("prod-cluster", "kind-skypilot"),
    ready_rc: int = 0,
    pvc_rc: int = 0,
    calls: pathlib.Path | None = None,
) -> None:
    """A kubectl that lists ``contexts``, answers /readyz and `get pvc` with the rcs."""
    script = tmp_path / "kubectl"
    log = f'echo "$*" >> {calls}\n' if calls else ""
    listing = "\\n".join(contexts)
    # Absolute interpreter: PATH is narrowed to tmp_path, so `env bash` would fail.
    script.write_text(
        f"#!{_BASH}\n"
        f"{log}"
        'case "$*" in\n'
        f'  "config get-contexts -o name") printf "{listing}\\n" ;;\n'
        f"  *readyz*) exit {ready_rc} ;;\n"
        f'  *" get pvc "*) exit {pvc_rc} ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def stub_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("GBTEST_SKY_KUBE_CONTEXT", raising=False)
    monkeypatch.delenv("GBTEST_SKY_VELA_CONTEXT", raising=False)
    monkeypatch.delenv("GBTEST_SKY_CPU_CONTEXT", raising=False)
    monkeypatch.delenv("GBTEST_SKY_CPU_NAMESPACE", raising=False)
    monkeypatch.delenv("GBTEST_SKY_CPU_CLAIM", raising=False)
    return tmp_path


class TestKindClusterGate:
    def test_ready_context_with_claim_passes(self, stub_path):
        _stub_kubectl(stub_path)
        assert kube.kind_cluster_reachable()

    def test_another_current_context_does_not_matter(self, stub_path):
        # The stub's contexts include prod-cluster; which one is CURRENT is never
        # consulted: the env pins kind-skypilot via allowed_contexts.
        _stub_kubectl(stub_path, contexts=("prod-cluster", "kind-skypilot"))
        assert kube.kind_cluster_reachable()

    def test_missing_kind_context_is_refused(self, stub_path):
        _stub_kubectl(stub_path, contexts=("prod-cluster",))
        assert not kube.kind_cluster_reachable()

    def test_unready_api_server_is_refused(self, stub_path):
        _stub_kubectl(stub_path, ready_rc=1)
        assert not kube.kind_cluster_reachable()

    def test_missing_shared_claim_is_refused(self, stub_path):
        # Without the claim the env's pods would sit Pending; skip instead.
        _stub_kubectl(stub_path, pvc_rc=1)
        assert not kube.kind_cluster_reachable()
        assert "gb-shared" in kube.kind_skip_reason()

    def test_missing_kubectl_is_refused(self, stub_path):
        assert not kube.kind_cluster_reachable()

    def test_context_override(self, stub_path, monkeypatch):
        monkeypatch.setenv("GBTEST_SKY_KUBE_CONTEXT", "kind-dpk")
        _stub_kubectl(stub_path, contexts=("kind-dpk",))
        assert kube.kind_cluster_reachable()
        assert "kind-dpk" in kube.kind_skip_reason()

    def test_default_context_is_the_sky_local_up_one(self, stub_path):
        _stub_kubectl(stub_path, contexts=("kind-other",))
        assert kube.expected_kind_context() == "kind-skypilot"
        assert not kube.kind_cluster_reachable()


class TestProbeIsCached:
    """The gate runs from ``skipif`` in every kube module on every xdist worker."""

    def test_repeated_checks_call_kubectl_once(self, stub_path):
        calls = stub_path / "calls"
        _stub_kubectl(stub_path, calls=calls)
        assert all(kube.kind_cluster_reachable() for _ in range(5))
        # get-contexts + /readyz + get pvc once, not five times each.
        assert len(calls.read_text().splitlines()) == 3

    def test_shared_marker_is_the_skypilot_group(self):
        """Same group as every skypilot_integration test, so kube never overlaps slurm."""
        assert kube.KUBE_XDIST_GROUP.name == "xdist_group"
        assert kube.KUBE_XDIST_GROUP.kwargs == {"name": "skypilot"}


class TestVelaClusterGate:
    """libgbtest.kube.vela_cluster_reachable: same probe, Vela context + namespace."""

    def test_ready_context_with_claim_passes(self, stub_path):
        _stub_kubectl(stub_path, contexts=("kind-skypilot", "gb-vela"))
        assert kube.vela_cluster_reachable()

    def test_claim_is_looked_up_in_granite_build(self, stub_path):
        calls = stub_path / "calls"
        _stub_kubectl(stub_path, contexts=("gb-vela",), calls=calls)
        assert kube.vela_cluster_reachable()
        assert "--context gb-vela -n granite-build get pvc gb-afm-read-write" in (
            calls.read_text()
        )

    def test_kind_gate_still_checks_default_namespace(self, stub_path):
        calls = stub_path / "calls"
        _stub_kubectl(stub_path, calls=calls)
        assert kube.kind_cluster_reachable()
        assert "--context kind-skypilot -n default get pvc gb-shared" in (
            calls.read_text()
        )

    def test_missing_alias_is_refused(self, stub_path):
        # A raw `oc login` context alone is not enough: the env pins the alias.
        _stub_kubectl(stub_path, contexts=("/vela-api:6443/someone@ibm.com",))
        assert not kube.vela_cluster_reachable()
        assert "gb-vela" in kube.vela_skip_reason()

    def test_unreachable_api_server_is_refused(self, stub_path):
        # Off VPN, or an expired oc login token.
        _stub_kubectl(stub_path, contexts=("gb-vela",), ready_rc=1)
        assert not kube.vela_cluster_reachable()

    def test_missing_shared_claim_is_refused(self, stub_path):
        _stub_kubectl(stub_path, contexts=("gb-vela",), pvc_rc=1)
        assert not kube.vela_cluster_reachable()
        assert "granite-build" in kube.vela_skip_reason()

    def test_context_override(self, stub_path, monkeypatch):
        monkeypatch.setenv("GBTEST_SKY_VELA_CONTEXT", "vela-mine")
        _stub_kubectl(stub_path, contexts=("vela-mine",))
        assert kube.vela_cluster_reachable()
        assert "vela-mine" in kube.vela_skip_reason()

    def test_missing_kubectl_is_refused(self, stub_path):
        assert not kube.vela_cluster_reachable()


class TestCpuClusterGate:
    """libgbtest.kube.cpu_cluster_reachable: gb-cpu alias, plus a namespace and claim
    that come from the environment (the real ones are kept out of the repo)."""

    @pytest.fixture
    def cpu_site(self, stub_path, monkeypatch):
        monkeypatch.setenv("GBTEST_SKY_CPU_NAMESPACE", "cpu-ns")
        monkeypatch.setenv("GBTEST_SKY_CPU_CLAIM", "cpu-claim")
        return stub_path

    def test_ready_context_with_claim_passes(self, cpu_site):
        _stub_kubectl(cpu_site, contexts=("gb-vela", "gb-cpu"))
        assert kube.cpu_cluster_reachable()

    def test_claim_is_looked_up_in_the_configured_namespace(self, cpu_site):
        calls = cpu_site / "calls"
        _stub_kubectl(cpu_site, contexts=("gb-cpu",), calls=calls)
        assert kube.cpu_cluster_reachable()
        assert "--context gb-cpu -n cpu-ns get pvc cpu-claim" in calls.read_text()

    @pytest.mark.parametrize(
        "unset", ["GBTEST_SKY_CPU_NAMESPACE", "GBTEST_SKY_CPU_CLAIM"]
    )
    def test_unset_site_config_skips_without_calling_kubectl(
        self, cpu_site, monkeypatch, unset
    ):
        calls = cpu_site / "calls"
        _stub_kubectl(cpu_site, contexts=("gb-cpu",), calls=calls)
        monkeypatch.delenv(unset)
        assert not kube.cpu_cluster_reachable()
        assert not calls.exists()
        assert unset in kube.cpu_skip_reason()

    def test_missing_alias_is_refused(self, cpu_site):
        _stub_kubectl(cpu_site, contexts=("gb-vela",))
        assert not kube.cpu_cluster_reachable()
        assert "gb-cpu" in kube.cpu_skip_reason()

    def test_missing_claim_is_refused(self, cpu_site):
        _stub_kubectl(cpu_site, contexts=("gb-cpu",), pvc_rc=1)
        assert not kube.cpu_cluster_reachable()
        assert "cpu-claim" in kube.cpu_skip_reason()

    def test_context_override(self, cpu_site, monkeypatch):
        monkeypatch.setenv("GBTEST_SKY_CPU_CONTEXT", "cpu-mine")
        _stub_kubectl(cpu_site, contexts=("cpu-mine",))
        assert kube.cpu_cluster_reachable()
        assert "cpu-mine" in kube.cpu_skip_reason()
