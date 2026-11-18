#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Command step on the Vela OpenShift cluster via SkyPilot/kubernetes, HF URI input.

The Vela copy of
``test/integration/standalone/buildrunner/skypilot_kube/test_skypilot_kube_1step.py``:

  HF URI input  ->  command step in a pod on Vela (granite-build)  ->  hf:// + env:// outputs

Runs on ``space://environments/skypilot/kubernetes/vela``, the same shared-volume
model as the kind env: a separate hfpull pod downloads the ``hf://`` input into the
``gb-shared`` claim every pod mounts at /shared, and the command step (a different
pod) fails unless it can see it. ``step_count: 5`` pins the separate pull and the
three pushes (one per artifact marker).

The hf:// outputs are pushed by hfpush pods and, through
:class:`libgbtest.hfpush.HfPushCleanupMixin`, named per run in
``GBTEST_HF_PUSH_ORG`` (default ``ibm-research``), checked on the Hub, and
deleted, so it needs an ``HF_TOKEN`` that can write that org.

Requires the ``gb-vela`` kube context (one-time setup in the env file's header),
a current ``oc login``, the IBM network, and the ``gb-shared`` claim in
``granite-build``; auto-skips otherwise, and always in CI/CD.

The fixture's build.yaml and buildtest.yaml live in the directory returned by
_get_yaml_spec_dir below.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only
from libgbtest.hfpush import HfPushCleanupMixin
from libgbtest.kube import KUBE_XDIST_GROUP, vela_cluster_reachable, vela_skip_reason

pytestmark = pytest.mark.ibm


# Same xdist group as the kind tests: every SkyPilot env deep-merges its
# cloud_config into the one global ~/.sky/config.yaml, so a Vela and a kind build
# running at once would overwrite each other's allowed_contexts.
@KUBE_XDIST_GROUP
@extended_testing_only
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD: no Vela context or IBM network there",
)
@pytest.mark.skipif(not vela_cluster_reachable(), reason=vela_skip_reason())
class TestSkypilotKubeVela1Step(HfPushCleanupMixin, AbstractYamlBuildRunnerTest):
    """HF input -> command step in a pod on Vela via SkyPilot/kubernetes -> hf:// outputs."""

    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step"
