# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""DWI preprocessing workflows."""

from dmriprep.workflows.dwi.base import init_dwi_wf
from dmriprep.workflows.dwi.fit import init_dwi_fit_wf, init_dwi_reference_wf
from dmriprep.workflows.dwi.hmc import init_dwi_hmc_wf
from dmriprep.workflows.dwi.outputs import (
    init_dwi_derivatives_wf,
    init_dwi_fit_derivatives_wf,
    init_dwi_preproc_derivatives_wf,
    init_reportlets_wf,
)
from dmriprep.workflows.dwi.registration import init_dwi_reg_wf
from dmriprep.workflows.dwi.resampling import init_dwi_native_wf, init_dwi_std_wf

__all__ = [
    'init_dwi_derivatives_wf',
    'init_dwi_fit_derivatives_wf',
    'init_dwi_fit_wf',
    'init_dwi_hmc_wf',
    'init_dwi_native_wf',
    'init_dwi_preproc_derivatives_wf',
    'init_dwi_reference_wf',
    'init_dwi_reg_wf',
    'init_dwi_std_wf',
    'init_dwi_wf',
    'init_reportlets_wf',
]
