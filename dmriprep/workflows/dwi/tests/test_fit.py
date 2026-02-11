# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""Test DWI fit workflow utilities."""

import pytest

from dmriprep.workflows.dwi.fit import _ensure_fmap_mask


def test_ensure_fmap_mask_uses_existing_file(tmp_path):
    fmap_ref = tmp_path / 'sub-01_desc-magnitude_fieldmap.nii.gz'
    fmap_ref.write_text('dummy')
    fmap_mask = tmp_path / 'sub-01_desc-brain_mask.nii.gz'
    fmap_mask.write_text('dummy')

    out_mask = _ensure_fmap_mask(str(fmap_mask), str(fmap_ref))

    assert out_mask == str(fmap_mask.absolute())


def test_ensure_fmap_mask_falls_back_to_reference(tmp_path):
    fmap_ref = tmp_path / 'sub-01_desc-magnitude_fieldmap.nii.gz'
    fmap_ref.write_text('dummy')

    out_mask = _ensure_fmap_mask('MISSING', str(fmap_ref))
    expected = str(fmap_ref.absolute())

    assert out_mask == expected


def test_ensure_fmap_mask_raises_on_ambiguous_reference():
    refs = ['sub-01_desc-magnitude1_fieldmap.nii.gz', 'sub-01_desc-magnitude2_fieldmap.nii.gz']

    with pytest.raises(ValueError, match='Expected one fieldmap reference'):
        _ensure_fmap_mask('MISSING', refs)
