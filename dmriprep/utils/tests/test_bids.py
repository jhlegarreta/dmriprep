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
"""Test BIDS utilities."""

from dmriprep.utils import bids


class _MockLayout:
    def __init__(self):
        self._matches = {
            'full': {
                'fieldmap': ['sub-01_desc-preproc_fieldmap.nii.gz'],
                'magnitude': ['sub-01_desc-magnitude_fieldmap.nii.gz'],
                'coeffs': [
                    'sub-01_desc-coeff0_fieldmap.nii.gz',
                    'sub-01_desc-coeff1_fieldmap.nii.gz',
                ],
            },
            'partial': {
                'fieldmap': ['sub-01_desc-preproc_fieldmap.nii.gz'],
                'magnitude': ['sub-01_desc-magnitude_fieldmap.nii.gz'],
            },
        }

    def get_fmapids(self, **entities):
        return ('full', 'partial')

    def get(self, return_type, fmapid, **query):
        assert return_type == 'filename'
        mapping = {
            'preproc': 'fieldmap',
            'magnitude': 'magnitude',
            'epi': 'magnitude',
            'coeff': 'coeffs',
            'coeff0': 'coeffs',
            'coeff1': 'coeffs',
        }

        query_desc = query.get('desc')
        if isinstance(query_desc, list):
            key = mapping[query_desc[0]]
        else:
            key = mapping[query_desc]

        return self._matches.get(fmapid, {}).get(key, [])


class _MockMissingCoeffStarLayout:
    def get_fmapids(self, **entities):
        return ('missing_coeffstar',)

    def get(self, return_type, fmapid, **query):
        assert return_type == 'filename'
        assert fmapid == 'missing_coeffstar'

        query_desc = query.get('desc')
        if query_desc == 'preproc':
            return ['sub-01_desc-preproc_fieldmap.nii.gz']
        if isinstance(query_desc, list) and set(query_desc) == {'magnitude', 'epi'}:
            return ['sub-01_desc-magnitude_fieldmap.nii.gz']
        if isinstance(query_desc, list) and set(query_desc) == {'coeff', 'coeff0', 'coeff1'}:
            return []

        raise RuntimeError(f'Unexpected query: {query}')


def test_collect_fieldmaps_ignores_incomplete_sets(monkeypatch, tmp_path):
    spec = {
        'fieldmaps': {
            'fieldmap': {'desc': 'preproc'},
            'coeffs': {'desc': ['coeff', 'coeff0', 'coeff1']},
            'magnitude': {'desc': ['magnitude', 'epi']},
        }
    }

    monkeypatch.setattr(bids, '_get_layout', lambda *args, **kwargs: _MockLayout())

    out = bids.collect_fieldmaps(tmp_path / 'derivatives', entities={'subject': '01'}, spec=spec)

    assert sorted(out) == ['full']
    assert out['full']['fieldmap'] == 'sub-01_desc-preproc_fieldmap.nii.gz'
    assert out['full']['magnitude'] == 'sub-01_desc-magnitude_fieldmap.nii.gz'
    assert out['full']['coeffs'] == [
        'sub-01_desc-coeff0_fieldmap.nii.gz',
        'sub-01_desc-coeff1_fieldmap.nii.gz',
    ]


def test_collect_fieldmaps_ignores_missing_coeffstar(monkeypatch, tmp_path):
    spec = {
        'fieldmaps': {
            'fieldmap': {'desc': 'preproc'},
            'coeffs': {'desc': ['coeff', 'coeff0', 'coeff1']},
            'magnitude': {'desc': ['magnitude', 'epi']},
        }
    }

    monkeypatch.setattr(bids, '_get_layout', lambda *args, **kwargs: _MockMissingCoeffStarLayout())

    out = bids.collect_fieldmaps(tmp_path / 'derivatives', entities={'subject': '01'}, spec=spec)

    assert out == {}
