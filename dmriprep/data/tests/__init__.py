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
"""Test data helpers."""

from functools import partial

from dipy.data.fetcher import UW_RW_URL, _make_fetcher

testdata_fetcher = partial(
    _make_fetcher,
    name='fetch_sherbrooke_3shell',
    baseurl=UW_RW_URL + '1773/38475/',
    remote_fnames=['HARDI193.nii.gz', 'HARDI193.bval', 'HARDI193.bvec'],
    local_fnames=['HARDI193.nii.gz', 'HARDI193.bval', 'HARDI193.bvec'],
    md5_list=[
        '0b735e8f16695a37bfbd66aab136eb66',
        'e9b9bb56252503ea49d31fb30a0ac637',
        '0c83f7e8b917cd677ad58a078658ebb7',
    ],
    doc='Download a 3shell HARDI dataset with 192 gradient direction',
)

__all__ = ['testdata_fetcher']
