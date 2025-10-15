"""Test data helpers."""

from __future__ import annotations

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
