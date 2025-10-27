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
"""Utilities to handle BIDS inputs."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path

from bids.layout import BIDSLayout
from bids.utils import listify

from dmriprep.data import load as load_data

DEFAULT_BIDS_IGNORE = {
    '*.html',
    'logs/',
    'figures/',  # Reports
    '*_xfm.*',  # Unspecified transform files
    '*.surf.gii',  # Unspecified structural outputs
    # Unspecified diffusion outputs
    '*_dwiref.nii.gz',
}


@cache
def _get_layout(derivatives_dir: Path) -> BIDSLayout:
    import niworkflows.data

    return BIDSLayout(
        derivatives_dir, config=[niworkflows.data.load('nipreps.json')], validate=False
    )


def collect_derivatives(
    derivatives_dir: Path,
    entities: dict,
    fieldmap_id: str | None = None,
    spec: dict | None = None,
    patterns: list[str] | None = None,
):
    """Gather existing derivatives and compose a cache."""
    if spec is None or patterns is None:
        _spec, _patterns = tuple(
            json.loads(load_data.readable('io_spec.json').read_text()).values()
        )

        if spec is None:
            spec = _spec
        if patterns is None:
            patterns = _patterns

    derivs_cache = defaultdict(list, {})
    layout = _get_layout(derivatives_dir)

    # search for both dwirefs
    for key, query in spec['baseline'].items():
        item = layout.get(return_type='filename', **{**entities, **query})
        if not item:
            continue
        derivs_cache[f'{key}_dwiref'] = item[0] if len(item) == 1 else item

    transforms_cache: dict[str, list | str] = {}
    for name, query in spec.get('transforms', {}).items():
        if name == 'dwi2fmap' and fieldmap_id:
            query = {**query, 'to': re.sub(r'[^a-zA-Z0-9]', '', fieldmap_id)}
        item = layout.get(return_type='filename', **{**entities, **query})
        if not item:
            continue
        transforms_cache[name] = item[0] if len(item) == 1 else item

    derivs_cache['transforms'] = transforms_cache
    return derivs_cache


def extract_entities(file_list):
    """
    Return a dictionary of common entities given a list of files.

    Examples
    --------
    >>> extract_entities("sub-01/anat/sub-01_T1w.nii.gz")
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_T1w.nii.gz"] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_run-1_T1w.nii.gz",
    ...                   "sub-01/anat/sub-01_run-2_T1w.nii.gz"])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}

    """
    from collections import defaultdict

    from bids.layout import parse_file_entities

    entities: dict[str, list] = defaultdict(list)
    for entity, value in [
        pair for fname in listify(file_list) for pair in parse_file_entities(fname).items()
    ]:
        entities[entity].append(value)

    def _unique(values):
        values = sorted(set(values))
        if len(values) == 1:
            return values[0]
        return values

    return {key: _unique(val) for key, val in entities.items()}


def write_derivative_description(bids_dir, deriv_dir):
    from ..__about__ import DOWNLOAD_URL, __url__, __version__

    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        'Name': 'dMRIPrep - dMRI PREProcessing workflow',
        'BIDSVersion': '1.1.1',
        'PipelineDescription': {
            'Name': 'dMRIPrep',
            'Version': __version__,
            'CodeURL': DOWNLOAD_URL,
        },
        'CodeURL': __url__,
        'HowToAcknowledge': 'Please cite https://doi.org/10.5281/zenodo.3392201.',
    }

    # Keys that can only be set by environment
    if 'DMRIPREP_DOCKER_TAG' in os.environ:
        desc['DockerHubContainerTag'] = os.environ['DMRIPREP_DOCKER_TAG']
    if 'DMRIPREP_SINGULARITY_URL' in os.environ:
        singularity_url = os.environ['DMRIPREP_SINGULARITY_URL']
        desc['SingularityContainerURL'] = singularity_url

        singularity_md5 = _get_shub_version(singularity_url)
        if singularity_md5 and singularity_md5 is not NotImplemented:
            desc['SingularityContainerMD5'] = _get_shub_version(singularity_url)

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / 'dataset_description.json'
    if fname.exists():
        with fname.open() as fobj:
            orig_desc = json.load(fobj)

    if 'DatasetDOI' in orig_desc:
        desc['SourceDatasetsURLs'] = [f'https://doi.org/{orig_desc["DatasetDOI"]}']
    if 'License' in orig_desc:
        desc['License'] = orig_desc['License']

    with (deriv_dir / 'dataset_description.json').open('w') as fobj:
        json.dump(desc, fobj, indent=4)


def validate_input_dir(exec_env, bids_dir, participant_label, need_T1w=True):
    # Ignore issues and warnings that should not influence dMRIPrep
    import subprocess
    import tempfile

    validator_config_dict = {
        'ignore': [
            'EVENTS_COLUMN_ONSET',
            'EVENTS_COLUMN_DURATION',
            'TSV_EQUAL_ROWS',
            'TSV_EMPTY_CELL',
            'TSV_IMPROPER_NA',
            'INCONSISTENT_SUBJECTS',
            'INCONSISTENT_PARAMETERS',
            'PARTICIPANT_ID_COLUMN',
            'PARTICIPANT_ID_MISMATCH',
            'TASK_NAME_MUST_DEFINE',
            'PHENOTYPE_SUBJECTS_MISSING',
            'STIMULUS_FILE_MISSING',
            'BOLD_NOT_4D',
            'EVENTS_TSV_MISSING',
            'TSV_IMPROPER_NA',
            'ACQTIME_FMT',
            'Participants age 89 or higher',
            'DATASET_DESCRIPTION_JSON_MISSING',
            'TASK_NAME_CONTAIN_ILLEGAL_CHARACTER',
            'FILENAME_COLUMN',
            'WRONG_NEW_LINE',
            'MISSING_TSV_COLUMN_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_ELECTRODES',
            'UNUSED_STIMULUS',
            'CHANNELS_COLUMN_SFREQ',
            'CHANNELS_COLUMN_LOWCUT',
            'CHANNELS_COLUMN_HIGHCUT',
            'CHANNELS_COLUMN_NOTCH',
            'CUSTOM_COLUMN_WITHOUT_DESCRIPTION',
            'ACQTIME_FMT',
            'SUSPICIOUSLY_LONG_EVENT_DESIGN',
            'SUSPICIOUSLY_SHORT_EVENT_DESIGN',
            'MISSING_TSV_COLUMN_EEG_ELECTRODES',
            'MISSING_SESSION',
        ],
        'error': ['NO_T1W'] if need_T1w else [],
        'ignoredFiles': ['/dataset_description.json', '/participants.tsv'],
    }
    # Limit validation only to data from requested participants
    if participant_label:
        all_subs = {s.name[4:] for s in bids_dir.glob('sub-*')}
        selected_subs = {s.removeprefix('sub-') for s in participant_label}
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            error_msg = (
                'Data for requested participant(s) label(s) not found. Could '
                'not find data for participant(s): %s. Please verify the requested '
                'participant labels.'
            )
            if exec_env == 'docker':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the docker container. Please make sure all '
                    'volumes are mounted properly (see https://docs.docker.com/'
                    'engine/reference/commandline/run/#mount-volume--v---read-only)'
                )
            if exec_env == 'singularity':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the singularity container. Please make sure '
                    'all paths are mapped properly (see https://www.sylabs.io/'
                    'guides/3.0/user-guide/bind_paths_and_mounts.html)'
                )
            raise RuntimeError(error_msg % ','.join(bad_labels))

        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                validator_config_dict['ignoredFiles'].append(f'/sub-{sub}/**')
    with tempfile.NamedTemporaryFile('w+') as temp:
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        try:
            subprocess.check_call(['bids-validator', bids_dir, '-c', temp.name])  # noqa: S607
        except FileNotFoundError:
            print('bids-validator does not appear to be installed', file=sys.stderr)


def write_bidsignore(deriv_dir, bids_ignore=DEFAULT_BIDS_IGNORE):
    (Path(deriv_dir) / '.bidsignore').write_text('\n'.join(bids_ignore) + '\n')


def _get_shub_version(singularity_url):
    return NotImplemented
