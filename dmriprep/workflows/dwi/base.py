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
"""Orchestrating the dMRI-preprocessing workflow."""

from pathlib import Path

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from dmriprep import config
from dmriprep.utils.misc import estimate_image_mem_usage

DEFAULT_MIN_DWI_SIZE = 7  # 6 DWI directions + 1 b=0, for DTI


def init_dwi_wf(
    *,
    dwi_series: list[str],
    precomputed: dict | None = None,
    fieldmap_id: str | None = None,
    jacobian: bool = False,
    min_dwi_vols: int = DEFAULT_MIN_DWI_SIZE,
) -> pe.Workflow:
    """
    Build a preprocessing workflow for one DWI run.

    This workflow follows the fit/transform architecture, where:

    - **Fit stage** (``--level minimal``): Estimates all transforms without applying them,
      including head motion correction, eddy current correction, susceptibility
      distortion correction, and coregistration to anatomical space.

    - **Transform stage** (``--level resampling`` or ``--level full``): Composes all
      estimated transforms and applies them in a single interpolation step to
      minimize blurring.

    Parameters
    ----------
    dwi_series
        List of paths to NIfTI files.
    precomputed
        Dictionary containing precomputed derivatives to reuse, if possible.
    fieldmap_id
        ID of the fieldmap to use to correct this DWI series. If :obj:`None`,
        no correction will be applied.
    jacobian
        Whether to apply Jacobian modulation during SDC.
    min_dwi_vols
        Minimum number of volumes required to process.

    Inputs
    ------
    t1w_preproc
        Preprocessed T1w image.
    t1w_mask
        Brain mask in T1w space.
    t1w_dseg
        Tissue segmentation in T1w space.
    t1w_tpms
        Tissue probability maps.
    subjects_dir
        FreeSurfer subjects directory.
    subject_id
        FreeSurfer subject ID.
    fsnative2t1w_xfm
        Transform from FreeSurfer native to T1w space.
    fmap, fmap_ref, fmap_coeff, fmap_mask
        Fieldmap-related inputs.
    fmap_id
        Fieldmap estimator ID.
    sdc_method
        SDC method identifier.
    anat2std_xfm
        Anatomical to standard space transform(s).
    std_t1w, std_mask
        Standard space reference and mask.

    Outputs
    -------
    hmc_dwiref
        3D b=0 reference for motion correction.
    coreg_dwiref
        SDC-corrected reference for coregistration.
    dwi_mask
        Brain mask in DWI space.
    motion_xfm
        Per-volume motion transforms.
    dwiref2anat_xfm
        DWI-to-anatomical coregistration.
    dwi_preproc
        Preprocessed DWI (if level > minimal).

    See Also
    --------
    * :py:func:`~dmriprep.workflows.dwi.fit.init_dwi_fit_wf`
    * :py:func:`~dmriprep.workflows.dwi.resampling.init_dwi_native_wf`
    * :py:func:`~dmriprep.workflows.dwi.outputs.init_dwi_fit_derivatives_wf`

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from dmriprep.workflows.dwi.fit import init_dwi_fit_wf
    from dmriprep.workflows.dwi.outputs import (
        init_dwi_fit_derivatives_wf,
        init_dwi_preproc_derivatives_wf,
        init_reportlets_wf,
    )

    if precomputed is None:
        precomputed = {}
    dwi_file = dwi_series[0]

    metadata = config.execution.layout.get_metadata(dwi_file)
    omp_nthreads = config.nipype.omp_nthreads

    nvols, mem_gb = estimate_image_mem_usage(dwi_file)
    if nvols <= min_dwi_vols - config.execution.sloppy:
        config.loggers.workflow.warning(
            f'Too short DWI series (<= {min_dwi_vols} timepoints). Skipping processing of <{dwi_file}>.'
        )
        return

    config.loggers.workflow.debug(
        f'Creating dMRI preprocessing workflow for <{dwi_file}> ({mem_gb["filesize"]:.2f} GB / {nvols} volumes). '
        f'Memory resampled/largemem={mem_gb["resampled"]:.2f}/{mem_gb["largemem"]:.2f} GB.'
    )

    workflow = Workflow(name=_get_wf_name(dwi_file, 'dwi'))
    workflow.__postdesc__ = """\
All resamplings can be performed with *a single interpolation
step* by composing all the pertinent transformations (i.e. head-motion
transform matrices, eddy-current distortion correction, susceptibility
distortion correction when available, and co-registrations to anatomical
and output spaces).
Gridded (volumetric) resamplings were performed using `nitransforms`,
configured with cubic B-spline interpolation.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Anatomical coregistration
                't1w_preproc',
                't1w_mask',
                't1w_dseg',
                't1w_tpms',
                # FreeSurfer outputs
                'subjects_dir',
                'subject_id',
                'fsnative2t1w_xfm',
                'white',
                'midthickness',
                'pial',
                'sphere_reg_fsLR',
                'midthickness_fsLR',
                'cortex_mask',
                'anat_ribbon',
                # Fieldmap registration
                'fmap',
                'fmap_ref',
                'fmap_coeff',
                'fmap_mask',
                'fmap_id',
                'sdc_method',
                # Volumetric templates
                'anat2std_xfm',
                'std_t1w',
                'std_mask',
                'std_space',
                'std_resolution',
                'std_cohort',
                # MNI152NLin6Asym warp, for CIFTI use
                'anat2mni6_xfm',
                'mni6_mask',
                # MNI152NLin2009cAsym inverse warp, for carpetplotting
                'mni2009c2anat_xfm',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'hmc_dwiref',
                'coreg_dwiref',
                'dwi_mask',
                'motion_xfm',
                'dwiref2anat_xfm',
                'out_bvec',
                'out_bval',
                'dwi_preproc',
            ],
        ),
        name='outputnode',
    )

    #
    # Fit stage (--level minimal)
    #
    dwi_fit_wf = init_dwi_fit_wf(
        dwi_file=dwi_file,
        precomputed=precomputed,
        fieldmap_id=fieldmap_id,
        omp_nthreads=omp_nthreads,
    )

    workflow.connect([
        (inputnode, dwi_fit_wf, [
            ('t1w_preproc', 'inputnode.t1w_preproc'),
            ('t1w_mask', 'inputnode.t1w_mask'),
            ('t1w_dseg', 'inputnode.t1w_dseg'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
            ('fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm'),
            ('fmap', 'inputnode.fmap'),
            ('fmap_ref', 'inputnode.fmap_ref'),
            ('fmap_coeff', 'inputnode.fmap_coeff'),
            ('fmap_mask', 'inputnode.fmap_mask'),
            ('fmap_id', 'inputnode.fmap_id'),
            ('sdc_method', 'inputnode.sdc_method'),
        ]),
        (dwi_fit_wf, outputnode, [
            ('outputnode.hmc_dwiref', 'hmc_dwiref'),
            ('outputnode.coreg_dwiref', 'coreg_dwiref'),
            ('outputnode.dwi_mask', 'dwi_mask'),
            ('outputnode.motion_xfm', 'motion_xfm'),
            ('outputnode.dwiref2anat_xfm', 'dwiref2anat_xfm'),
            ('outputnode.out_bvec', 'out_bvec'),
            ('outputnode.out_bval', 'out_bval'),
        ]),
    ])  # fmt:skip

    # Save fit-stage derivatives
    fit_derivatives_wf = init_dwi_fit_derivatives_wf(
        output_dir=str(config.execution.dmriprep_dir),
        fieldmap_id=fieldmap_id,
    )
    fit_derivatives_wf.inputs.inputnode.source_file = dwi_file

    workflow.connect([
        (dwi_fit_wf, fit_derivatives_wf, [
            ('outputnode.hmc_dwiref', 'inputnode.hmc_dwiref'),
            ('outputnode.coreg_dwiref', 'inputnode.coreg_dwiref'),
            ('outputnode.dwi_mask', 'inputnode.dwi_mask'),
            ('outputnode.motion_xfm', 'inputnode.motion_xfm'),
            ('outputnode.dwiref2anat_xfm', 'inputnode.dwiref2anat_xfm'),
            ('outputnode.dwiref2fmap_xfm', 'inputnode.dwiref2fmap_xfm'),
            ('outputnode.fmap_coeff', 'inputnode.fmap_coeff'),
            ('outputnode.out_bvec', 'inputnode.out_bvec'),
            ('outputnode.out_bval', 'inputnode.out_bval'),
        ]),
    ])  # fmt:skip

    # Reportlets
    reportlets_wf = init_reportlets_wf(
        output_dir=str(config.execution.dmriprep_dir),
        sdc_report=fieldmap_id is not None,
    )
    reportlets_wf.inputs.inputnode.source_file = dwi_file

    workflow.connect([
        (dwi_fit_wf, reportlets_wf, [
            ('outputnode.hmc_dwiref', 'inputnode.dwi_ref'),
            ('outputnode.dwi_mask', 'inputnode.dwi_mask'),
        ]),
    ])  # fmt:skip

    if config.workflow.level == 'minimal':
        return workflow

    #
    # Resampling stage (--level resampling)
    #
    from dmriprep.workflows.dwi.resampling import init_dwi_native_wf

    native_wf = init_dwi_native_wf(
        fieldmap_id=fieldmap_id,
        jacobian=jacobian,
        omp_nthreads=omp_nthreads,
    )
    native_wf.inputs.inputnode.dwi_file = dwi_file
    native_wf.inputs.inputnode.metadata = metadata

    workflow.connect([
        (dwi_fit_wf, native_wf, [
            ('outputnode.hmc_dwiref', 'inputnode.hmc_dwiref'),
            ('outputnode.dwi_mask', 'inputnode.dwi_mask'),
            ('outputnode.motion_xfm', 'inputnode.motion_xfm'),
            ('outputnode.fmap_coeff', 'inputnode.fmap_coeff'),
        ]),
        (native_wf, outputnode, [('outputnode.dwi_preproc', 'dwi_preproc')]),
    ])  # fmt:skip

    # Save native-space derivatives
    native_derivatives_wf = init_dwi_preproc_derivatives_wf(
        output_dir=str(config.execution.dmriprep_dir),
        space='orig',
    )
    native_derivatives_wf.inputs.inputnode.source_file = dwi_file

    workflow.connect([
        (native_wf, native_derivatives_wf, [
            ('outputnode.dwi_preproc', 'inputnode.dwi_preproc'),
            ('outputnode.dwi_ref', 'inputnode.dwi_ref'),
        ]),
        (dwi_fit_wf, native_derivatives_wf, [
            ('outputnode.dwi_mask', 'inputnode.dwi_mask'),
            ('outputnode.out_bvec', 'inputnode.out_bvec'),
            ('outputnode.out_bval', 'inputnode.out_bval'),
        ]),
    ])  # fmt:skip

    if config.workflow.level == 'resampling':
        return workflow

    #
    # Full workflow (--level full): Add T1w space resampling
    #
    from dmriprep.workflows.dwi.resampling import init_dwi_std_wf

    t1w_wf = init_dwi_std_wf(
        fieldmap_id=fieldmap_id,
        jacobian=jacobian,
        omp_nthreads=omp_nthreads,
        name='dwi_t1w_wf',
    )
    t1w_wf.inputs.inputnode.dwi_file = dwi_file
    t1w_wf.inputs.inputnode.metadata = metadata

    workflow.connect([
        (inputnode, t1w_wf, [
            ('t1w_preproc', 'inputnode.t1w_preproc'),
            ('t1w_mask', 'inputnode.t1w_mask'),
        ]),
        (dwi_fit_wf, t1w_wf, [
            ('outputnode.hmc_dwiref', 'inputnode.hmc_dwiref'),
            ('outputnode.motion_xfm', 'inputnode.motion_xfm'),
            ('outputnode.dwiref2anat_xfm', 'inputnode.dwiref2anat_xfm'),
            ('outputnode.fmap_coeff', 'inputnode.fmap_coeff'),
        ]),
    ])  # fmt:skip

    # Save T1w-space derivatives
    t1w_derivatives_wf = init_dwi_preproc_derivatives_wf(
        output_dir=str(config.execution.dmriprep_dir),
        space='T1w',
        name='dwi_t1w_derivatives_wf',
    )
    t1w_derivatives_wf.inputs.inputnode.source_file = dwi_file

    workflow.connect([
        (t1w_wf, t1w_derivatives_wf, [
            ('outputnode.dwi_preproc', 'inputnode.dwi_preproc'),
            ('outputnode.dwi_ref', 'inputnode.dwi_ref'),
            ('outputnode.dwi_mask', 'inputnode.dwi_mask'),
        ]),
        (dwi_fit_wf, t1w_derivatives_wf, [
            ('outputnode.out_bvec', 'inputnode.out_bvec'),
            ('outputnode.out_bval', 'inputnode.out_bval'),
        ]),
    ])  # fmt:skip

    return workflow


def _get_wf_name(filename, prefix='dwi'):
    """
    Derive the workflow name for supplied DWI file.

    Examples
    --------
    >>> _get_wf_name("/completely/made/up/path/sub-01_dir-AP_acq-64grad_dwi.nii.gz")
    'dwi_dir_AP_acq_64grad_wf'

    >>> _get_wf_name("/completely/made/up/path/sub-01_dir-RL_run-01_echo-1_dwi.nii.gz")
    'dwi_dir_RL_run_01_echo_1_wf'

    """

    fname = Path(filename).name.rpartition('.nii')[0].replace('_dwi', '_wf')
    fname_nosub = '_'.join(fname.split('_')[1:])
    return f'{prefix}_{fname_nosub.replace(".", "_").replace(" ", "").replace("-", "_")}'
