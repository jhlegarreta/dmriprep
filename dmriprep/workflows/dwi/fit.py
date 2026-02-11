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
"""
DWI fit workflows.

The fit stage estimates all transforms without applying them, following
the fit/transform architecture from fMRIPrep. This enables:

1. Single-interpolation resampling by composing all transforms
2. Reuse of precomputed derivatives via --derivatives flag
3. Separation of compute-intensive estimation from I/O-intensive resampling

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from dmriprep import config


def init_dwi_fit_wf(
    *,
    dwi_file: str,
    precomputed: dict | None = None,
    fieldmap_id: str | None = None,
    omp_nthreads: int = 1,
) -> Workflow:
    """
    Build a workflow to estimate all transforms for DWI preprocessing.

    This workflow orchestrates the "fit" stage of DWI preprocessing,
    estimating head motion, eddy current distortions, susceptibility
    distortion corrections, and anatomical coregistration without
    applying any transforms to the data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from dmriprep.config.testing import mock_config
            from dmriprep.workflows.dwi.fit import init_dwi_fit_wf
            with mock_config():
                wf = init_dwi_fit_wf(dwi_file='sub-01_dwi.nii.gz')

    Parameters
    ----------
    dwi_file
        Path to the DWI NIfTI file.
    precomputed
        Dictionary containing precomputed derivatives to reuse.
    fieldmap_id
        ID of the fieldmap to use for SDC. If None, no SDC is performed.
    omp_nthreads
        Number of threads for parallel processing.

    Inputs
    ------
    dwi_file
        DWI NIfTI file.
    in_bvec
        File path of the b-vectors.
    in_bval
        File path of the b-values.
    t1w_preproc
        Preprocessed T1w image.
    t1w_mask
        Brain mask in T1w space.
    t1w_dseg
        Tissue segmentation in T1w space.
    subjects_dir
        FreeSurfer subjects directory.
    subject_id
        FreeSurfer subject ID.
    fsnative2t1w_xfm
        Transform from FreeSurfer native to T1w space.
    fmap
        Fieldmap image (if available).
    fmap_ref
        Fieldmap reference image.
    fmap_coeff
        Fieldmap B-spline coefficients.
    fmap_mask
        Fieldmap brain mask.
    fmap_id
        Fieldmap estimator ID.
    sdc_method
        SDC method identifier.

    Outputs
    -------
    hmc_dwiref
        3D b=0 reference used for motion correction.
    coreg_dwiref
        SDC-corrected reference used for coregistration.
    dwi_mask
        Brain mask in DWI space.
    motion_xfm
        Per-volume affine transforms for head motion correction.
    dwiref2anat_xfm
        Transform from DWI reference to anatomical space.
    dwiref2fmap_xfm
        Transform from DWI reference to fieldmap space.
    fmap_coeff
        B-spline coefficients for susceptibility distortion correction.
    out_bvec
        Motion-corrected (rotated) gradient directions.

    """
    from niworkflows.interfaces.nibabel import ApplyMask

    from dmriprep.interfaces.vectors import CheckGradientTable

    if precomputed is None:
        precomputed = {}

    layout = config.execution.layout

    workflow = Workflow(name='dwi_fit_wf')
    workflow.__desc__ = """\
The DWI fit stage estimated all spatial transforms without applying them,
enabling single-interpolation resampling in downstream processing.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'in_bvec',
                'in_bval',
                # Anatomical inputs
                't1w_preproc',
                't1w_mask',
                't1w_dseg',
                'subjects_dir',
                'subject_id',
                'fsnative2t1w_xfm',
                # Fieldmap inputs
                'fmap',
                'fmap_ref',
                'fmap_coeff',
                'fmap_mask',
                'fmap_id',
                'sdc_method',
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
                'dwiref2fmap_xfm',
                'fmap_coeff',
                'out_bvec',
                'out_bval',
            ],
        ),
        name='outputnode',
    )

    # Set up DWI file inputs
    inputnode.inputs.dwi_file = dwi_file
    inputnode.inputs.in_bvec = str(layout.get_bvec(dwi_file))
    inputnode.inputs.in_bval = str(layout.get_bval(dwi_file))

    # Check and normalize gradient table
    gradient_table = pe.Node(CheckGradientTable(), name='gradient_table')

    # Buffer nodes for precomputed derivatives
    hmc_buffer = pe.Node(
        niu.IdentityInterface(fields=['hmc_dwiref', 'dwi_mask']),
        name='hmc_buffer',
    )
    xfm_buffer = pe.Node(
        niu.IdentityInterface(fields=['motion_xfm', 'dwiref2anat_xfm']),
        name='xfm_buffer',
    )

    workflow.connect([
        (inputnode, gradient_table, [
            ('dwi_file', 'dwi_file'),
            ('in_bvec', 'in_bvec'),
            ('in_bval', 'in_bval'),
        ]),
    ])  # fmt:skip

    # Check for precomputed derivatives
    have_hmc_ref = 'hmc_dwiref' in precomputed
    have_mask = 'dwi_mask' in precomputed
    have_motion_xfm = precomputed.get('transforms', {}).get('hmc')
    have_coreg_xfm = precomputed.get('transforms', {}).get('dwiref2anat')

    # Stage 1: Create DWI reference
    if not have_hmc_ref:
        dwi_reference_wf = init_dwi_reference_wf(omp_nthreads=omp_nthreads)

        workflow.connect([
            (inputnode, dwi_reference_wf, [('dwi_file', 'inputnode.dwi_file')]),
            (gradient_table, dwi_reference_wf, [('b0_ixs', 'inputnode.b0_ixs')]),
            (dwi_reference_wf, hmc_buffer, [('outputnode.dwi_reference', 'hmc_dwiref')]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info(
            f'Found precomputed HMC reference: {precomputed["hmc_dwiref"]}'
        )
        hmc_buffer.inputs.hmc_dwiref = precomputed['hmc_dwiref']

    # Stage 2: Brain extraction
    if not have_mask:
        from sdcflows.workflows.ancillary import init_brainextraction_wf

        brainextraction_wf = init_brainextraction_wf()

        workflow.connect([
            (hmc_buffer, brainextraction_wf, [('hmc_dwiref', 'inputnode.in_file')]),
            (brainextraction_wf, hmc_buffer, [('outputnode.out_mask', 'dwi_mask')]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info(f'Found precomputed brain mask: {precomputed["dwi_mask"]}')
        hmc_buffer.inputs.dwi_mask = precomputed['dwi_mask']

    # Stage 3: Head motion and eddy current estimation
    if not have_motion_xfm:
        from dmriprep.workflows.dwi.hmc import init_dwi_hmc_wf

        hmc_wf = init_dwi_hmc_wf(omp_nthreads=omp_nthreads)

        workflow.connect([
            (inputnode, hmc_wf, [
                ('dwi_file', 'inputnode.dwi_file'),
                ('in_bvec', 'inputnode.in_bvec'),
                ('in_bval', 'inputnode.in_bval'),
            ]),
            (hmc_buffer, hmc_wf, [
                ('hmc_dwiref', 'inputnode.dwi_reference'),
                ('dwi_mask', 'inputnode.dwi_mask'),
            ]),
            (hmc_wf, xfm_buffer, [('outputnode.motion_xfm', 'motion_xfm')]),
            (hmc_wf, outputnode, [('outputnode.out_bvec', 'out_bvec')]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info(
            f'Found precomputed motion transforms: {precomputed["transforms"]["hmc"]}'
        )
        xfm_buffer.inputs.motion_xfm = precomputed['transforms']['hmc']
        # Still need to rotate b-vectors based on precomputed transforms
        from dmriprep.interfaces.nifreeze import RotateBVecs

        rotate_bvecs = pe.Node(RotateBVecs(), name='rotate_bvecs')
        workflow.connect([
            (inputnode, rotate_bvecs, [
                ('in_bvec', 'in_bvec'),
                ('in_bval', 'in_bval'),
            ]),
            (xfm_buffer, rotate_bvecs, [('motion_xfm', 'transforms')]),
            (rotate_bvecs, outputnode, [('out_bvec', 'out_bvec')]),
        ])  # fmt:skip

    # Stage 4: Susceptibility distortion correction (if fieldmap available)
    coreg_buffer = pe.Node(
        niu.IdentityInterface(fields=['coreg_dwiref', 'fmap_coeff', 'dwiref2fmap_xfm']),
        name='coreg_buffer',
    )

    if fieldmap_id is not None:
        from niworkflows.interfaces.utility import KeySelect
        from sdcflows.workflows.apply.registration import init_coeff2epi_wf

        # Select the appropriate fieldmap
        fmap_select = pe.Node(
            KeySelect(fields=['fmap', 'fmap_ref', 'fmap_coeff', 'fmap_mask']),
            name='fmap_select',
            run_without_submitting=True,
        )
        fmap_select.inputs.key = fieldmap_id

        # Register fieldmap to DWI reference
        coeff2epi_wf = init_coeff2epi_wf(
            debug='fieldmaps' in config.execution.debug,
            omp_nthreads=omp_nthreads,
            write_coeff=True,
        )
        ensure_fmap_mask = pe.Node(
            niu.Function(
                input_names=['fmap_mask', 'fmap_ref'],
                output_names=['out_mask'],
                function=_ensure_fmap_mask,
            ),
            name='ensure_fmap_mask',
        )

        workflow.connect([
            (inputnode, fmap_select, [
                ('fmap', 'fmap'),
                ('fmap_ref', 'fmap_ref'),
                ('fmap_coeff', 'fmap_coeff'),
                ('fmap_mask', 'fmap_mask'),
                ('fmap_id', 'keys'),
            ]),
            (fmap_select, coeff2epi_wf, [
                ('fmap_ref', 'inputnode.fmap_ref'),
                ('fmap_coeff', 'inputnode.fmap_coeff'),
            ]),
            (fmap_select, ensure_fmap_mask, [
                ('fmap_mask', 'fmap_mask'),
                ('fmap_ref', 'fmap_ref'),
            ]),
            (ensure_fmap_mask, coeff2epi_wf, [('out_mask', 'inputnode.fmap_mask')]),
            (hmc_buffer, coeff2epi_wf, [
                ('hmc_dwiref', 'inputnode.target_ref'),
                ('dwi_mask', 'inputnode.target_mask'),
            ]),
            (coeff2epi_wf, coreg_buffer, [
                ('outputnode.fmap_coeff', 'fmap_coeff'),
                ('outputnode.target2fmap_xfm', 'dwiref2fmap_xfm'),
            ]),
        ])  # fmt:skip

        # Apply SDC to reference for coregistration
        from sdcflows.workflows.apply.correction import init_unwarp_wf

        unwarp_wf = init_unwarp_wf(
            debug='fieldmaps' in config.execution.debug,
            omp_nthreads=omp_nthreads,
        )
        unwarp_wf.inputs.inputnode.metadata = layout.get_metadata(dwi_file)

        workflow.connect([
            (hmc_buffer, unwarp_wf, [('hmc_dwiref', 'inputnode.distorted')]),
            (coeff2epi_wf, unwarp_wf, [('outputnode.fmap_coeff', 'inputnode.fmap_coeff')]),
            (unwarp_wf, coreg_buffer, [('outputnode.corrected', 'coreg_dwiref')]),
        ])  # fmt:skip
    else:
        # No SDC - use HMC reference for coregistration
        workflow.connect([
            (hmc_buffer, coreg_buffer, [('hmc_dwiref', 'coreg_dwiref')]),
        ])  # fmt:skip

    # Stage 5: Coregistration to anatomical
    if not have_coreg_xfm:
        from dmriprep.workflows.dwi.registration import init_dwi_reg_wf

        reg_wf = init_dwi_reg_wf(
            freesurfer=config.workflow.run_reconall,
            omp_nthreads=omp_nthreads,
        )

        # Mask the T1w for registration
        t1w_brain = pe.Node(ApplyMask(), name='t1w_brain')

        workflow.connect([
            (inputnode, t1w_brain, [
                ('t1w_preproc', 'in_file'),
                ('t1w_mask', 'in_mask'),
            ]),
            (inputnode, reg_wf, [
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id'),
                ('fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm'),
                ('t1w_dseg', 'inputnode.t1w_dseg'),
            ]),
            (t1w_brain, reg_wf, [('out_file', 'inputnode.t1w_brain')]),
            (coreg_buffer, reg_wf, [('coreg_dwiref', 'inputnode.dwi_ref')]),
            (hmc_buffer, reg_wf, [('dwi_mask', 'inputnode.dwi_mask')]),
            (reg_wf, xfm_buffer, [('outputnode.dwiref2anat_xfm', 'dwiref2anat_xfm')]),
        ])  # fmt:skip
    else:
        config.loggers.workflow.info(
            f'Found precomputed coregistration: {precomputed["transforms"]["dwiref2anat"]}'
        )
        xfm_buffer.inputs.dwiref2anat_xfm = precomputed['transforms']['dwiref2anat']

    # Connect outputs
    workflow.connect([
        (hmc_buffer, outputnode, [
            ('hmc_dwiref', 'hmc_dwiref'),
            ('dwi_mask', 'dwi_mask'),
        ]),
        (coreg_buffer, outputnode, [
            ('coreg_dwiref', 'coreg_dwiref'),
            ('fmap_coeff', 'fmap_coeff'),
            ('dwiref2fmap_xfm', 'dwiref2fmap_xfm'),
        ]),
        (xfm_buffer, outputnode, [
            ('motion_xfm', 'motion_xfm'),
            ('dwiref2anat_xfm', 'dwiref2anat_xfm'),
        ]),
        (gradient_table, outputnode, [('out_bval', 'out_bval')]),
    ])  # fmt:skip

    return workflow


def init_dwi_reference_wf(
    *,
    omp_nthreads: int = 1,
    name: str = 'dwi_reference_wf',
) -> Workflow:
    """
    Build a workflow to generate a DWI reference from b=0 volumes.

    This workflow extracts b=0 volumes, aligns them, and averages to
    create a robust reference image for motion correction.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from dmriprep.workflows.dwi.fit import init_dwi_reference_wf
            wf = init_dwi_reference_wf()

    Parameters
    ----------
    omp_nthreads
        Number of threads for parallel processing.
    name
        Workflow name.

    Inputs
    ------
    dwi_file
        DWI NIfTI file.
    b0_ixs
        Indices of b=0 volumes in the DWI series.

    Outputs
    -------
    dwi_reference
        3D b=0 reference image.
    validation_report
        HTML reportlet for validation.

    """
    from niworkflows.interfaces.images import RobustAverage, ValidateImage
    from niworkflows.interfaces.nibabel import IntensityClip

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A DWI reference image was generated by averaging b=0 volumes after
robust alignment.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_file', 'b0_ixs']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['dwi_reference', 'validation_report']),
        name='outputnode',
    )

    # Validate input image
    validate = pe.Node(ValidateImage(), name='validate')

    # Extract and average b=0 volumes
    extract_b0 = pe.Node(
        niu.Function(
            input_names=['in_file', 'indices'],
            output_names=['out_file'],
            function=_extract_b0_volumes,
        ),
        name='extract_b0',
    )

    # Robust average of b=0 volumes
    average_b0 = pe.Node(
        RobustAverage(two_pass=True),
        name='average_b0',
        n_procs=omp_nthreads,
    )

    # Clip intensity outliers
    clip_intensity = pe.Node(
        IntensityClip(p_min=10.0, p_max=99.5),
        name='clip_intensity',
    )

    workflow.connect([
        (inputnode, validate, [('dwi_file', 'in_file')]),
        (inputnode, extract_b0, [('b0_ixs', 'indices')]),
        (validate, extract_b0, [('out_file', 'in_file')]),
        (extract_b0, average_b0, [('out_file', 'in_file')]),
        (average_b0, clip_intensity, [('out_file', 'in_file')]),
        (clip_intensity, outputnode, [('out_file', 'dwi_reference')]),
        (validate, outputnode, [('out_report', 'validation_report')]),
    ])  # fmt:skip

    return workflow


def _extract_b0_volumes(in_file, indices, newpath=None):
    """Extract b=0 volumes from a 4D DWI image."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    if newpath is None:
        newpath = Path.cwd()

    img = nb.load(in_file)
    data = np.asanyarray(img.dataobj)

    # Handle single index
    if isinstance(indices, int):
        indices = [indices]

    # Extract b=0 volumes
    if len(indices) == 1:
        out_data = data[..., indices[0]]
    else:
        out_data = data[..., indices]

    out_file = fname_presuffix(in_file, suffix='_b0', newpath=str(newpath))
    out_img = nb.Nifti1Image(out_data, img.affine, img.header)
    out_img.to_filename(out_file)

    return out_file


def _ensure_fmap_mask(fmap_mask, fmap_ref):
    """Return a valid fieldmap mask path, falling back to ``fmap_ref`` if missing."""
    from pathlib import Path

    if isinstance(fmap_mask, tuple | list):
        fmap_mask = fmap_mask[0] if fmap_mask else None
    if isinstance(fmap_ref, tuple | list):
        if len(fmap_ref) != 1:
            raise ValueError(f'Expected one fieldmap reference, got {len(fmap_ref)}.')
        fmap_ref = fmap_ref[0]

    if fmap_mask and fmap_mask != 'MISSING' and Path(fmap_mask).exists():
        return str(Path(fmap_mask).absolute())

    if fmap_ref is None:
        raise ValueError('Missing fieldmap reference image for mask fallback.')

    # coeff2epi_wf requires a mask file input; when no mask derivative is available,
    # use the fieldmap reference itself as an all-positive fallback mask source.
    return str(Path(fmap_ref).absolute())
