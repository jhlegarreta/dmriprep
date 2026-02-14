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
Head motion and eddy-current correction estimation workflows.

These workflows estimate transforms without applying them, following
the fit/transform architecture. Motion and eddy current distortions
are estimated using NiFreeze's leave-one-out cross-validation approach.

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def init_dwi_hmc_wf(
    *,
    omp_nthreads: int = 1,
    model: str = 'DTI',
    name: str = 'dwi_hmc_wf',
) -> Workflow:
    """
    Build a workflow for head motion and eddy-current estimation.

    This workflow uses NiFreeze to estimate per-volume affine transforms
    for head motion correction and eddy-current distortion correction.
    The estimation uses leave-one-out cross-validation with diffusion
    models to predict each volume and register predicted to actual.

    Importantly, this workflow only estimates transforms - it does not
    apply them. This enables downstream composition of all transforms
    for single-interpolation resampling.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from dmriprep.workflows.dwi.hmc import init_dwi_hmc_wf
            wf = init_dwi_hmc_wf()

    Parameters
    ----------
    omp_nthreads
        Number of threads for parallel processing.
    model
        Diffusion model for leave-one-out prediction ('DTI', 'DKI', 'GP', 'average').
    name
        Workflow name.

    Inputs
    ------
    dwi_file
        DWI NIfTI file.
    in_bvec
        File path of the b-vectors.
    in_bval
        File path of the b-values.
    dwi_reference
        Pre-computed b=0 reference image.
    dwi_mask
        Brain mask in DWI space.

    Outputs
    -------
    motion_xfm
        Per-volume affine transforms (list of files).
    out_bvec
        Motion-corrected (rotated) gradient directions.
    motion_params
        Motion parameters TSV file (BIDS confounds format).
    estimated_file
        HDF5 file containing full estimation results.

    Notes
    -----
    The estimation approach varies by model:

    - **DTI**: Tensor model, fast but less accurate for high b-values
    - **DKI**: Kurtosis model, better for multi-shell data
    - **GP**: Gaussian Process, most flexible but computationally intensive
    - **average**: Simple averaging, fastest but least accurate

    """
    from dmriprep.interfaces.nifreeze import (
        ExtractTransforms,
        NiFreezeEstimate,
        RotateBVecs,
    )

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
Head motion and eddy-current distortions were estimated using NiFreeze
[@nifreeze], which employs a leave-one-out cross-validation approach
with {model} diffusion model predictions. Per-volume affine transforms
were estimated but not applied at this stage, enabling single-interpolation
resampling during the transform stage.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'in_bvec',
                'in_bval',
                'dwi_reference',
                'dwi_mask',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'motion_xfm',
                'out_bvec',
                'motion_params',
                'estimated_file',
            ],
        ),
        name='outputnode',
    )

    # Run NiFreeze estimation
    estimate = pe.Node(
        NiFreezeEstimate(
            model=model,
            n_threads=omp_nthreads,
        ),
        name='estimate',
        n_procs=omp_nthreads,
    )

    # Extract transforms to standard format
    extract_xfm = pe.Node(
        ExtractTransforms(output_format='itk'),
        name='extract_xfm',
    )

    # Rotate b-vectors based on motion transforms
    rotate_bvecs = pe.Node(
        RotateBVecs(),
        name='rotate_bvecs',
    )

    workflow.connect([
        # Estimation
        (inputnode, estimate, [
            ('dwi_file', 'dwi_file'),
            ('in_bvec', 'bvec_file'),
            ('in_bval', 'bval_file'),
            ('dwi_mask', 'brainmask'),
            ('dwi_reference', 'b0_file'),
        ]),
        # Transform extraction
        (estimate, extract_xfm, [('out_file', 'in_file')]),
        (inputnode, extract_xfm, [('dwi_reference', 'reference')]),
        # B-vector rotation
        (inputnode, rotate_bvecs, [
            ('in_bvec', 'in_bvec'),
            ('in_bval', 'in_bval'),
        ]),
        (extract_xfm, rotate_bvecs, [('motion_xfm', 'transforms')]),
        # Outputs
        (extract_xfm, outputnode, [
            ('motion_xfm', 'motion_xfm'),
            ('motion_params', 'motion_params'),
        ]),
        (rotate_bvecs, outputnode, [('out_bvec', 'out_bvec')]),
        (estimate, outputnode, [('out_file', 'estimated_file')]),
    ])  # fmt:skip

    return workflow


def init_dwi_hmc_flirt_wf(
    *,
    omp_nthreads: int = 1,
    name: str = 'dwi_hmc_flirt_wf',
) -> Workflow:
    """
    Build a fallback HMC workflow using FSL FLIRT.

    This is a simpler alternative to NiFreeze-based estimation that
    uses FSL's FLIRT for volume-to-reference registration. It is
    faster but less accurate, particularly for high b-value data
    where signal dropout makes direct registration challenging.

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
    in_bvec
        File path of the b-vectors.
    in_bval
        File path of the b-values.
    dwi_reference
        Pre-computed b=0 reference image.
    dwi_mask
        Brain mask in DWI space.

    Outputs
    -------
    motion_xfm
        Per-volume affine transforms.
    out_bvec
        Motion-corrected gradient directions.

    """
    from nipype.interfaces.fsl import FLIRT, Split

    from dmriprep.interfaces.nifreeze import RotateBVecs

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Head motion correction transforms were estimated using FSL FLIRT
[@flirt], registering each volume to the b=0 reference.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_file',
                'in_bvec',
                'in_bval',
                'dwi_reference',
                'dwi_mask',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'motion_xfm',
                'out_bvec',
            ],
        ),
        name='outputnode',
    )

    # Split 4D into individual volumes
    split_dwi = pe.Node(
        Split(dimension='t'),
        name='split_dwi',
    )

    # Register each volume to reference
    flirt = pe.MapNode(
        FLIRT(
            dof=6,  # Rigid body
            cost='mutualinfo',
            interp='spline',
        ),
        iterfield=['in_file'],
        name='flirt',
        n_procs=omp_nthreads,
    )

    # Rotate b-vectors
    rotate_bvecs = pe.Node(
        RotateBVecs(),
        name='rotate_bvecs',
    )

    workflow.connect([
        (inputnode, split_dwi, [('dwi_file', 'in_file')]),
        (inputnode, flirt, [
            ('dwi_reference', 'reference'),
            ('dwi_mask', 'ref_weight'),
        ]),
        (split_dwi, flirt, [('out_files', 'in_file')]),
        (inputnode, rotate_bvecs, [
            ('in_bvec', 'in_bvec'),
            ('in_bval', 'in_bval'),
        ]),
        (flirt, rotate_bvecs, [('out_matrix_file', 'transforms')]),
        (flirt, outputnode, [('out_matrix_file', 'motion_xfm')]),
        (rotate_bvecs, outputnode, [('out_bvec', 'out_bvec')]),
    ])  # fmt:skip

    return workflow
