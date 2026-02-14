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
DWI-to-anatomical registration workflows.

These workflows estimate the transform between DWI and anatomical
(T1w/T2w) spaces without applying it, following the fit/transform
architecture.

"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from dmriprep import config


def init_dwi_reg_wf(
    *,
    freesurfer: bool = False,
    use_bbr: bool | None = None,
    dwi2anat_dof: int = 6,
    dwi2anat_init: str = 't1w',
    omp_nthreads: int = 1,
    name: str = 'dwi_reg_wf',
) -> Workflow:
    """
    Build a workflow to register DWI reference to anatomical space.

    This workflow computes the transform from DWI reference space to
    anatomical (T1w or T2w) space. When FreeSurfer outputs are available,
    boundary-based registration (BBR) is used for improved accuracy.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from dmriprep.workflows.dwi.registration import init_dwi_reg_wf
            wf = init_dwi_reg_wf()

    Parameters
    ----------
    freesurfer
        Whether FreeSurfer outputs are available for BBR.
    use_bbr
        Force or disable BBR. If None, BBR is used when FreeSurfer
        outputs are available, with fallback to rigid registration.
    dwi2anat_dof
        Degrees of freedom for registration (default: 6 = rigid).
    dwi2anat_init
        Initialization method ('t1w' or 't2w').
    omp_nthreads
        Number of threads for parallel processing.
    name
        Workflow name.

    Inputs
    ------
    dwi_ref
        DWI reference image (SDC-corrected if fieldmap available).
    dwi_mask
        Brain mask in DWI space.
    t1w_brain
        Skull-stripped T1w image.
    t1w_dseg
        Tissue segmentation in T1w space.
    subjects_dir
        FreeSurfer subjects directory.
    subject_id
        FreeSurfer subject ID.
    fsnative2t1w_xfm
        Transform from FreeSurfer native to T1w space.

    Outputs
    -------
    dwiref2anat_xfm
        Transform from DWI reference to anatomical space.
    anat2dwiref_xfm
        Inverse transform (anatomical to DWI reference).
    fallback
        Whether BBR failed and registration fell back to rigid.
    out_report
        Registration reportlet for QC.

    """
    if use_bbr is None:
        use_bbr = freesurfer

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_ref',
                'dwi_mask',
                't1w_brain',
                't1w_dseg',
                'subjects_dir',
                'subject_id',
                'fsnative2t1w_xfm',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwiref2anat_xfm',
                'anat2dwiref_xfm',
                'fallback',
                'out_report',
            ],
        ),
        name='outputnode',
    )

    if freesurfer and use_bbr:
        # Use boundary-based registration with FreeSurfer
        workflow.__desc__ = """\
The DWI reference was co-registered to the T1w reference using
boundary-based registration (BBR) [@bbr] implemented in FreeSurfer,
using the white matter surface as the registration target.
"""
        from niworkflows.anat.coregistration import init_bbreg_wf

        bbr_wf = init_bbreg_wf(
            debug=config.execution.sloppy,
            epi2t1w_init='header' if dwi2anat_init == 'header' else 'register',
            omp_nthreads=omp_nthreads,
        )

        workflow.connect([
            (inputnode, bbr_wf, [
                ('dwi_ref', 'inputnode.in_file'),
                ('fsnative2t1w_xfm', 'inputnode.fsnative2t1w_xfm'),
                (('subject_id', _remove_sub_prefix), 'inputnode.subject_id'),
                ('subjects_dir', 'inputnode.subjects_dir'),
            ]),
            (bbr_wf, outputnode, [
                ('outputnode.itk_epi_to_t1w', 'dwiref2anat_xfm'),
                ('outputnode.itk_t1w_to_epi', 'anat2dwiref_xfm'),
                ('outputnode.fallback', 'fallback'),
                ('outputnode.out_report', 'out_report'),
            ]),
        ])  # fmt:skip
    else:
        # Use FLIRT-based registration (no FreeSurfer)
        workflow.__desc__ = f"""\
The DWI reference was co-registered to the T1w reference using
`mri_coreg` (FreeSurfer) or `flirt` (FSL) with {dwi2anat_dof}
degrees of freedom.
"""
        from niworkflows.interfaces.freesurfer import PatchedMRICoreg

        coreg = pe.Node(
            PatchedMRICoreg(
                dof=dwi2anat_dof,
                sep=[4, 2],
                ftol=0.0001,
                linmintol=0.01,
            ),
            name='coreg',
            n_procs=omp_nthreads,
        )

        workflow.connect([
            (inputnode, coreg, [
                ('dwi_ref', 'source_file'),
                ('t1w_brain', 'reference_file'),
            ]),
            (coreg, outputnode, [
                ('out_lta_file', 'dwiref2anat_xfm'),
            ]),
        ])  # fmt:skip

        # Create inverse transform
        from nipype.interfaces.freesurfer import LTAConvert

        invert_xfm = pe.Node(
            LTAConvert(invert=True),
            name='invert_xfm',
        )

        workflow.connect([
            (coreg, invert_xfm, [('out_lta_file', 'in_lta')]),
            (invert_xfm, outputnode, [('out_lta', 'anat2dwiref_xfm')]),
        ])  # fmt:skip

        # Set fallback to False (no BBR attempted)
        outputnode.inputs.fallback = False

        # Generate registration reportlet
        from niworkflows.interfaces.reportlets.registration import (
            SimpleBeforeAfterRPT as SimpleBeforeAfter,
        )

        reg_report = pe.Node(
            SimpleBeforeAfter(
                before_label='DWI',
                after_label='T1w',
            ),
            name='reg_report',
            mem_gb=0.1,
        )

        workflow.connect([
            (inputnode, reg_report, [
                ('dwi_ref', 'before'),
                ('t1w_brain', 'after'),
            ]),
            (reg_report, outputnode, [('out_report', 'out_report')]),
        ])  # fmt:skip

    return workflow


def init_dwi_t2w_reg_wf(
    *,
    omp_nthreads: int = 1,
    name: str = 'dwi_t2w_reg_wf',
) -> Workflow:
    """
    Build a workflow to register DWI reference to T2w space.

    T2w images often provide better contrast matching for DWI
    registration due to similar signal characteristics. This
    workflow can be used as an intermediate step for DWI-to-T1w
    registration.

    Parameters
    ----------
    omp_nthreads
        Number of threads for parallel processing.
    name
        Workflow name.

    Inputs
    ------
    dwi_ref
        DWI reference image.
    dwi_mask
        Brain mask in DWI space.
    t2w_preproc
        Preprocessed T2w image.
    t2w_mask
        Brain mask in T2w space.

    Outputs
    -------
    dwiref2t2w_xfm
        Transform from DWI reference to T2w space.

    """
    from niworkflows.interfaces.freesurfer import PatchedMRICoreg

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The DWI reference was co-registered to the T2w reference using
`mri_coreg` (FreeSurfer), leveraging the similar contrast between
T2w and diffusion-weighted images.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwi_ref',
                'dwi_mask',
                't2w_preproc',
                't2w_mask',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'dwiref2t2w_xfm',
                'out_report',
            ],
        ),
        name='outputnode',
    )

    # Use mri_coreg for T2w registration
    coreg = pe.Node(
        PatchedMRICoreg(
            dof=6,
            sep=[4, 2],
            ftol=0.0001,
            linmintol=0.01,
        ),
        name='coreg',
        n_procs=omp_nthreads,
    )

    workflow.connect([
        (inputnode, coreg, [
            ('dwi_ref', 'source_file'),
            ('t2w_preproc', 'reference_file'),
        ]),
        (coreg, outputnode, [('out_lta_file', 'dwiref2t2w_xfm')]),
    ])  # fmt:skip

    # Generate reportlet
    from niworkflows.interfaces.reportlets.registration import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )

    reg_report = pe.Node(
        SimpleBeforeAfter(
            before_label='DWI',
            after_label='T2w',
        ),
        name='reg_report',
        mem_gb=0.1,
    )

    workflow.connect([
        (inputnode, reg_report, [
            ('dwi_ref', 'before'),
            ('t2w_preproc', 'after'),
        ]),
        (reg_report, outputnode, [('out_report', 'out_report')]),
    ])  # fmt:skip

    return workflow


def _remove_sub_prefix(subject_id):
    """Remove 'sub-' prefix from subject ID for FreeSurfer compatibility."""
    if subject_id.startswith('sub-'):
        return subject_id[4:]
    return subject_id
