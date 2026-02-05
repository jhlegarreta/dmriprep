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
"""Write outputs (derivatives and reportlets)."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ...interfaces import DerivativesDataSink


def init_reportlets_wf(output_dir, sdc_report=False, name='reportlets_wf'):
    """Set up a battery of datasinks to store reports in the right location."""
    from niworkflows.interfaces.reportlets.masks import SimpleShowMaskRPT

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',
                'summary_report',
                'dwi_ref',
                'dwi_mask',
                'validation_report',
                'sdc_report',
            ]
        ),
        name='inputnode',
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='summary', datatype='figures'),
        name='ds_report_summary',
        run_without_submitting=True,
    )

    mask_reportlet = pe.Node(SimpleShowMaskRPT(), name='mask_reportlet')

    ds_report_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, desc='brain', suffix='mask', datatype='figures'
        ),
        name='ds_report_mask',
        run_without_submitting=True,
    )
    ds_report_validation = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='validation', datatype='figures'),
        name='ds_report_validation',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_report_summary, [('source_file', 'source_file'),
                                        ('summary_report', 'in_file')]),
        (inputnode, mask_reportlet, [('dwi_ref', 'background_file'),
                                     ('dwi_mask', 'mask_file')]),
        (inputnode, ds_report_mask, [('source_file', 'source_file')]),
        (inputnode, ds_report_validation, [('source_file', 'source_file'),
                                           ('validation_report', 'in_file')]),
        (mask_reportlet, ds_report_mask, [('out_report', 'in_file')]),
    ])
    # fmt:on
    if sdc_report:
        ds_report_sdc = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir, desc='sdc', suffix='dwi', datatype='figures'
            ),
            name='ds_report_sdc',
            run_without_submitting=True,
        )
        # fmt:off
        workflow.connect([
            (inputnode, ds_report_sdc, [('source_file', 'source_file'),
                                        ('sdc_report', 'in_file')]),
        ])
        # fmt:on
    return workflow


def init_dwi_derivatives_wf(output_dir, name='dwi_derivatives_wf'):
    """
    Set up a battery of datasinks to store dwi derivatives in the right location.

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives.
    name : :obj:`str`
        Workflow name (default: ``"dwi_derivatives_wf"``).

    Inputs
    ------
    source_file
        One dwi file that will serve as a file naming reference.
    dwi_ref
        The b0 reference.
    dwi_mask
        The brain mask for the dwi file.

    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 'dwi_ref', 'dwi_mask']),
        name='inputnode',
    )

    ds_reference = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            compress=True,
            suffix='epiref',
            datatype='dwi',
        ),
        name='ds_reference',
    )

    ds_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            compress=True,
            desc='brain',
            suffix='mask',
            datatype='dwi',
        ),
        name='ds_mask',
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_reference, [('source_file', 'source_file'),
                                   ('dwi_ref', 'in_file')]),
        (inputnode, ds_mask, [('source_file', 'source_file'),
                              ('dwi_mask', 'in_file')]),
    ])
    # fmt:on

    return workflow


def init_dwi_fit_derivatives_wf(
    output_dir,
    fieldmap_id=None,
    name='dwi_fit_derivatives_wf',
):
    """
    Set up datasinks to store fit-stage derivatives.

    This workflow saves the outputs of the fit stage, including
    reference images, transforms, and rotated gradient directions.

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives.
    fieldmap_id : :obj:`str` or None
        Fieldmap identifier for SDC outputs.
    name : :obj:`str`
        Workflow name.

    Inputs
    ------
    source_file
        DWI file used as naming reference.
    hmc_dwiref
        HMC reference image.
    coreg_dwiref
        Coregistration reference (SDC-corrected if available).
    dwi_mask
        Brain mask in DWI space.
    motion_xfm
        Per-volume motion transforms.
    dwiref2anat_xfm
        DWI-to-anatomical coregistration transform.
    dwiref2fmap_xfm
        DWI-to-fieldmap registration transform.
    fmap_coeff
        Fieldmap B-spline coefficients.
    out_bvec
        Motion-corrected (rotated) b-vectors.
    out_bval
        b-values file.

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',
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
        name='inputnode',
    )

    # HMC reference
    ds_hmc_ref = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            compress=True,
            desc='hmc',
            suffix='dwiref',
            datatype='dwi',
        ),
        name='ds_hmc_ref',
        run_without_submitting=True,
    )

    # Coregistration reference (SDC-corrected if available)
    ds_coreg_ref = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            compress=True,
            desc='coreg',
            suffix='dwiref',
            datatype='dwi',
        ),
        name='ds_coreg_ref',
        run_without_submitting=True,
    )

    # Brain mask
    ds_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            compress=True,
            desc='brain',
            suffix='mask',
            datatype='dwi',
        ),
        name='ds_mask',
        run_without_submitting=True,
    )

    # Motion transforms (as a concatenated file)
    ds_motion_xfm = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            suffix='xfm',
            **{'from': 'orig', 'to': 'dwiref', 'mode': 'image'},
            datatype='dwi',
            extension='.txt',
        ),
        name='ds_motion_xfm',
        run_without_submitting=True,
    )

    # Coregistration transform
    ds_coreg_xfm = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            suffix='xfm',
            **{'from': 'dwiref', 'to': 'T1w', 'mode': 'image'},
            datatype='dwi',
            extension='.txt',
        ),
        name='ds_coreg_xfm',
        run_without_submitting=True,
    )

    # Rotated b-vectors
    ds_bvec = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            suffix='dwi',
            datatype='dwi',
            extension='.bvec',
        ),
        name='ds_bvec',
        run_without_submitting=True,
    )

    # b-values (unchanged but included for completeness)
    ds_bval = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            suffix='dwi',
            datatype='dwi',
            extension='.bval',
        ),
        name='ds_bval',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_hmc_ref, [
            ('source_file', 'source_file'),
            ('hmc_dwiref', 'in_file'),
        ]),
        (inputnode, ds_coreg_ref, [
            ('source_file', 'source_file'),
            ('coreg_dwiref', 'in_file'),
        ]),
        (inputnode, ds_mask, [
            ('source_file', 'source_file'),
            ('dwi_mask', 'in_file'),
        ]),
        (inputnode, ds_coreg_xfm, [
            ('source_file', 'source_file'),
            ('dwiref2anat_xfm', 'in_file'),
        ]),
        (inputnode, ds_bvec, [
            ('source_file', 'source_file'),
            ('out_bvec', 'in_file'),
        ]),
        (inputnode, ds_bval, [
            ('source_file', 'source_file'),
            ('out_bval', 'in_file'),
        ]),
    ])
    # fmt:on

    # Motion transforms need special handling (concatenate list)
    concat_motion = pe.Node(
        niu.Function(
            input_names=['in_files'],
            output_names=['out_file'],
            function=_concat_xfms,
        ),
        name='concat_motion',
    )
    workflow.connect([
        (inputnode, concat_motion, [('motion_xfm', 'in_files')]),
        (inputnode, ds_motion_xfm, [('source_file', 'source_file')]),
        (concat_motion, ds_motion_xfm, [('out_file', 'in_file')]),
    ])  # fmt:skip

    # SDC-related outputs
    if fieldmap_id is not None:
        import re

        fmap_id_safe = re.sub(r'[^a-zA-Z0-9]', '', fieldmap_id)

        ds_fmap_xfm = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                suffix='xfm',
                **{'from': 'dwiref', 'to': fmap_id_safe, 'mode': 'image'},
                datatype='dwi',
                extension='.txt',
            ),
            name='ds_fmap_xfm',
            run_without_submitting=True,
        )

        ds_fmap_coeff = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                compress=True,
                suffix='fieldcoeff',
                datatype='dwi',
            ),
            name='ds_fmap_coeff',
            run_without_submitting=True,
        )

        workflow.connect([
            (inputnode, ds_fmap_xfm, [
                ('source_file', 'source_file'),
                ('dwiref2fmap_xfm', 'in_file'),
            ]),
            (inputnode, ds_fmap_coeff, [
                ('source_file', 'source_file'),
                ('fmap_coeff', 'in_file'),
            ]),
        ])  # fmt:skip

    return workflow


def init_dwi_preproc_derivatives_wf(
    output_dir,
    space='orig',
    name='dwi_preproc_derivatives_wf',
):
    """
    Set up datasinks to store preprocessed DWI derivatives.

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives.
    space : :obj:`str`
        Output space identifier ('orig', 'T1w', or template name).
    name : :obj:`str`
        Workflow name.

    Inputs
    ------
    source_file
        DWI file used as naming reference.
    dwi_preproc
        Preprocessed DWI image.
    dwi_ref
        Reference volume from preprocessed DWI.
    dwi_mask
        Brain mask in output space.
    out_bvec
        Motion-corrected b-vectors.
    out_bval
        b-values file.

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_file',
                'dwi_preproc',
                'dwi_ref',
                'dwi_mask',
                'out_bvec',
                'out_bval',
            ],
        ),
        name='inputnode',
    )

    # Preprocessed DWI
    ds_dwi = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            compress=True,
            space=space if space != 'orig' else None,
            desc='preproc',
            suffix='dwi',
            datatype='dwi',
        ),
        name='ds_dwi',
        run_without_submitting=True,
    )

    # Reference image
    ds_ref = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            compress=True,
            space=space if space != 'orig' else None,
            desc='preproc',
            suffix='dwiref',
            datatype='dwi',
        ),
        name='ds_ref',
        run_without_submitting=True,
    )

    # Brain mask
    ds_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            compress=True,
            space=space if space != 'orig' else None,
            desc='brain',
            suffix='mask',
            datatype='dwi',
        ),
        name='ds_mask',
        run_without_submitting=True,
    )

    # b-vectors
    ds_bvec = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            space=space if space != 'orig' else None,
            suffix='dwi',
            datatype='dwi',
            extension='.bvec',
        ),
        name='ds_bvec',
        run_without_submitting=True,
    )

    # b-values
    ds_bval = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            space=space if space != 'orig' else None,
            suffix='dwi',
            datatype='dwi',
            extension='.bval',
        ),
        name='ds_bval',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_dwi, [
            ('source_file', 'source_file'),
            ('dwi_preproc', 'in_file'),
        ]),
        (inputnode, ds_ref, [
            ('source_file', 'source_file'),
            ('dwi_ref', 'in_file'),
        ]),
        (inputnode, ds_mask, [
            ('source_file', 'source_file'),
            ('dwi_mask', 'in_file'),
        ]),
        (inputnode, ds_bvec, [
            ('source_file', 'source_file'),
            ('out_bvec', 'in_file'),
        ]),
        (inputnode, ds_bval, [
            ('source_file', 'source_file'),
            ('out_bval', 'in_file'),
        ]),
    ])
    # fmt:on

    return workflow


def _concat_xfms(in_files, newpath=None):
    """Concatenate multiple transform files into a single file."""
    from pathlib import Path

    if newpath is None:
        newpath = Path.cwd()
    else:
        newpath = Path(newpath)

    out_file = newpath / 'motion_xfms.txt'

    with open(out_file, 'w') as f:
        for idx, xfm_file in enumerate(in_files):
            f.write(f'# Volume {idx}\n')
            with open(xfm_file) as xf:
                f.write(xf.read())
            f.write('\n')

    return str(out_file)
