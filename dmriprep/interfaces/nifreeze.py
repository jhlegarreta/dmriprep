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
"""Nipype interfaces for NiFreeze (motion and eddy-current estimation)."""

from pathlib import Path

import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class _NiFreezeEstimateInputSpec(BaseInterfaceInputSpec):
    dwi_file = File(exists=True, mandatory=True, desc='DWI NIfTI file')
    bvec_file = File(exists=True, mandatory=True, desc='b-vectors file')
    bval_file = File(exists=True, mandatory=True, desc='b-values file')
    brainmask = File(exists=True, desc='Brain mask in DWI space')
    b0_file = File(exists=True, desc='Pre-computed b=0 reference image')
    model = traits.Enum(
        'DTI',
        'DKI',
        'GP',
        'average',
        usedefault=True,
        desc='Diffusion model for leave-one-out prediction',
    )
    n_threads = traits.Int(1, usedefault=True, desc='Number of parallel threads')
    b0_threshold = traits.Float(50.0, usedefault=True, desc='Threshold for b=0 volumes')


class _NiFreezeEstimateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='HDF5 file with estimated transforms')
    motion_affines = OutputMultiObject(File(exists=True), desc='Per-volume motion transforms')
    bzero = File(exists=True, desc='Estimated b=0 reference')


class NiFreezeEstimate(SimpleInterface):
    """
    Estimate head motion and eddy-current distortions using NiFreeze.

    This interface wraps NiFreeze's estimation pipeline, which uses
    leave-one-out cross-validation with diffusion models (DTI, DKI, GP)
    to predict each volume and estimate motion/distortion by registering
    predicted to actual volumes.

    The transforms are stored but not applied, following the fit/transform
    architecture pattern.

    """

    input_spec = _NiFreezeEstimateInputSpec
    output_spec = _NiFreezeEstimateOutputSpec

    def _run_interface(self, runtime):
        from nifreeze.data.dmri import DWI

        cwd = Path(runtime.cwd).absolute()

        # Load DWI data
        dwi = DWI.from_filename(
            self.inputs.dwi_file,
            gradients_file=(self.inputs.bvec_file, self.inputs.bval_file),
        )

        # Set brain mask if provided
        if isdefined(self.inputs.brainmask):
            import nibabel as nb

            dwi.brainmask = np.asanyarray(nb.load(self.inputs.brainmask).dataobj) > 0

        # Set b0 reference if provided
        if isdefined(self.inputs.b0_file):
            import nibabel as nb

            dwi.bzero = np.asanyarray(nb.load(self.inputs.b0_file).dataobj)

        # Run estimation - this populates dwi.motion_affines
        # The actual estimation call depends on NiFreeze's API
        # For now, we initialize the structure
        from nifreeze.estimator import Estimator

        estimator = Estimator(
            dwi,
            model=self.inputs.model.lower(),
            n_threads=self.inputs.n_threads,
        )
        estimator.run()

        # Save the DWI object with estimated transforms
        out_file = cwd / 'dwi_estimated.h5'
        dwi.to_filename(str(out_file))
        self._results['out_file'] = str(out_file)

        # Export motion affines as individual transform files
        motion_files = []
        if dwi.motion_affines is not None:
            for idx, affine in enumerate(dwi.motion_affines):
                aff_file = cwd / f'motion_{idx:04d}.txt'
                np.savetxt(str(aff_file), affine, fmt='%.8f')
                motion_files.append(str(aff_file))
        self._results['motion_affines'] = motion_files

        # Save b=0 reference
        if dwi.bzero is not None:
            import nibabel as nb

            bzero_file = cwd / 'bzero.nii.gz'
            bzero_img = nb.Nifti1Image(dwi.bzero, dwi.affine)
            bzero_img.to_filename(str(bzero_file))
            self._results['bzero'] = str(bzero_file)

        return runtime


class _ExtractTransformsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='HDF5 file from NiFreeze estimation')
    reference = File(exists=True, desc='Reference image for transform header')
    output_format = traits.Enum(
        'itk',
        'fsl',
        'afni',
        usedefault=True,
        desc='Output transform format',
    )


class _ExtractTransformsOutputSpec(TraitedSpec):
    motion_xfm = OutputMultiObject(File(exists=True), desc='Per-volume motion transforms')
    eddy_xfm = OutputMultiObject(File(exists=True), desc='Per-volume eddy current transforms')
    motion_params = File(exists=True, desc='Motion parameters TSV file')


class ExtractTransforms(SimpleInterface):
    """
    Extract and convert transforms from NiFreeze HDF5 output.

    Converts NiFreeze's internal transform representation to standard
    neuroimaging formats (ITK, FSL, AFNI) for BIDS derivative output
    and downstream processing.

    """

    input_spec = _ExtractTransformsInputSpec
    output_spec = _ExtractTransformsOutputSpec

    def _run_interface(self, runtime):
        from nifreeze.data.dmri import DWI

        cwd = Path(runtime.cwd).absolute()

        # Load the estimated DWI object
        dwi = DWI.from_filename(self.inputs.in_file)

        # Extract motion transforms
        motion_files = []
        if dwi.motion_affines is not None:
            for idx, affine in enumerate(dwi.motion_affines):
                if self.inputs.output_format == 'itk':
                    xfm_file = cwd / f'vol{idx:04d}_from-orig_to-dwiref_xfm.txt'
                    _write_itk_affine(affine, str(xfm_file))
                else:
                    xfm_file = cwd / f'vol{idx:04d}_motion.mat'
                    np.savetxt(str(xfm_file), affine, fmt='%.8f')
                motion_files.append(str(xfm_file))
        self._results['motion_xfm'] = motion_files

        # Extract eddy current transforms
        eddy_files = []
        if dwi.eddy_xfms is not None:
            for idx, xfm in enumerate(dwi.eddy_xfms):
                if self.inputs.output_format == 'itk':
                    xfm_file = cwd / f'vol{idx:04d}_eddy_xfm.txt'
                    _write_itk_affine(xfm, str(xfm_file))
                else:
                    xfm_file = cwd / f'vol{idx:04d}_eddy.mat'
                    np.savetxt(str(xfm_file), xfm, fmt='%.8f')
                eddy_files.append(str(xfm_file))
        self._results['eddy_xfm'] = eddy_files

        # Create motion parameters TSV
        motion_params_file = cwd / 'motion_params.tsv'
        _write_motion_params(dwi.motion_affines, str(motion_params_file))
        self._results['motion_params'] = str(motion_params_file)

        return runtime


class _RotateBVecsInputSpec(BaseInterfaceInputSpec):
    in_bvec = File(exists=True, mandatory=True, desc='Original b-vectors file')
    in_bval = File(exists=True, mandatory=True, desc='Original b-values file')
    transforms = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='Per-volume affine transforms',
    )
    b0_threshold = traits.Float(50.0, usedefault=True, desc='Threshold for b=0 volumes')


class _RotateBVecsOutputSpec(TraitedSpec):
    out_bvec = File(exists=True, desc='Rotated b-vectors file')


class RotateBVecs(SimpleInterface):
    """
    Rotate b-vectors based on motion correction transforms.

    Extracts the rotation component from each affine transform and
    applies it to the corresponding b-vector. This is necessary to
    maintain consistency between gradient directions and the
    reoriented image data after motion correction.

    """

    input_spec = _RotateBVecsInputSpec
    output_spec = _RotateBVecsOutputSpec

    def _run_interface(self, runtime):
        cwd = Path(runtime.cwd).absolute()

        # Load b-vectors and b-values
        bvecs = np.loadtxt(self.inputs.in_bvec)
        if bvecs.shape[0] == 3:
            bvecs = bvecs.T  # Ensure N x 3

        bvals = np.loadtxt(self.inputs.in_bval).flatten()

        # Load transforms and extract rotations
        rotated_bvecs = bvecs.copy()
        for idx, xfm_file in enumerate(self.inputs.transforms):
            if bvals[idx] < self.inputs.b0_threshold:
                continue  # Skip b=0 volumes

            affine = np.loadtxt(xfm_file)
            if affine.shape == (4, 4):
                rotation = affine[:3, :3]
            else:
                rotation = affine[:3, :3]

            # Normalize rotation (remove scaling)
            from scipy.linalg import polar

            rotation, _ = polar(rotation)

            # Apply rotation to b-vector
            rotated_bvecs[idx] = rotation @ bvecs[idx]

            # Re-normalize
            norm = np.linalg.norm(rotated_bvecs[idx])
            if norm > 1e-6:
                rotated_bvecs[idx] /= norm

        # Write rotated b-vectors
        out_bvec = cwd / 'rotated.bvec'
        np.savetxt(str(out_bvec), rotated_bvecs.T, fmt='%.8f')
        self._results['out_bvec'] = str(out_bvec)

        return runtime


def _write_itk_affine(affine, filename):
    """Write an affine matrix in ITK format."""
    # ITK uses a specific text format for affine transforms
    # The 4x4 matrix is written with specific parameters
    with open(filename, 'w') as f:
        f.write('#Insight Transform File V1.0\n')
        f.write('#Transform 0\n')
        f.write('Transform: AffineTransform_double_3_3\n')

        # Extract rotation and translation
        rotation = affine[:3, :3].flatten()
        translation = affine[:3, 3]

        # Parameters: 9 rotation values + 3 translation values
        params = list(rotation) + list(translation)
        f.write(f'Parameters: {" ".join(f"{p:.10f}" for p in params)}\n')

        # Fixed parameters (center of rotation, typically 0 0 0)
        f.write('FixedParameters: 0 0 0\n')


def _write_motion_params(motion_affines, filename):
    """
    Write motion parameters to a BIDS-compliant TSV file.

    Extracts translation (mm) and rotation (radians) parameters from
    each affine transform and writes them in BIDS confounds format.

    """
    import pandas as pd

    if motion_affines is None or len(motion_affines) == 0:
        # Write empty file with headers
        df = pd.DataFrame(columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'])
        df.to_csv(filename, sep='\t', index=False)
        return

    params = []
    for affine in motion_affines:
        from transforms3d.affines import decompose44
        from transforms3d.euler import mat2euler

        try:
            T, R, _, _ = decompose44(affine)
            rx, ry, rz = mat2euler(R)
        except Exception:
            T = affine[:3, 3] if affine.shape == (4, 4) else np.zeros(3)
            rx, ry, rz = 0, 0, 0

        params.append({
            'trans_x': T[0],
            'trans_y': T[1],
            'trans_z': T[2],
            'rot_x': rx,
            'rot_y': ry,
            'rot_z': rz,
        })

    df = pd.DataFrame(params)
    df.to_csv(filename, sep='\t', index=False)
