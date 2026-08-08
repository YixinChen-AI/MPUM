#!/usr/bin/env python3
"""Build a late-window SUVbw reference volume from UIH Enhanced PET frames.

The source directory is treated as read-only. Each DICOM object is expected to
contain one complete 3-D dynamic frame. Frame selection is based on overlap
with the requested late time window, and aggregation is duration weighted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from .adapter import (
    injection_datetime,
    injection_datetime_source,
    scan_datetime,
    validate_quantitative_corrections,
)


def _shared_item(dataset, sequence_name):
    shared = dataset.SharedFunctionalGroupsSequence[0]
    return getattr(shared, sequence_name)[0]


def _frame_geometry(dataset):
    pixel_measures = _shared_item(dataset, "PixelMeasuresSequence")
    orientation = np.asarray(
        _shared_item(dataset, "PlaneOrientationSequence").ImageOrientationPatient,
        dtype=np.float64,
    )
    positions = np.asarray(
        [
            frame.PlanePositionSequence[0].ImagePositionPatient
            for frame in dataset.PerFrameFunctionalGroupsSequence
        ],
        dtype=np.float64,
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]
    if len(positions) < 2:
        slice_direction = np.cross(row_direction, column_direction)
        slice_spacing = float(pixel_measures.SliceThickness)
    else:
        slice_step = np.median(np.diff(positions, axis=0), axis=0)
        slice_spacing = float(np.linalg.norm(slice_step))
        if slice_spacing <= 0:
            raise ValueError("Invalid zero slice spacing")
        slice_direction = slice_step / slice_spacing

    # SimpleITK axes are x=columns, y=rows, z=frames. Direction is a
    # row-major matrix whose columns are the physical directions of x/y/z.
    direction = np.column_stack(
        (row_direction, column_direction, slice_direction)
    ).reshape(-1)
    pixel_spacing = [float(x) for x in pixel_measures.PixelSpacing]
    spacing = (pixel_spacing[1], pixel_spacing[0], slice_spacing)
    origin = tuple(float(x) for x in positions[0])
    return spacing, tuple(direction.tolist()), origin, positions


def _rescale(dataset):
    transform = _shared_item(dataset, "PixelValueTransformationSequence")
    slope = float(transform.RescaleSlope)
    intercept = float(transform.RescaleIntercept)
    pixels = dataset.pixel_array.astype(np.float32)
    return pixels * slope + intercept, slope, intercept


def _frame_table(source_dir: Path):
    rows = []
    for path in sorted(source_dir.rglob("*.dcm")):
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        if str(dataset.Modality) != "PT":
            continue
        midpoint = float(dataset.FrameReferenceTime) / 1000.0
        duration = float(dataset.ActualFrameDuration) / 1000.0
        rows.append(
            {
                "path": path,
                "midpoint": midpoint,
                "duration": duration,
                "start": midpoint - duration / 2.0,
                "end": midpoint + duration / 2.0,
                "header": dataset,
            }
        )
    rows.sort(key=lambda row: row["midpoint"])
    if not rows:
        raise ValueError(f"No PET DICOM files found in {source_dir}")
    return rows


def build_reference(source_dir: Path, output_dir: Path, late_seconds: float, case_key="case"):
    rows = _frame_table(source_dir)
    series = {str(row["header"].SeriesInstanceUID) for row in rows}
    if len(series) != 1:
        raise ValueError(f"Expected one PET series, found {len(series)}")

    scan_end = max(row["end"] for row in rows)
    window_start = scan_end - late_seconds
    selected = []
    for index, row in enumerate(rows):
        overlap = max(0.0, min(row["end"], scan_end) - max(row["start"], window_start))
        if overlap > 0:
            selected.append((index, row, overlap))
    if not selected:
        raise ValueError("No dynamic frames overlap the requested late window")

    first_header = rows[0]["header"]
    if str(first_header.Units).upper() != "BQML":
        raise ValueError(f"Expected BQML input, got {first_header.Units}")
    corrected = validate_quantitative_corrections(first_header)

    weight_kg = float(first_header.PatientWeight)
    radiopharm = first_header.RadiopharmaceuticalInformationSequence[0]
    total_dose_bq = float(radiopharm.RadionuclideTotalDose)
    half_life_s = float(radiopharm.RadionuclideHalfLife)
    if weight_kg <= 0 or total_dose_bq <= 0 or half_life_s <= 0:
        raise ValueError("PatientWeight, RadionuclideTotalDose, and half-life must be positive")
    elapsed_s = (scan_datetime(first_header) - injection_datetime(first_header, radiopharm)).total_seconds()
    if not 0 <= elapsed_s <= 24 * 3600:
        raise ValueError(f"Implausible injection-to-scan interval: {elapsed_s} s")
    decay_correction = str(getattr(first_header, "DecayCorrection", "")).upper()
    if decay_correction == "START":
        denominator_dose_bq = total_dose_bq * math.exp(-math.log(2.0) * elapsed_s / half_life_s)
        decay_note = "activity and injected dose referenced to scan start"
    elif decay_correction == "ADMIN":
        denominator_dose_bq = total_dose_bq
        decay_note = "activity and injected dose referenced to administration"
    elif not decay_correction and elapsed_s <= 10.0:
        denominator_dose_bq = total_dose_bq
        decay_note = "decay anchor absent; accepted because injection-to-scan interval is <=10 s"
    else:
        raise ValueError("DecayCorrection must be START or ADMIN unless injection and scan start differ by <=10 s")

    weighted_sum = None
    total_weight = 0.0
    reference_geometry = None
    selected_metadata = []
    for index, row, overlap in selected:
        dataset = pydicom.dcmread(str(row["path"]))
        bqml, slope, intercept = _rescale(dataset)
        suv = bqml * weight_kg * 1000.0 / denominator_dose_bq
        suv = np.clip(suv, 0.0, None).astype(np.float32, copy=False)

        spacing, direction, origin, positions = _frame_geometry(dataset)
        geometry = (suv.shape, spacing, direction, origin)
        if reference_geometry is None:
            reference_geometry = geometry
            reference_positions = positions
        else:
            if suv.shape != reference_geometry[0]:
                raise ValueError("Selected frames have different array shapes")
            if not np.allclose(spacing, reference_geometry[1], atol=1e-5):
                raise ValueError("Selected frames have different voxel spacing")
            if not np.allclose(direction, reference_geometry[2], atol=1e-5):
                raise ValueError("Selected frames have different orientation")
            if not np.allclose(origin, reference_geometry[3], atol=1e-3):
                raise ValueError("Selected frames have different origins")
            if not np.allclose(positions, reference_positions, atol=1e-3):
                raise ValueError("Selected frames have different slice positions")

        if weighted_sum is None:
            weighted_sum = np.zeros_like(suv, dtype=np.float64)
        weighted_sum += suv * overlap
        total_weight += overlap
        selected_metadata.append(
            {
                "index": index,
                "midpoint_seconds": row["midpoint"],
                "duration_seconds": row["duration"],
                "overlap_seconds": overlap,
                "rescale_slope": slope,
                "rescale_intercept": intercept,
            }
        )

    reference = (weighted_sum / total_weight).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pet_late_{int(late_seconds)}s_suvbw.nii.gz"
    image = sitk.GetImageFromArray(reference)
    image.SetSpacing(reference_geometry[1])
    image.SetDirection(reference_geometry[2])
    image.SetOrigin(reference_geometry[3])
    sitk.WriteImage(image, str(output_path), True)

    positive = reference[reference > 0]
    percentiles = {
        str(q): float(np.percentile(positive, q))
        for q in (1, 50, 90, 95, 99, 99.9, 100)
    }
    report = {
        "schema_version": "1.0",
        "case_key": case_key,
        "source_format": "enhanced PET; one complete 3-D volume per DICOM object",
        "source_files_modified": False,
        "source_series_count": 1,
        "total_frames": len(rows),
        "scan_start_seconds": min(row["start"] for row in rows),
        "scan_end_seconds": scan_end,
        "late_window_seconds": late_seconds,
        "late_window_start_seconds": window_start,
        "aggregation": "duration_weighted_mean",
        "selected_frames": selected_metadata,
        "input_units": str(first_header.Units),
        "output_units": "SUVbw",
        "patient_weight_kg": weight_kg,
        "injected_dose_bq": total_dose_bq,
        "half_life_s": half_life_s,
        "injection_to_scan_start_s": elapsed_s,
        "injection_datetime_source": injection_datetime_source(first_header, radiopharm),
        "denominator_dose_bq": denominator_dose_bq,
        "corrected_image": corrected,
        "decay_correction": decay_correction or None,
        "decay_handling": decay_note,
        "shape_zyx": list(reference.shape),
        "spacing_xyz_mm": list(reference_geometry[1]),
        "direction": list(reference_geometry[2]),
        "origin_lps_mm": list(reference_geometry[3]),
        "positive_suv_percentiles": percentiles,
        "output_path": str(output_path),
    }
    report_path = output_dir / f"pet_late_{int(late_seconds)}s_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--late-seconds", type=float, default=600.0)
    parser.add_argument("--case-key", default="case")
    args = parser.parse_args()
    build_reference(args.source_dir, args.output_dir, args.late_seconds, args.case_key)


if __name__ == "__main__":
    main()
