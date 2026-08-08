#!/usr/bin/env python3
"""Build a late-window SUVbw reference from classic single-frame PET DICOM.

The implementation is deliberately strict: only BQML data with valid patient
weight, injected activity, half-life, decay correction, and consistent geometry
are accepted. Source files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


HEADER_TAGS = [
    "SOPInstanceUID", "SeriesInstanceUID", "SeriesNumber", "SeriesDescription",
    "Modality", "Rows", "Columns", "FrameReferenceTime", "ActualFrameDuration",
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
    "SliceThickness", "RescaleSlope", "RescaleIntercept", "PatientWeight",
    "Units", "DecayCorrection", "CorrectedImage",
    "RadiopharmaceuticalInformationSequence", "SeriesDate", "SeriesTime",
]


def dicom_datetime(date_value: str, time_value: str) -> datetime:
    date_text = str(date_value)
    time_text = str(time_value).split(".")[0].ljust(6, "0")[:6]
    return datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")


def injection_datetime(dataset, radiopharm) -> datetime:
    value = getattr(radiopharm, "RadiopharmaceuticalStartDateTime", None)
    if value:
        candidate = datetime.strptime(str(value).split(".")[0][:14], "%Y%m%d%H%M%S")
        series_date = datetime.strptime(str(dataset.SeriesDate), "%Y%m%d")
        if abs((candidate.date() - series_date.date()).days) <= 1:
            return candidate
    return dicom_datetime(dataset.SeriesDate, radiopharm.RadiopharmaceuticalStartTime)


def injection_datetime_source(dataset, radiopharm) -> str:
    value = getattr(radiopharm, "RadiopharmaceuticalStartDateTime", None)
    if value:
        candidate = datetime.strptime(str(value).split(".")[0][:14], "%Y%m%d%H%M%S")
        series_date = datetime.strptime(str(dataset.SeriesDate), "%Y%m%d")
        if abs((candidate.date() - series_date.date()).days) <= 1:
            return "RadiopharmaceuticalStartDateTime"
        return "SeriesDate + RadiopharmaceuticalStartTime (full datetime date conflict)"
    return "SeriesDate + RadiopharmaceuticalStartTime"


def scan_datetime(dataset) -> datetime:
    result = dicom_datetime(dataset.SeriesDate, dataset.SeriesTime)
    injected = injection_datetime(dataset, dataset.RadiopharmaceuticalInformationSequence[0])
    if result < injected:
        result += timedelta(days=1)
    return result


def series_uid_hash(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()[:16]


def read_series(case_dir: Path):
    groups = defaultdict(list)
    failed = 0
    for path in case_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=HEADER_TAGS)
        except Exception:
            failed += 1
            continue
        if getattr(ds, "Modality", None) == "PT" and hasattr(ds, "SeriesInstanceUID"):
            groups[str(ds.SeriesInstanceUID)].append((path, ds))
    if not groups:
        raise ValueError("No readable PT DICOM series found")
    return groups, failed


def frame_groups(items):
    frames = defaultdict(list)
    for path, ds in items:
        if not hasattr(ds, "FrameReferenceTime") or not hasattr(ds, "ActualFrameDuration"):
            continue
        frames[float(ds.FrameReferenceTime)].append((path, ds))
    return frames


def complete_frame_groups(items):
    frames = frame_groups(items)
    if not frames:
        return {}, []
    expected_slices = Counter(len(slices) for slices in frames.values()).most_common(1)[0][0]
    complete = {time: slices for time, slices in frames.items() if len(slices) == expected_slices}
    discarded = [
        {"midpoint_s": time / 1000.0, "slice_count": len(slices), "expected_slice_count": expected_slices}
        for time, slices in sorted(frames.items()) if len(slices) != expected_slices
    ]
    return complete, discarded


def series_score(items):
    first = items[0][1]
    if str(getattr(first, "Units", "")).upper() != "BQML":
        return None
    frames, _ = complete_frame_groups(items)
    if not frames:
        return None
    timing = []
    for midpoint_ms, slices in frames.items():
        duration_ms = float(slices[0][1].ActualFrameDuration)
        timing.append((midpoint_ms + duration_ms / 2.0, duration_ms))
    end_ms = max(value[0] for value in timing)
    late_durations = [duration for end, duration in timing if end > end_ms - 600_000]
    return (float(np.median(late_durations)), -len(frames), len(items))


def choose_series(groups):
    candidates = []
    rejected = []
    for uid, items in groups.items():
        score = series_score(items)
        ds = items[0][1]
        row = {
            "series_uid_hash": series_uid_hash(uid),
            "series_number": str(getattr(ds, "SeriesNumber", "")),
            "series_description": str(getattr(ds, "SeriesDescription", "")),
            "units": str(getattr(ds, "Units", "")),
            "file_count": len(items),
            "frame_count": len(frame_groups(items)),
            "complete_frame_count": len(complete_frame_groups(items)[0]),
        }
        if score is None:
            row["reason"] = "requires PT frames in BQML with explicit timing"
            rejected.append(row)
        else:
            candidates.append((score, uid, items, row))
    if not candidates:
        raise ValueError("No eligible BQML dynamic PT series found")
    candidates.sort(reverse=True, key=lambda value: value[0])
    return candidates[0], rejected + [value[3] | {"reason": "lower auto-selection score"} for value in candidates[1:]]


def validate_suv_metadata(ds):
    if str(ds.Units).upper() != "BQML":
        raise ValueError(f"Unsupported PET units: {ds.Units!s}; BQML is required")
    weight_kg = float(ds.PatientWeight)
    radiopharm = ds.RadiopharmaceuticalInformationSequence[0]
    dose_bq = float(radiopharm.RadionuclideTotalDose)
    half_life_s = float(radiopharm.RadionuclideHalfLife)
    if weight_kg <= 0 or dose_bq <= 0 or half_life_s <= 0:
        raise ValueError("Positive PatientWeight, RadionuclideTotalDose, and half-life are required")
    decay = str(getattr(ds, "DecayCorrection", "")).upper()
    if decay not in {"START", "ADMIN"}:
        raise ValueError(f"Unsupported or missing DecayCorrection: {decay!r}")
    injected_at = injection_datetime(ds, radiopharm)
    scan_at = scan_datetime(ds)
    elapsed_s = (scan_at - injected_at).total_seconds()
    if not 0 <= elapsed_s <= 24 * 3600:
        raise ValueError(f"Implausible injection-to-scan interval: {elapsed_s} s")
    denominator_dose_bq = dose_bq
    if decay == "START":
        denominator_dose_bq *= math.exp(-math.log(2.0) * elapsed_s / half_life_s)
    return weight_kg, dose_bq, half_life_s, decay, elapsed_s, denominator_dose_bq


def geometry(items):
    first = items[0][1]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    axis_x = orientation[:3]
    axis_y = orientation[3:]
    axis_z = np.cross(axis_x, axis_y)
    projections = []
    for path, ds in items:
        current = np.asarray(ds.ImageOrientationPatient, dtype=float)
        if not np.allclose(current, orientation, atol=1e-5):
            raise ValueError("Inconsistent ImageOrientationPatient within frame")
        projections.append((float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), axis_z)), path, ds))
    projections.sort(key=lambda value: value[0])
    if len(projections) < 2:
        raise ValueError("At least two slices are required")
    steps = np.diff([value[0] for value in projections])
    spacing_z = float(np.median(steps))
    if spacing_z <= 0 or not np.allclose(steps, spacing_z, rtol=1e-3, atol=1e-3):
        raise ValueError("Nonuniform or duplicate slice positions")
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=float)
    direction = np.column_stack((axis_x, axis_y, axis_z)).ravel().tolist()
    origin = np.asarray(projections[0][2].ImagePositionPatient, dtype=float).tolist()
    spacing = [float(pixel_spacing[1]), float(pixel_spacing[0]), spacing_z]
    return projections, direction, origin, spacing


def load_frame(items):
    ordered, direction, origin, spacing = geometry(items)
    planes = []
    for _, path, header in ordered:
        ds = pydicom.dcmread(str(path))
        slope = float(getattr(ds, "RescaleSlope", getattr(header, "RescaleSlope", 1.0)))
        intercept = float(getattr(ds, "RescaleIntercept", getattr(header, "RescaleIntercept", 0.0)))
        planes.append(ds.pixel_array.astype(np.float32) * slope + intercept)
    return np.stack(planes), direction, origin, spacing


def build(case_dir: Path, output_nifti: Path, output_json: Path, window_s: float, case_key: str):
    groups, failed = read_series(case_dir)
    (score, uid, items, selected), rejected = choose_series(groups)
    first = items[0][1]
    radiopharm = first.RadiopharmaceuticalInformationSequence[0]
    weight_kg, dose_bq, half_life_s, decay, elapsed_s, denominator_dose_bq = validate_suv_metadata(first)
    frames, discarded_frames = complete_frame_groups(items)
    timing = []
    for midpoint_ms, slices in frames.items():
        duration_ms = float(slices[0][1].ActualFrameDuration)
        timing.append((midpoint_ms - duration_ms / 2.0, midpoint_ms + duration_ms / 2.0, midpoint_ms, slices))
    acquisition_end_ms = max(value[1] for value in timing)
    window_start_ms = acquisition_end_ms - window_s * 1000.0
    selected_frames = []
    for start_ms, end_ms, midpoint_ms, slices in sorted(timing):
        overlap_ms = max(0.0, min(end_ms, acquisition_end_ms) - max(start_ms, window_start_ms))
        if overlap_ms > 0:
            selected_frames.append((overlap_ms, midpoint_ms, start_ms, end_ms, slices))
    if not selected_frames:
        raise ValueError("No frames overlap the requested late window")

    weighted_sum = None
    total_weight_ms = 0.0
    reference_geometry = None
    frame_rows = []
    for overlap_ms, midpoint_ms, start_ms, end_ms, slices in selected_frames:
        activity, direction, origin, spacing = load_frame(slices)
        current_geometry = (activity.shape, direction, origin, spacing)
        if reference_geometry is None:
            reference_geometry = current_geometry
        else:
            shape0, direction0, origin0, spacing0 = reference_geometry
            if activity.shape != shape0 or not np.allclose(direction, direction0) or not np.allclose(origin, origin0) or not np.allclose(spacing, spacing0):
                raise ValueError("Selected frames do not share one geometry")
        weighted_sum = activity * overlap_ms if weighted_sum is None else weighted_sum + activity * overlap_ms
        total_weight_ms += overlap_ms
        frame_rows.append({
            "midpoint_s": midpoint_ms / 1000.0,
            "duration_s": (end_ms - start_ms) / 1000.0,
            "overlap_s": overlap_ms / 1000.0,
            "slice_count": len(slices),
        })
    activity_bqml = weighted_sum / total_weight_ms
    suvbw = activity_bqml * (weight_kg * 1000.0) / denominator_dose_bq
    shape, direction, origin, spacing = reference_geometry
    image = sitk.GetImageFromArray(suvbw.astype(np.float32))
    image.SetDirection(direction)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_nifti))

    report = {
        "schema_version": "1.0",
        "case_key": case_key,
        "operation": "duration-weighted late-window SUVbw reference",
        "source_format": "classic single-frame PET DICOM",
        "source_files_modified": False,
        "failed_dicom_headers": failed,
        "selected_series": selected | {"selection_score": list(score)},
        "rejected_series": rejected,
        "window": {
            "requested_duration_s": window_s,
            "requested_duration_s": window_s,
            "acquisition_end_s": acquisition_end_ms / 1000.0,
            "window_start_s": window_start_ms / 1000.0,
            "effective_coverage_s": total_weight_ms / 1000.0,
            "frames": frame_rows,
            "discarded_incomplete_frames": discarded_frames,
        },
        "suvbw": {
            "input_units": "BQML",
            "patient_weight_kg": weight_kg,
            "injected_dose_bq": dose_bq,
            "half_life_s": half_life_s,
            "decay_correction": decay,
            "injection_to_scan_start_s": elapsed_s,
            "denominator_dose_bq": denominator_dose_bq,
            "injection_datetime_source": injection_datetime_source(first, radiopharm),
        },
        "image": {
            "shape_zyx": list(shape), "spacing_xyz_mm": spacing,
            "origin_lps_mm": origin, "direction_lps": direction,
            "suv_percentiles": {str(p): float(np.percentile(suvbw, p)) for p in (50, 90, 99, 99.9, 100)},
        },
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("output_nifti", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--window-seconds", type=float, default=600.0)
    parser.add_argument("--case-key", required=True)
    args = parser.parse_args()
    build(args.case_dir, args.output_nifti, args.output_json, args.window_seconds, args.case_key)


if __name__ == "__main__":
    main()
