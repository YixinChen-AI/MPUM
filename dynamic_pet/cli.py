#!/usr/bin/env python3
"""Command-line entry point for MPUM dynamic PET reference generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pydicom

from .adapter import build as build_classic
from .enhanced import build_reference as build_enhanced


def detect_format(source_dir: Path):
    first_path = next(source_dir.rglob("*.dcm"), None)
    if first_path is None:
        raise ValueError(f"No .dcm files found below {source_dir}")
    dataset = pydicom.dcmread(str(first_path), stop_before_pixels=True)
    if hasattr(dataset, "SharedFunctionalGroupsSequence") and int(getattr(dataset, "NumberOfFrames", 1)) > 1:
        return "enhanced"
    return "classic"


def main():
    parser = argparse.ArgumentParser(
        description="Create a duration-weighted late SUVbw reference for MPUM."
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--format", choices=("auto", "classic", "enhanced"), default="auto")
    parser.add_argument("--late-seconds", type=float, default=600.0)
    parser.add_argument("--case-key", default="case")
    args = parser.parse_args()

    source_format = detect_format(args.source_dir) if args.format == "auto" else args.format
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if source_format == "enhanced":
        build_enhanced(args.source_dir, args.output_dir, args.late_seconds, args.case_key)
    else:
        build_classic(
            args.source_dir,
            args.output_dir / f"pet_late_{int(args.late_seconds)}s_suvbw.nii.gz",
            args.output_dir / "provenance.json",
            args.late_seconds,
            args.case_key,
        )


if __name__ == "__main__":
    main()
