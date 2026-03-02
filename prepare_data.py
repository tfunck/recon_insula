#!/usr/bin/env python3
"""
prepare_brainbuilder_inputs.py

Creates BrainBuilder input CSVs + converts .tif sections to .nii.gz.

Assumptions (from your note):
- Filenames look like: "<SUB> <HEMI> <CHUNK> <ACQ> <SAMPLE>.tif"
  e.g. "H003 R B3 Gallyas 10.tif"
- Only 2D rigid inter-section alignment (no 3D-to-template):
  * chunk_info/hemi_info are written but template-related path fields are set to "NA".
- Voxel sizes:
  * in-plane: 1 micron (x and z)
  * thickness: 20 microns (y)
- NIfTI layout we write is (x, y=1, z) with zooms (1, 20, 1) in *microns*.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

# Prefer tifffile for TIFFs; fallback to imageio if needed.
try:
    import tifffile  # type: ignore
    _HAVE_TIFFFILE = True
except Exception:
    _HAVE_TIFFFILE = False
    import imageio.v3 as iio  # type: ignore


FNAME_RE = re.compile(
    r"^(?P<sub>\S+)\s+(?P<hemi>[LR])\s+(?P<chunk>B\d+)\s+(?P<acq>\S+)\s+(?P<sample>\d+)$"
)

DEFAULT_VOX_SIZES_MICRON = (1.0, 20.0, 1.0)  # (x, y, z) in microns


@dataclass(frozen=True)
class ParsedName:
    sub: str
    hemisphere: str  # "L" or "R"
    chunk_label: str  # e.g., "B3"
    chunk: int        # e.g., 3
    acquisition: str  # e.g., "Nissl" / "Gallyas"
    sample: int       # integer


def parse_filename(p: Path) -> ParsedName:
    stem = p.stem  # keeps spaces, drops ".tif"
    m = FNAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"Filename does not match '<SUB> <HEMI> <CHUNK> <ACQ> <SAMPLE>.tif': {p.name}"
        )
    d = m.groupdict()
    chunk_label = d["chunk"]
    chunk_num = int(chunk_label[1:])  # "B3" -> 3
    return ParsedName(
        sub=d["sub"],
        hemisphere=d["hemi"],
        chunk_label=chunk_label,
        chunk=chunk_num,
        acquisition=d["acq"],
        sample=int(d["sample"]),
    )


def read_tif_grayscale(path: Path) -> np.ndarray:
    """Return a 2D grayscale array (float32)."""
    if _HAVE_TIFFFILE:
        arr = tifffile.imread(str(path))
    else:
        arr = iio.imread(str(path))

    arr = np.asarray(arr)

    # Handle common cases: HxW, HxWxC, or multi-page stacks.
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        # If RGB/RGBA: HxWx3/4
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            gray = rgb.mean(axis=-1)
        else:
            # Likely multi-page stack: take first page
            gray = arr[0]
    elif arr.ndim == 4: # 3 x 3 x H x W or similar
        gray = np.mean(arr, axis=(0, 1))  # average over first two dims
    else:
        arr = arr.squeeze()
        if arr.ndim != 2:
            raise ValueError(f"Unsupported TIFF shape after squeeze: {arr.shape} for {path}")
        gray = arr

    gray = np.squeeze(gray)

    # invert colors
    gray = gray.max() - gray

    #gray = np.fliplr(np.rot90(gray, -1))  # rotate 90° CW and flip vertically to get (z, x)
    gray = np.rot90(gray,-1)

    return gray.astype(np.float32)


def tif_to_nifti(
    tif_path: Path,
    out_path: Path,
    voxel_sizes_micron: Tuple[float, float, float] = DEFAULT_VOX_SIZES_MICRON,
) -> None:
    """
    Save a single 2D TIFF section as a 3D NIfTI with singleton y dimension.

    TIFF is assumed to be (z, x) = (rows, cols).
    We store NIfTI as (x, y, z) by transposing and inserting y=1:
      data3d shape = (x, 1, z)
    """
    data2d = read_tif_grayscale(tif_path)          # (z, x)

    sx, _, sz = voxel_sizes_micron
    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = sx
    affine[1, 1] = sz

    nii = nib.Nifti1Image(data2d, affine)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(out_path))
    print(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif_dir", type=Path, required=True, help="Directory containing the .tif files")
    ap.add_argument("--nii_dir", type=Path, required=True, help="Output directory for .nii.gz files")
    ap.add_argument(
        "--csv_dir",
        type=Path,
        required=True,
        help="Output directory for sect_info.csv, chunk_info.csv, hemi_info.csv",
    )
    ap.add_argument("--pixel_size_micron", type=float, default=.001, help="In-plane pixel size (mm)")
    ap.add_argument("--section_thickness_micron", type=float, default=.02, help="Section thickness (mm)")
    ap.add_argument(
        "--direction",
        type=str,
        default="caudal_to_rostral",
        help="chunk_info.direction (use whatever BrainBuilder expects)",
    )
    args = ap.parse_args()
    
    missing_sections = 3

    tif_dir: Path = args.tif_dir
    nii_dir: Path = args.nii_dir
    csv_dir: Path = args.csv_dir
    csv_dir.mkdir(parents=True, exist_ok=True)

    sx = float(args.pixel_size_micron)
    sz = float(args.pixel_size_micron)
    sy = float(args.section_thickness_micron)
    vox = (sx, sy, sz)

    tifs = sorted(tif_dir.glob("*.tif")) + sorted(tif_dir.glob("*.tiff"))
    if not tifs:
        raise SystemExit(f"No .tif/.tiff files found in {tif_dir}")

    sect_rows: List[Dict] = []
    seen_subjects = set()
    seen_hemi = set()
    seen_chunks = set()

    for tif in tifs:
        meta = parse_filename(tif)
        seen_subjects.add(meta.sub)
        seen_hemi.add(meta.hemisphere)
        seen_chunks.add(meta.chunk)

        out_name = f"sub-{meta.sub}_chunk-{meta.chunk}_sample-{meta.sample}_{meta.acquisition}.nii.gz"
        out_nii = (nii_dir / out_name).resolve()

        # Convert to NIfTI if needed
        if not out_nii.exists():
            tif_to_nifti(tif, out_nii, voxel_sizes_micron=vox)

        sect_rows.append(
            {
                "raw": str(out_nii),
                "sub": meta.sub,
                "hemisphere": meta.hemisphere,
                "acquisition": meta.acquisition,
                "sample": meta.sample,
                "chunk": meta.chunk,
            }
        )

    # Sort sect_info for readability: by acquisition then sample
    sect_df = pd.DataFrame(sect_rows)
    sect_df = sect_df.sort_values([ "sample","acquisition"], kind="mergesort").reset_index(drop=True)
    
    # Some Nissl and Gallyas samples have the same number, assume Galyas comes first 
    # and create a new column with unique integer sample numbers across acquisitions
    # Assign a unique integer to each row, incrementing by 1 (starting from 1)
    # But if you want a unique sample index per acquisition/sample combo, use:
    sect_df["unique_sample"] = np.arange(1, len(sect_df) + 1)
    
    for i, sample in enumerate(sorted(sect_df['sample'].unique())):
        sect_df['unique_sample'].loc[sample==sect_df['sample']] += missing_sections * i 
    
    sect_df['sample'] = sect_df['unique_sample']
    sect_df = sect_df.drop(columns=["unique_sample"])
    
    
    sect_df = sect_df.sort_values([ "sample","acquisition"], kind="mergesort").reset_index(drop=True)
    

    # chunk_info: one row per (sub, chunk, hemisphere)
    # (If you truly only have one chunk, this stays one row.)
    chunk_rows = []
    for sub in sorted(seen_subjects):
        for hemi in sorted(seen_hemi):
            for chunk in sorted(seen_chunks):
                chunk_rows.append(
                    {
                        "sub": sub,
                        "chunk": chunk,
                        "hemisphere": hemi,
                        "pixel_size_0": sx,
                        "pixel_size_1": sz,
                        "section_thickness": sy,
                        "direction": "rostral_to_caudal",  # or "caudal_to_rostral" based on your note
                    }
                )
    chunk_df = pd.DataFrame(chunk_rows)

    # hemi_info: placeholders since you’re skipping 3D-to-template alignment
    hemi_rows = []
    for sub in sorted(seen_subjects):
        for hemi in sorted(seen_hemi):
            hemi_rows.append(
                {
                    "sub": sub,
                    "hemisphere": hemi,
                    "struct_ref_vol": "mni_icbm152_01_tal_nlin_asym_09c.nii.gz",
                    "gm_surf": "NA",
                    "wm_surf": "NA",
                }
            )
    hemi_df = pd.DataFrame(hemi_rows)

    sect_path = csv_dir / "sect_info.csv"
    chunk_path = csv_dir / "chunk_info.csv"
    hemi_path = csv_dir / "hemi_info.csv"

    sect_df.to_csv(sect_path, index=False)
    chunk_df.to_csv(chunk_path, index=False)
    hemi_df.to_csv(hemi_path, index=False)

    print(f"Wrote: {sect_path}")
    print(f"Wrote: {chunk_path}")
    print(f"Wrote: {hemi_path}")
    print(f"Converted {len(sect_df)} TIFFs to NIfTI in: {nii_dir.resolve()}")


if __name__ == "__main__":
    main()