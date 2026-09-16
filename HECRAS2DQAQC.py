#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------
Reads a fixed HEC-RAS HDF file, scans ALL time steps & ALL 2D areas, and lists:

 • Face Courant > FACE_COURANT_THRESHOLD (default 6.0)
   -> OUTPUT 1: Only the FaceNumber(s) with repeated exceedances (no TimeIndex lines)
   -> SUMMARY: For repeated faces, include ExceedanceCount, FirstTimeIndex, LastTimeIndex, Max/Min/Mean Face Courant

 • Cell Water Surface Error > CELL_WSE_ERR_THRESHOLD (default 0.01)
   -> TimeIndex, CellID, CellWSEError (unchanged)

Saves a simple, paginated PDF next to the HDF:
 RAS2D_QAQC_Report.pdf

Notes:
- This script DOES NOT compute Courant or WSE error; it reads precomputed results from the HDF.
- Time axis is inferred per dataset (shorter dimension assumed to be time if ambiguous).
- Edit CONFIG if needed. No command-line args.
"""
import os
import numpy as np
import h5py
os.environ["MPLBACKEND"] = "Agg"  # force headless backend BEFORE importing matplotlib
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter
# ------------------------------ CONFIG (edit if needed) ------------------------------ #
HDF_PATH = r"enter your Plan HDF path ending with .hdf"
OUT_PDF_NAME = "RAS2D_QAQC_Report_Summary.pdf"

FACE_COURANT_THRESHOLD = 6.0   # Face Courant exceedance threshold
CELL_WSE_ERR_THRESHOLD  = 0.01 # Cell Water Surface Error exceedance threshold

ONE_BASED_FACE_NUMBERS = False # If True, FaceNumber in output is 1-based
ONE_BASED_CELL_IDS     = False # If True, CellID in output is 1-based

ROWS_PER_PAGE      = 60        # Lines per page (excluding header)
MIN_REPEAT_COUNT   = 2         # A "repeated" face must exceed threshold in >= this many time steps
# ------------------------------------------------------------------------------------- #

# ------------------------------ Utilities ------------------------------ #
def walk_datasets(group, parent_path=""):
    """Yield (full_path, dataset) recursively under an h5py Group."""
    for key, item in group.items():
        full = f"{parent_path}/{key}" if parent_path else key
        if isinstance(item, h5py.Dataset):
            yield full, item
        elif isinstance(item, h5py.Group):
            yield from walk_datasets(item, full)

def list_2d_flow_areas_group(h5):
    """Return the 2D Flow Areas group under Unsteady Time Series, else None."""
    return h5.get("/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas")

def infer_time_axis_from_shape(ds):
    """Infer time axis for a 2D dataset. Returns (time_axis, n_times, n_items)."""
    if len(ds.shape) != 2:
        raise ValueError(f"Expected 2D dataset, got shape={ds.shape}")
    r, c = ds.shape
    if r <= c:
        return 0, r, c
    else:
        return 1, c, r

def format_rows_for_pages(rows, header, rows_per_page=60):
    """Split pre-formatted row strings into page text blocks (header repeats)."""
    pages = []
    for i in range(0, len(rows), rows_per_page):
        chunk = rows[i:i+rows_per_page]
        block = [header] + chunk
        pages.append("\n".join(block))
    return pages

def write_text_pages(pdf, title_prefix, area_name, threshold_str, page_text_blocks):
    """Write one or more text pages for a given area/variable to the PDF."""
    for page_idx, text_block in enumerate(page_text_blocks, start=1):
        fig = plt.figure(figsize=(8.5, 11))  # portrait
        ax = fig.add_subplot(111)
        ax.axis("off")
        title = f"{title_prefix} > {threshold_str} — {area_name} (page {page_idx}/{len(page_text_blocks)})"
        ax.text(0.5, 0.98, title, ha="center", va="top", fontsize=12, fontweight="bold")
        ax.text(0.02, 0.94, text_block, ha="left", va="top", family="monospace", fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

# ------------------------------ Dataset Finders (robust) ------------------------------ #
def find_face_courant_dataset(area_grp):
    """Find a face-based Courant dataset under the area group."""
    candidates = []
    for full, ds in walk_datasets(area_grp):
        low = full.lower()
        if ("courant" in low) and (("face" in low) or ("faces" in low)) and len(ds.shape) == 2:
            candidates.append((full, ds))
    if not candidates:
        return None, None
    _, ds = candidates[0]
    time_axis, n_times, n_items = infer_time_axis_from_shape(ds)
    return ds, time_axis

def find_cell_wse_error_dataset(area_grp):
    """Find a cell-based 'Water Surface Error' dataset under the area group."""
    def matches(name: str) -> bool:
        n = name.lower()
        return ("error" in n) and (("water surface" in n) or ("wse" in n)) and ("cell" in n)
    candidates = []
    for full, ds in walk_datasets(area_grp):
        if len(ds.shape) == 2 and matches(full):
            candidates.append((full, ds))
    if not candidates:
        # Fallback for variants like "Cell Error WSE"
        for full, ds in walk_datasets(area_grp):
            low = full.lower()
            if len(ds.shape) == 2 and ("error" in low) and ("cell" in low) and (("wse" in low) or ("water" in low and "surface" in low)):
                candidates.append((full, ds))
                break
    if not candidates:
        return None, None
    _, ds = candidates[0]
    time_axis, n_times, n_items = infer_time_axis_from_shape(ds)
    return ds, time_axis

# ------------------------------ Main ------------------------------ #
def main():
    if not os.path.isfile(HDF_PATH):
        raise FileNotFoundError(f"HDF not found: {HDF_PATH}")

    out_dir = os.path.dirname(os.path.abspath(HDF_PATH))
    out_pdf = os.path.join(out_dir, OUT_PDF_NAME)

    with h5py.File(HDF_PATH, "r") as h5, PdfPages(out_pdf) as pdf:
        areas_grp = list_2d_flow_areas_group(h5)
        if areas_grp is None or len(areas_grp.keys()) == 0:
            # Single info page
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111); ax.axis("off")
            ax.text(
                0.5, 0.5,
                "No '2D Flow Areas' group found under Unsteady Time Series.\nNothing to report.",
                ha="center", va="center", fontsize=12
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print("[INFO] No 2D Flow Areas found.")
            return

        for area_name in areas_grp.keys():
            print(f"[INFO] Processing area: {area_name}")
            area_grp = areas_grp[area_name]

            # -------- Section 1: Face Courant > threshold (ONLY repeated exceedances) --------
            ds_fc, time_axis_fc = find_face_courant_dataset(area_grp)
            if ds_fc is None:
                # Area info page for missing Face Courant dataset
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111); ax.axis("off")
                ax.text(0.5, 0.5, f"No face-based Courant dataset found for area:\n{area_name}",
                        ha="center", va="center", fontsize=12)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            else:
                # Dimensions
                if time_axis_fc == 0:
                    n_times_fc, n_faces = ds_fc.shape
                else:
                    n_faces, n_times_fc = ds_fc.shape

                # 1) Collect all exceedances and count per face (counting on zero-based ids)
                rows_exceed = []     # tuples: (t, face0, value)
                face_counts = Counter()
                by_face_vals = {}    # face0 -> list of exceedance values
                by_face_times = {}   # face0 -> list of time indices (ints)

                for t in range(n_times_fc):
                    row = ds_fc[t, :] if time_axis_fc == 0 else ds_fc[:, t]
                    row = np.array(row, dtype=float)
                    bad = (~np.isfinite(row)) | (row < -1e20)
                    row[bad] = np.nan
                    mask = row > FACE_COURANT_THRESHOLD
                    if not np.any(mask):
                        continue

                    faces0 = np.where(mask)[0]
                    vals = row[mask]

                    for f0, v in zip(faces0, vals):
                        f0 = int(f0)
                        rows_exceed.append((t, f0, float(v)))
                        face_counts[f0] += 1
                        by_face_vals.setdefault(f0, []).append(float(v))
                        by_face_times.setdefault(f0, []).append(int(t))

                # 2) Determine repeated faces (>= MIN_REPEAT_COUNT)
                repeated_faces = {f0 for f0, cnt in face_counts.items() if cnt >= MIN_REPEAT_COUNT}

                if not repeated_faces:
                    # Info page: no repeated exceedances
                    fig = plt.figure(figsize=(8.5, 11))
                    ax = fig.add_subplot(111); ax.axis("off")
                    msg = (
                        f"Area: {area_name}\n\n"
                        f"No Face Courant values > {FACE_COURANT_THRESHOLD} "
                        f"with repeated exceedances (count ≥ {MIN_REPEAT_COUNT}) across all {n_times_fc} time step(s)."
                    )
                    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                else:
                    # ---------- OUTPUT 1: List only the FaceNumber(s) with repeated exceedances ----------
                    # Order: by exceedance count (desc), then FaceNumber (asc)
                    ordered_repeated = sorted(
                        repeated_faces,
                        key=lambda f0: (-face_counts[f0], f0)
                    )

                    header_faces = f"{'FaceNumber':>12}"
                    lines_faces = []
                    for f0 in ordered_repeated:
                        f_out = f0 + 1 if ONE_BASED_FACE_NUMBERS else f0
                        lines_faces.append(f"{int(f_out):12d}")

                    pages_faces = format_rows_for_pages(lines_faces, header_faces, rows_per_page=ROWS_PER_PAGE)
                    title_faces = f"Faces with repeated Face Courant exceedances (count ≥ {MIN_REPEAT_COUNT})"
                    write_text_pages(pdf, title_faces, area_name, f"{FACE_COURANT_THRESHOLD}", pages_faces)

                    # ---------- SUMMARY: For each repeated face, add stats and first/last exceedance time ----------
                    # Build stats list as tuples for sort+print:
                    # (f0, count, first_t, last_t, max_v, min_v, mean_v)
                    stats_rows = []
                    for f0 in ordered_repeated:
                        ts = by_face_times[f0]
                        vs = by_face_vals[f0]
                        count = len(vs)
                        first_t = int(np.min(ts))
                        last_t  = int(np.max(ts))
                        max_v   = float(np.nanmax(vs))
                        min_v   = float(np.nanmin(vs))
                        mean_v  = float(np.nanmean(vs))
                        stats_rows.append((f0, count, first_t, last_t, max_v, min_v, mean_v))

                    header_sum = (
                        f"{'FaceNumber':>12} {'ExceedanceCount':>18} "
                        f"{'FirstTimeIndex':>16} {'LastTimeIndex':>15} "
                        f"{'MaxFaceCourant':>18} {'MinFaceCourant':>18} {'MeanFaceCourant':>18}"
                    )

                    lines_sum = []
                    for (f0, count, first_t, last_t, max_v, min_v, mean_v) in stats_rows:
                        f_out = f0 + 1 if ONE_BASED_FACE_NUMBERS else f0
                        lines_sum.append(
                            f"{int(f_out):12d} {int(count):18d} "
                            f"{int(first_t):16d} {int(last_t):15d} "
                            f"{max_v:18.6g} {min_v:18.6g} {mean_v:18.6g}"
                        )

                    pages_sum = format_rows_for_pages(lines_sum, header_sum, rows_per_page=ROWS_PER_PAGE)
                    title_sum = f"Face Courant Summary (repeated exceedances, count ≥ {MIN_REPEAT_COUNT})"
                    write_text_pages(pdf, title_sum, area_name, f"{FACE_COURANT_THRESHOLD}", pages_sum)

            # -------- Section 2: Cell WSE Error > threshold (unchanged) --------
            ds_err, time_axis_err = find_cell_wse_error_dataset(area_grp)
            if ds_err is None:
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111); ax.axis("off")
                ax.text(0.5, 0.5, f"No 'Cell Water Surface Error' dataset found for area:\n{area_name}",
                        ha="center", va="center", fontsize=12)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue
            if time_axis_err == 0:
                n_times_err, n_cells = ds_err.shape
            else:
                n_cells, n_times_err = ds_err.shape

            header_err = f"{'TimeIndex':>8} {'CellID':>10} {'CellWSEError':>12}"
            lines_err = []

            for t in range(n_times_err):
                row = ds_err[t, :] if time_axis_err == 0 else ds_err[:, t]
                row = np.array(row, dtype=float)
                bad = (~np.isfinite(row)) | (row < -1e20)
                row[bad] = np.nan
                mask = row > CELL_WSE_ERR_THRESHOLD
                if not np.any(mask):
                    continue
                cells = np.where(mask)[0]
                vals = row[mask]
                if ONE_BASED_CELL_IDS:
                    cells = cells + 1
                for c, v in zip(cells, vals):
                    lines_err.append(f"{t:8d} {int(c):10d} {v:12.6g}")

            if not lines_err:
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111); ax.axis("off")
                msg = (f"Area: {area_name}\n\n"
                       f"No Cell Water Surface Error values > {CELL_WSE_ERR_THRESHOLD} across all {n_times_err} time step(s).")
                ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            else:
                pages_err = format_rows_for_pages(lines_err, header_err, rows_per_page=ROWS_PER_PAGE)
                write_text_pages(pdf, "Cell Water Surface Error", area_name, f"{CELL_WSE_ERR_THRESHOLD}", pages_err)

        print(f"[OK] PDF saved: {out_pdf}")

if __name__ == "__main__":
    main()
