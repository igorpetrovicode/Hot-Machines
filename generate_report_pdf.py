#!/usr/bin/env python3
"""
Generate a consolidated PDF report for the wind turbine generator
thermal study.

Contents (in order):
  1. ABB motor datasheet          (source PDF)
  2. Form 4 fleet calibration     (generated PNG → PDF page)
  3. ZEV relay trip curves         (generated PNG → PDF page)
  4. Per-turbine Machine Study     (one PNG → PDF page per turbine)
  5. Model Equations summary       (generated PNG → PDF page)
  6. Model Deficiency — Loss Stack-up  (efficiency side study, final)

Usage:
    python generate_report_pdf.py                  # default paths
    python generate_report_pdf.py --out report.pdf  # custom output path

Requires: pypdf, Pillow (PIL)
"""
import os
import sys
import tempfile
from pypdf import PdfReader, PdfWriter
from PIL import Image


# ── Default paths ──────────────────────────────────────────────────────

# Source datasheets (uploaded PDFs)
# All paths are resolved relative to this script's directory, so the
# repo works wherever it's checked out. Place the ABB and ZEV datasheet
# PDFs alongside this script.
_HERE = os.path.dirname(os.path.abspath(__file__))

ABB_DATASHEET = os.path.join(_HERE, "M3AA 250SMA 4 G 400V 50Hz 55kW.pdf")
ZEV_DATASHEET = os.path.join(_HERE, "Pages from h1433g.pdf")

# Generated plots (from run_fleet.py --plots)
PLOT_DIR = _HERE
FORM4_PLOT = os.path.join(PLOT_DIR, "form4_calibration.png")
ZEV_PLOT = os.path.join(PLOT_DIR, "zev_trip_curves.png")
EQUATIONS_PLOT = os.path.join(PLOT_DIR, "model_equations.png")
EFFICIENCY_PLOT = os.path.join(PLOT_DIR, "efficiency_stackup_study.png")

# Default turbine labels (matches run_fleet.py DEFAULT_FLEET)
TURBINE_LABELS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

DEFAULT_OUTPUT = os.path.join(PLOT_DIR, "machine_study_report.pdf")


def png_to_pdf(png_path, pdf_path):
    """Convert a PNG image to a single-page PDF, fitting to page."""
    img = Image.open(png_path)
    # Convert to RGB if necessary (PDF doesn't support RGBA)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    # Save as PDF; Pillow handles the conversion
    img.save(pdf_path, "PDF", resolution=150)


def generate_report(output_path=None, plot_dir=None, turbine_labels=None):
    """
    Collate all study materials into a single PDF.

    Parameters
    ----------
    output_path : str
        Path for the output PDF. Defaults to DEFAULT_OUTPUT.
    plot_dir : str
        Directory containing the generated plots. Defaults to PLOT_DIR.
    turbine_labels : list of str
        Turbine labels (e.g. ["T1", "T2", ...]). Defaults to TURBINE_LABELS.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT
    if plot_dir is None:
        plot_dir = PLOT_DIR
    if turbine_labels is None:
        turbine_labels = TURBINE_LABELS

    writer = PdfWriter()
    tmp_files = []  # track temp PDFs for cleanup

    def add_pdf(pdf_path, description):
        """Append all pages from an existing PDF."""
        if not os.path.exists(pdf_path):
            print(f"  WARNING: {description} not found: {pdf_path}")
            return 0
        reader = PdfReader(pdf_path)
        n = len(reader.pages)
        for page in reader.pages:
            writer.add_page(page)
        print(f"  + {description} ({n} page{'s' if n > 1 else ''})")
        return n

    def add_png(png_path, description):
        """Convert a PNG to a temp PDF and append it."""
        if not os.path.exists(png_path):
            print(f"  WARNING: {description} not found: {png_path}")
            return 0
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.close()
        tmp_files.append(tmp.name)
        png_to_pdf(png_path, tmp.name)
        return add_pdf(tmp.name, description)

    print("=" * 60)
    print("  Generating Machine Study Report")
    print("=" * 60)
    total = 0

    # 1. ABB motor datasheet
    total += add_pdf(ABB_DATASHEET, "ABB M3AA 250SMA datasheet")
    
    # 2. ZEV relay datasheet
    total += add_pdf(ZEV_DATASHEET, "Pages from h1433g.pdf")
    
    # 3. Form 4 fleet calibration
    total += add_png(FORM4_PLOT, "Form 4 fleet calibration")

    # 4. ZEV trip curves (our model)
    total += add_png(ZEV_PLOT, "ZEV relay trip curves (model)")

    # 5. Per-turbine Machine Study pages
    for label in turbine_labels:
        png = os.path.join(plot_dir, f"{label}_thevenin_summary.png")
        total += add_png(png, f"Machine Study — {label}")

    # 6. Model Equations summary
    total += add_png(EQUATIONS_PLOT, "Model Equations summary")

    # 7. Model Deficiency — Loss Stack-up (final page, side study)
    total += add_png(EFFICIENCY_PLOT, "Model Deficiency — Loss Stack-up")

    # Write the combined PDF
    with open(output_path, "wb") as f:
        writer.write(f)

    # Clean up temp files
    for tmp in tmp_files:
        try:
            os.remove(tmp)
        except OSError:
            pass

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Report: {output_path}")
    print(f"  Pages:  {total}")
    print(f"  Size:   {size_mb:.1f} MB")
    print("=" * 60)
    return output_path


if __name__ == "__main__":
    args = sys.argv[1:]
    out = DEFAULT_OUTPUT
    for i, a in enumerate(args):
        if a == "--out" and i + 1 < len(args):
            out = args[i + 1]
    generate_report(output_path=out)
