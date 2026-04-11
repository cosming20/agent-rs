"""Generate minimal PDF fixtures for smoke testing.

Run this script once to produce the test PDF files:
    python generate_fixtures.py

Requires: pip install reportlab
"""

from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    print("reportlab not installed; skipping fixture generation")
    raise SystemExit(0)

FIXTURES = Path(__file__).parent


def _make_pdf(name: str, content_fn) -> None:
    path = FIXTURES / name
    c = canvas.Canvas(str(path), pagesize=letter)
    content_fn(c)
    c.save()
    print(f"Written: {path}")


def single_paragraph(c):
    c.setFont("Helvetica", 12)
    c.drawString(72, 720, "Introduction")
    c.setFont("Helvetica", 10)
    c.drawString(72, 700, "This is a single paragraph fixture for smoke-testing the Docling sidecar.")
    c.drawString(72, 688, "It contains exactly one block of plain text with no tables or images.")
    c.showPage()


def one_table(c):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 720, "Data Table")
    data = [["Name", "Value", "Unit"],
            ["Temperature", "36.6", "°C"],
            ["Pressure", "101.3", "kPa"],
            ["Humidity", "65", "%"]]
    t = Table(data, colWidths=[120, 80, 60])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    t.wrapOn(c, 400, 300)
    t.drawOn(c, 72, 600)
    c.showPage()


def two_column(c):
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "Two-Column Layout")
    c.setFont("Helvetica", 10)
    # Column 1
    c.drawString(72, 700, "Left column: Lorem ipsum dolor sit amet,")
    c.drawString(72, 688, "consectetur adipiscing elit. Sed do eiusmod")
    c.drawString(72, 676, "tempor incididunt ut labore et dolore magna.")
    # Column 2
    c.drawString(320, 700, "Right column: Ut enim ad minim veniam,")
    c.drawString(320, 688, "quis nostrud exercitation ullamco laboris")
    c.drawString(320, 676, "nisi ut aliquip ex ea commodo consequat.")
    c.showPage()


def image_with_caption(c):
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 720, "Figure 1: Example Diagram")
    # Placeholder rectangle for image
    c.rect(72, 550, 400, 150, stroke=1, fill=0)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(72, 536, "Figure 1. A placeholder diagram for testing picture+caption extraction.")
    c.showPage()


if __name__ == "__main__":
    _make_pdf("single_paragraph.pdf", single_paragraph)
    _make_pdf("one_table.pdf", one_table)
    _make_pdf("two_column.pdf", two_column)
    _make_pdf("image_with_caption.pdf", image_with_caption)
    print("All fixtures generated.")
