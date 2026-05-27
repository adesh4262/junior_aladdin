"""Convert a PDF to a plain text file using pypdf.

Usage:
    f:\junior_aladdin\.venv\Scripts\python.exe scripts\pdf_to_text.py "Junior Aladdin Data Center Master Blueprint V1.pdf"

This writes an output file next to the PDF with the same base name and `.txt` extension.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


def extract_text_from_pdf(path: Path) -> Iterable[str]:
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            yield page.extract_text() or ""
        except Exception:
            yield ""


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: pdf_to_text.py <pdf-path>")
        return 2

    pdf_path = Path(argv[0])
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return 3

    out_path = pdf_path.with_suffix(".txt")
    parts = []
    for p in extract_text_from_pdf(pdf_path):
        parts.append(p)

    text = "\n\n".join(parts)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
