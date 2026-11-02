"""PDF and text extraction."""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium


class ExtractionError(RuntimeError):
    """Raised when a source file cannot be decoded into text pages."""


def _extract_txt(path: Path) -> list[str]:
    text = path.read_bytes().decode("utf-8")
    if "\f" in text:
        return text.split("\f")
    return [text]


def _extract_pdf(path: Path) -> list[str]:
    pages: list[str] = []
    document = pdfium.PdfDocument(str(path))
    try:
        for index in range(len(document)):
            page = document[index]
            try:
                text_page = page.get_textpage()
                try:
                    pages.append(str(text_page.get_text_bounded()))
                finally:
                    text_page.close()
            finally:
                page.close()
    finally:
        document.close()
    return pages


def extract_pages(path: Path, file_type: str) -> list[str]:
    """Extract raw page text. Index 0 is page 1."""
    try:
        if file_type == "txt":
            return _extract_txt(path)
        if file_type == "pdf":
            return _extract_pdf(path)
    except ExtractionError:
        raise
    except Exception as exc:  # pdfium raises its own error types
        raise ExtractionError(f"cannot extract {path}: {exc}") from exc
    raise ExtractionError(f"unsupported file type {file_type!r} for {path}")
