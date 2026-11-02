from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from policy_assistant.ingest.extract import ExtractionError, extract_pages
from policy_assistant.ingest.quality import QualityThresholds, assess_page


def test_txt_pages_split_on_form_feed(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("page one\n\fpage two\n\fpage three", encoding="utf-8")
    assert extract_pages(path, "txt") == ["page one\n", "page two\n", "page three"]


def test_txt_without_form_feed_is_one_page(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("only page", encoding="utf-8")
    assert extract_pages(path, "txt") == ["only page"]


def test_txt_must_be_utf8(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_bytes(b"\xff\xfe broken")
    with pytest.raises(ExtractionError, match="cannot extract"):
        extract_pages(path, "txt")


def _write_pdf(path: Path, pages: list[str]) -> None:
    pdf = canvas.Canvas(str(path))
    for text in pages:
        if text:
            pdf.drawString(72, 720, text)
        pdf.showPage()
    pdf.save()


def test_pdf_extracts_two_pages(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    _write_pdf(path, ["Page one policy text.", "Page two policy text."])
    pages = extract_pages(path, "pdf")
    assert len(pages) == 2
    assert "Page one policy text" in pages[0]
    assert "Page two policy text" in pages[1]


def test_blank_pdf_page_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    _write_pdf(path, ["This page has readable text.", ""])
    pages = extract_pages(path, "pdf")
    assert len(pages) == 2
    assessment = assess_page(pages[1], "en", QualityThresholds())
    assert "empty" in assessment.flags


def test_unsupported_file_type(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="unsupported file type"):
        extract_pages(tmp_path / "doc.docx", "docx")
