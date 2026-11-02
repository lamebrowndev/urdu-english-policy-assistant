from policy_assistant.answering.citations import resolve_answer
from policy_assistant.answering.model import CitationRef, ReplySegment
from tests.fakes import make_chunk


def test_citation_validation_numbering_and_reuse() -> None:
    chunks = (
        make_chunk(
            chunk_id=10,
            document_id="doc-a",
            sentences=("S1", "S2", "S3"),
            page_start=1,
            page_end=2,
            section_heading="1. Purpose",
        ),
        make_chunk(chunk_id=20, document_id="doc-b", sentences=("S4",)),
    )
    segments = (
        ReplySegment(
            "A",
            (
                CitationRef(0, 0, 1, "S1"),
                CitationRef(0, 1, 2, "S2"),
                CitationRef(0, 0, 1, "S1"),
            ),
        ),
        ReplySegment(
            "B",
            (
                CitationRef(1, 0, 1, "S4"),
                CitationRef(9, 0, 1, "bad"),
            ),
        ),
        ReplySegment("C", (CitationRef(0, 0, 1, "S1"),)),
    )
    resolution = resolve_answer(segments, chunks)
    assert resolution.text == "A[1][2]B[3]C[1]"
    assert [source.number for source in resolution.sources] == [1, 2, 3]
    assert resolution.sources[0].chunk_id == 10
    assert resolution.sources[0].start_sentence == 0
    assert resolution.sources[0].end_sentence == 1
    assert resolution.sources[0].document_title == "Document"
    assert resolution.sources[0].section_heading == "1. Purpose"
    assert "invalid_citation" in resolution.flags
    assert resolution.segments[0]["citations"][0]["number"] == 1


def test_all_invalid_citations_produce_no_sources() -> None:
    resolution = resolve_answer(
        (ReplySegment("A", (CitationRef(4, 0, 1, "bad"),)),),
        (make_chunk(),),
    )
    assert resolution.sources == ()
    assert resolution.text == "A"
    assert resolution.flags == ("invalid_citation",)
