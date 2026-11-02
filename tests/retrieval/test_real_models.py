import os

import numpy as np
import pytest

from policy_assistant.config import Settings
from policy_assistant.retrieval.models import CrossEncoderReranker, SentenceTransformerEmbedder

pytestmark = pytest.mark.skipif(
    os.environ.get("UPA_MODEL_TESTS") != "1",
    reason="set UPA_MODEL_TESTS=1 to load the real bge-m3 and reranker models",
)


def test_real_model_embeddings_and_reranker() -> None:
    settings = Settings()
    embedder = SentenceTransformerEmbedder(settings)
    vectors = embedder.embed(
        [
            "How many days of annual leave do employees get?",
            "ملازمین کو کتنے دن کی سالانہ چھٹی ملتی ہے؟",
            "The finance director approves large purchases.",
        ]
    )
    assert vectors.shape == (3, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    related = float(np.dot(vectors[0], vectors[1]))
    unrelated = float(np.dot(vectors[0], vectors[2]))
    assert related >= unrelated + 0.1

    reranker = CrossEncoderReranker(settings)
    scores = reranker.score(
        "How many days of annual leave do employees get?",
        [
            "Every full-time employee is entitled to 20 working days of paid annual leave.",
            "The Finance Director may approve expenditure up to Rs. 2,000,000.",
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]
