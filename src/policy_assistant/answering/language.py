"""Question language selection."""

from __future__ import annotations

from typing import Literal

from policy_assistant.text.language import script_letter_counts


def answer_language(question: str) -> Literal["ur", "en"]:
    arabic, latin = script_letter_counts(question)
    if arabic > 0 and arabic >= latin:
        return "ur"
    return "en"
