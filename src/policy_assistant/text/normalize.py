"""Two text forms: display text (``clean_text``) and lookup keys (``fold``)."""

from __future__ import annotations

import re
import unicodedata

_CLEAN_DELETE = frozenset(
    {
        0x00AD,
        0x200B,
        0x200D,
        0x200E,
        0x200F,
        0x202A,
        0x202B,
        0x202C,
        0x202D,
        0x202E,
        0x2066,
        0x2067,
        0x2068,
        0x2069,
        0xFEFF,
    }
)
_SPACE_RUN = re.compile(r" +")
_BLANK_RUN = re.compile(r"\n{3,}")

# Order matters: NFKC composes most hamza sequences, but not U+06CC + U+0654.
_FOLD_TRANSLATION: dict[int, str] = {
    0x064A: "\u06cc",  # Arabic yeh -> Urdu yeh
    0x0649: "\u06cc",  # Arabic alef maksura -> Urdu yeh
    0x0643: "\u06a9",  # Arabic kaf -> Urdu kaf
    0x0647: "\u06c1",  # Arabic heh -> Urdu heh goal
    0x06C0: "\u06c1",  # heh with yeh above -> heh goal
    0x06C2: "\u06c1",  # heh goal with hamza above -> heh goal
    0x06D5: "\u06c1",  # AE -> heh goal
    0x06D3: "\u0626\u06d2",  # yeh barree with hamza above -> ئے
    0x0623: "\u0627",  # alef with hamza above -> alef
    0x0625: "\u0627",  # alef with hamza below -> alef
    0x0671: "\u0627",  # alef wasla -> alef
    0x06D4: ".",  # Urdu full stop
    0x060C: ",",
    0x061F: "?",
    0x061B: ";",
    0x066A: "%",
}
for _digit in range(10):
    _FOLD_TRANSLATION[0x06F0 + _digit] = str(_digit)  # extended Arabic-Indic
    _FOLD_TRANSLATION[0x0660 + _digit] = str(_digit)  # Arabic-Indic


def clean_text(raw: str) -> str:
    """Return normalized display text. Stored and shown, never a lookup key."""
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = "".join(char for char in text if char == "\n" or unicodedata.category(char) != "Cc")
    text = "".join(char for char in text if ord(char) not in _CLEAN_DELETE)
    text = "\n".join(_SPACE_RUN.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_RUN.sub("\n\n", text).strip()


def fold(text: str) -> str:
    """Return the lossy lookup key for ``text``. Never store this as display text."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = unicodedata.normalize("NFKC", folded)
    folded = folded.replace("\u06cc\u0654", "\u0626")
    folded = folded.translate(_FOLD_TRANSLATION)
    folded = "".join(
        char
        for char in folded
        if ord(char) != 0x0640
        and ord(char) != 0x00AD
        and unicodedata.category(char) not in {"Mn", "Cf"}
    )
    return re.sub(r"\s+", " ", folded).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize folded text. Callers must not use these tokens as display text."""
    return re.findall(r"\w+", fold(text))
