"""Emission coverage over the pinned symbol inventory with units deduplicated
across categories, under the word-initial convention and the bare SAN-marker
exception. Writes emission_{detail,by_category,summary}_word_initial.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from audit_full import (  # noqa: E402
    GPT2_REPO,
    GPT2_REV,
    MISTRAL_REPO,
    MISTRAL_REV,
    PGN_REPO,
    PGN_REV,
    GrammarBaseline,
    HFByteLevel,
    HFSentencePiece,
    TiktokenTok,
)


def main() -> None:
    with (ROOT / "data" / "top20_san.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        top20 = [row["san"] for row in csv.DictReader(handle)]

    squares = [file_name + rank for file_name in "abcdefgh" for rank in "12345678"]
    categories = {
        "castling": ["O-O", "O-O-O"],
        "piece_letters": ["K", "Q", "R", "B", "N"],
        "markers": ["x", "+", "#", "="],
        "squares": squares,
        "san_promotions": ["e8=Q", "e8=N", "a1=Q", "h1=Q"],
        "uci_moves": ["e2e4", "g1f3", "e7e8q", "a2a1q"],
        "fen_atoms": ["w", "b", "KQkq", "-"],
        "top20_san": top20,
    }
    unique_units = list(
        dict.fromkeys(unit for values in categories.values() for unit in values)
    )

    tokenizers = [
        HFByteLevel("gpt2", GPT2_REPO, revision=GPT2_REV),
        TiktokenTok("cl100k_base"),
        TiktokenTok("o200k_base"),
        HFSentencePiece("mistral", MISTRAL_REPO, revision=MISTRAL_REV),
        HFByteLevel("pgn_tokenizer", PGN_REPO, revision=PGN_REV),
        GrammarBaseline(),
    ]

    detail_rows = []
    category_rows = []
    summary_rows = []
    print(f"unique_units={len(unique_units)}")
    for tokenizer in tokenizers:
        def encoding_result(unit: str, *, bare: bool = False) -> tuple[int, bool, bool]:
            encoded = (
                tokenizer.encode_bare(unit)
                if bare
                else tokenizer.encode_unit(unit)
            )
            has_unknown = (
                tokenizer.unk is not None and tokenizer.unk in encoded
            )
            return len(encoded), len(encoded) == 1 and not has_unknown, has_unknown

        strict_by_unit = {}
        for category, units in categories.items():
            category_singles = 0
            for unit in units:
                n_tokens, single_token, has_unknown = encoding_result(unit)
                strict_by_unit[unit] = single_token
                category_singles += int(single_token)
                detail_rows.append(
                    {
                        "tokenizer": tokenizer.name,
                        "category": category,
                        "unit": unit,
                        "n_tokens": n_tokens,
                        "single_token": single_token,
                        "has_unknown": has_unknown,
                    }
                )
            category_rows.append(
                {
                    "tokenizer": tokenizer.name,
                    "category": category,
                    "coverage": category_singles / len(units),
                }
            )

        word_initial = sum(strict_by_unit[unit] for unit in unique_units)
        marker_exception = sum(
            encoding_result(unit, bare=unit in categories["markers"])[1]
            for unit in unique_units
        )
        marker_word_initial = [
            strict_by_unit[unit] for unit in categories["markers"]
        ]
        marker_bare = [
            encoding_result(unit, bare=True)[1]
            for unit in categories["markers"]
        ]
        summary_rows.append(
            {
                "tokenizer": tokenizer.name,
                "unique_units": len(unique_units),
                "word_initial_single": word_initial,
                "word_initial_coverage": word_initial / len(unique_units),
                "bare_marker_single": marker_exception,
                "bare_marker_coverage": marker_exception / len(unique_units),
            }
        )
        print(
            f"{tokenizer.name}: "
            f"word_initial={word_initial}/91 "
            f"({100 * word_initial / len(unique_units):.1f}%); "
            f"marker_exception={marker_exception}/91 "
            f"({100 * marker_exception / len(unique_units):.1f}%); "
            f"marker_word_initial={marker_word_initial}; "
            f"marker_bare={marker_bare}"
        )

    outputs = [
        (
            ROOT / "data" / "emission_detail_word_initial.csv",
            detail_rows,
            ["tokenizer", "category", "unit", "n_tokens", "single_token", "has_unknown"],
        ),
        (
            ROOT / "data" / "emission_by_category_word_initial.csv",
            category_rows,
            ["tokenizer", "category", "coverage"],
        ),
        (
            ROOT / "data" / "emission_summary_word_initial.csv",
            summary_rows,
            [
                "tokenizer",
                "unique_units",
                "word_initial_single",
                "word_initial_coverage",
                "bare_marker_single",
                "bare_marker_coverage",
            ],
        ),
    ]
    for output_path, rows, fieldnames in outputs:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
