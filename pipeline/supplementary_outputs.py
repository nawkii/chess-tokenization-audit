"""Split-rate 95% CIs via the by-game clustered bootstrap (same seed and draw
sequence as audit_full, so the resamples match the published run) plus movetext
and FEN-record token percentiles. Writes split_rate_ci.csv,
movetext_distribution.csv and fen_record_distribution.csv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from audit_full import (  # noqa: E402
    DATA,
    GPT2_REPO,
    GPT2_REV,
    MISTRAL_REPO,
    MISTRAL_REV,
    N_BOOTSTRAP,
    PGN_REPO,
    PGN_REV,
    SEED,
    GrammarBaseline,
    HFByteLevel,
    HFSentencePiece,
    TiktokenTok,
    extract,
    unit_counts_for,
)


def main() -> None:
    print("loading tokenizers...", flush=True)
    tokenizers = [
        HFByteLevel("gpt2", GPT2_REPO, revision=GPT2_REV),
        TiktokenTok("cl100k_base"),
        TiktokenTok("o200k_base"),
        HFSentencePiece("mistral", MISTRAL_REPO, revision=MISTRAL_REV),
        HFByteLevel("pgn_tokenizer", PGN_REPO, revision=PGN_REV),
        GrammarBaseline(),
    ]
    print("extracting corpus...", flush=True)
    per_game = extract()
    n_games = len(per_game)
    print(f"{n_games} games", flush=True)

    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, n_games, size=(N_BOOTSTRAP, n_games))

    rows = []
    for tk in tokenizers:
        print(f"measuring {tk.name}...", flush=True)
        counts = unit_counts_for(tk, per_game)
        for notation in ("san", "uci", "fen"):
            cache = counts[notation]
            g_units = np.array([len(pg[notation]) for pg in per_game], dtype=np.int64)
            g_split = np.array([sum(1 for u in pg[notation] if cache[u][0] >= 2)
                                for pg in per_game], dtype=np.int64)
            S, U = g_split.sum(), g_units.sum()
            bs = g_split[boot_idx].sum(axis=1) / g_units[boot_idx].sum(axis=1)
            rows.append({
                "tokenizer": tk.name, "notation": notation,
                "split_rate": S / U,
                "split_lo": float(np.percentile(bs, 2.5)),
                "split_hi": float(np.percentile(bs, 97.5)),
            })

    with open(rf"{DATA}/split_rate_ci.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tokenizer", "notation", "split_rate",
                                          "split_lo", "split_hi"])
        w.writeheader()
        w.writerows(rows)
    print(rf"wrote {DATA}\split_rate_ci.csv")

    arrays = np.load(rf"{DATA}/arrays.npz")

    # movetext percentiles: tokens per numbered game
    mt_rows = []
    for tk in tokenizers:
        key = f"{tk.name}_movetext_tokens"
        if key not in arrays:
            continue
        a = arrays[key]
        mt_rows.append({
            "tokenizer": tk.name, "games": int(a.size),
            "mean": float(a.mean()),
            "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
        })
    with open(rf"{DATA}/movetext_distribution.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tokenizer", "games", "mean",
                                          "p5", "p25", "p50", "p75", "p95"])
        w.writeheader()
        w.writerows(mt_rows)
    print(rf"wrote {DATA}\movetext_distribution.csv")

    dist_rows = []
    for tk in tokenizers:
        key = f"{tk.name}_fen_record_tokens"
        if key not in arrays:
            print(f"note: {key} not in arrays.npz, skipped")
            continue
        a = arrays[key]
        dist_rows.append({
            "tokenizer": tk.name, "records": int(a.size),
            "mean": float(a.mean()),
            "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
        })
    with open(rf"{DATA}/fen_record_distribution.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tokenizer", "records", "mean",
                                          "p5", "p25", "p50", "p75", "p95"])
        w.writeheader()
        w.writerows(dist_rows)
    print(rf"wrote {DATA}\fen_record_distribution.csv")


if __name__ == "__main__":
    main()
