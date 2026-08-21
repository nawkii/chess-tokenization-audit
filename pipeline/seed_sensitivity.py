"""Re-execute the sampling at seeds 7 and 2024 and recompute fertility, tokens per character and split rate; writes seed_sensitivity.csv."""

import io
import os
import sys
import json
import random
import datetime
import collections

import numpy as np
import pandas as pd
import requests
import zstandard
import chess.pgn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

from sample_corpus import (DUMP_URL, POOL_TARGET, N_TARGET, MIN_PLIES,
                           STRATUM_FLOOR, CountingRaw, tc_category, elo_band)
from audit_full import (HFByteLevel, TiktokenTok, HFSentencePiece,
                        GrammarBaseline, FEN_INTERVAL_K,
                        GPT2_REPO, GPT2_REV, MISTRAL_REPO,
                        MISTRAL_REV, PGN_REPO, PGN_REV)

DATA = str(Path(__file__).resolve().parents[1] / "data")
SEEDS = [42, 7, 2024]
NOTATIONS = ("san", "uci", "fen")


def build_pool():
    """Stream the dump head and rebuild the candidate pool (order-stable)."""
    dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
    pool = collections.defaultdict(list)
    n_seen = n_short = n_malformed = n_no_elo = 0
    pool_size = 0
    with requests.get(DUMP_URL, stream=True, timeout=180) as r:
        r.raise_for_status()
        counting = CountingRaw(r.raw)
        reader = dctx.stream_reader(counting, read_across_frames=True)
        text = io.TextIOWrapper(io.BufferedReader(reader, buffer_size=1 << 20),
                                encoding="utf-8", errors="replace")
        while pool_size < POOL_TARGET:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            n_seen += 1
            if game.errors:
                n_malformed += 1
                continue
            band = elo_band(game.headers.get("WhiteElo"), game.headers.get("BlackElo"))
            if band is None:
                n_no_elo += 1
                continue
            if sum(1 for _ in game.mainline_moves()) < MIN_PLIES:
                n_short += 1
                continue
            stratum = f"{tc_category(game.headers.get('TimeControl'))}|{band}"
            pool[stratum].append(str(game))
            pool_size += 1
            if n_seen % 10000 == 0:
                print(f"  seen {n_seen}, pool {pool_size}, "
                      f"{counting.bytes_read/1e6:.1f} MB", flush=True)
        bytes_fetched = counting.bytes_read
    print(f"pool complete: {pool_size} games in {len(pool)} strata; "
          f"{bytes_fetched/1e6:.1f} MB; seen {n_seen} (short {n_short}, "
          f"malformed {n_malformed}, no-Elo {n_no_elo})", flush=True)
    return pool, pool_size, bytes_fetched, n_seen


def allocate(pool, pool_size):
    """Proportional allocation with a floor , identical to sample_corpus,."""
    alloc = {}
    for s, games in pool.items():
        floor = min(STRATUM_FLOOR, len(games))
        prop = round(N_TARGET * len(games) / pool_size)
        alloc[s] = min(len(games), max(floor, prop))

    def total():
        return sum(alloc.values())

    by_size = sorted(alloc, key=lambda s: len(pool[s]), reverse=True)
    i = 0
    while total() != N_TARGET:
        s = by_size[i % len(by_size)]
        if total() > N_TARGET and alloc[s] > min(STRATUM_FLOOR, len(pool[s])):
            alloc[s] -= 1
        elif total() < N_TARGET and alloc[s] < len(pool[s]):
            alloc[s] += 1
        i += 1
    return alloc


def draw(pool, alloc, seed):
    rng = random.Random(seed)
    sample = []
    for s in sorted(pool):
        sample.extend(rng.sample(pool[s], alloc[s]))
    rng.shuffle(sample)
    return sample


def extract(pgn_texts):
    per_game = []
    for txt in pgn_texts:
        g = chess.pgn.read_game(io.StringIO(txt))
        if g is None:
            continue
        sans, ucis, fens = [], [], []
        board = g.board()
        ply = 0
        for mv in g.mainline_moves():
            sans.append(board.san(mv))
            ucis.append(mv.uci())
            board.push(mv)
            ply += 1
            if ply % FEN_INTERVAL_K == 0:
                fens.append(board.fen())
        per_game.append({"san": sans, "uci": ucis, "fen": fens,
                         "site": g.headers.get("Site", "")})
    return per_game


def site_urls(pgn_path):
    urls = []
    with open(pgn_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith('[Site '):
                urls.append(line.strip())
    return urls


def main():
    pool, pool_size, bytes_fetched, n_seen = build_pool()
    alloc = allocate(pool, pool_size)

    samples = {s: draw(pool, alloc, s) for s in SEEDS}
    for s in SEEDS:
        print(f"seed {s}: {len(samples[s])} games", flush=True)

    print("extracting corpora...", flush=True)
    corpora = {s: extract(samples[s]) for s in SEEDS}
    for s in SEEDS:
        pg = corpora[s]
        print(f"  seed {s}: {sum(len(g['san']) for g in pg):,} SAN/UCI units, "
              f"{sum(len(g['fen']) for g in pg):,} FEN records", flush=True)

    orig = site_urls(rf"{DATA}/corpus.pgn")
    redrawn = [f'[Site "{g["site"]}"]' for g in corpora[42]]
    reproduced = (sorted(orig) == sorted(redrawn))
    print(f"seed-42 redraw reproduces released corpus: {reproduced} "
          f"({len(orig)} vs {len(redrawn)} games)", flush=True)

    union = {n: sorted({u for s in SEEDS for g in corpora[s] for u in g[n]})
             for n in NOTATIONS}
    for n in NOTATIONS:
        print(f"union distinct {n}: {len(union[n]):,}", flush=True)

    tokenizers = [
        HFByteLevel("gpt2", GPT2_REPO, revision=GPT2_REV),
        TiktokenTok("cl100k_base"),
        TiktokenTok("o200k_base"),
        HFSentencePiece("mistral", MISTRAL_REPO, revision=MISTRAL_REV),
        HFByteLevel("pgn_tokenizer", PGN_REPO, revision=PGN_REV),
        GrammarBaseline(),
    ]

    rows = []
    top20 = {}
    for s in SEEDS:
        freq = collections.Counter(u for g in corpora[s] for u in g["san"])
        top20[s] = [u for u, _ in freq.most_common(20)]

    for tk in tokenizers:
        print(f"measuring {tk.name}...", flush=True)
        cache = {}
        for n in NOTATIONS:
            units = union[n]
            if n == "fen" and not isinstance(tk, GrammarBaseline):
                ids_list = tk.encode_units(units)
                if tk.unk is not None:
                    unk_id = tk.tok.get_vocab()[tk.unk]
                    cache[n] = {u: (len(i), unk_id in i) for u, i in zip(units, ids_list)}
                else:
                    cache[n] = {u: (len(i), False) for u, i in zip(units, ids_list)}
            elif n == "fen":
                cache[n] = {u: (len(u), False) for u in units}
            else:
                c = {}
                for u in units:
                    toks = tk.encode_unit(u)
                    c[u] = (len(toks), tk.unk is not None and tk.unk in toks)
                cache[n] = c

        for s in SEEDS:
            for n in NOTATIONS:
                cc = cache[n]
                T = sum(cc[u][0] for g in corpora[s] for u in g[n])
                U = sum(len(g[n]) for g in corpora[s])
                C = sum(len(u) for g in corpora[s] for u in g[n])
                S = sum(1 for g in corpora[s] for u in g[n] if cc[u][0] >= 2)
                K = sum(1 for g in corpora[s] for u in g[n] if cc[u][1])
                rows.append({"seed": s, "tokenizer": tk.name, "notation": n,
                             "units": U, "fertility": T / U,
                             "tokens_per_char": T / C,
                             "split_rate": S / U, "unk_unit_rate": K / U})

    df = pd.DataFrame(rows)
    df.to_csv(rf"{DATA}/seed_sensitivity.csv", index=False)

    base = df[df.seed == 42].set_index(["tokenizer", "notation"])
    worst = {"fertility": 0.0, "tokens_per_char": 0.0, "split_rate": 0.0}
    detail = []
    for _, r in df[df.seed != 42].iterrows():
        b = base.loc[(r.tokenizer, r.notation)]
        for m in worst:
            if b[m] > 0:
                d = abs(r[m] - b[m]) / b[m] * 100
                worst[m] = max(worst[m], d)
                detail.append({"seed": r.seed, "tokenizer": r.tokenizer,
                               "notation": r.notation, "metric": m, "pct_dev": d})
    dd = pd.DataFrame(detail)
    print("\n=== MAX RELATIVE DEVIATION FROM SEED 42 (%) ===")
    print(json.dumps({k: round(v, 4) for k, v in worst.items()}, indent=2))
    print("\nlargest single deviations:")
    print(dd.sort_values("pct_dev", ascending=False).head(8).to_string(index=False))

    t20_overlap = {s: len(set(top20[42]) & set(top20[s])) for s in SEEDS}
    print("\ntop-20 SAN overlap with seed 42:", t20_overlap)

    meta = {
        "run_date": datetime.date.today().isoformat(),
        "seeds": SEEDS,
        "pool_size": pool_size,
        "games_seen": n_seen,
        "compressed_bytes_fetched": bytes_fetched,
        "n_strata": len(pool),
        "sample_size": N_TARGET,
        "seed42_reproduces_released_corpus": bool(reproduced),
        "units_per_seed": {str(s): {n: int(sum(len(g[n]) for g in corpora[s]))
                                    for n in NOTATIONS} for s in SEEDS},
        "max_pct_deviation_from_seed42": {k: round(v, 4) for k, v in worst.items()},
        "top20_san_overlap_with_seed42": {str(k): v for k, v in t20_overlap.items()},
    }
    with open(rf"{DATA}/seed_sensitivity_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("\nwritten: seed_sensitivity.csv, seed_sensitivity_meta.json")


if __name__ == "__main__":
    main()
