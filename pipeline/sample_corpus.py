"""Stratified corpus sampler: streams the May 2026 Lichess standard rated
dump, draws a seeded stratified sample of N_TARGET games, and writes
corpus.pgn, sample_meta.json, and strata.csv to data/.
"""

import io
import json
import random
import datetime
import collections
import csv

import requests
import zstandard
import chess.pgn
from pathlib import Path

DUMP_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-05.pgn.zst"
POOL_TARGET = 60000
N_TARGET = 8000
MIN_PLIES = 10
STRATUM_FLOOR = 100
SEED = 42

OUT_DIR = str(Path(__file__).resolve().parents[1] / "data")


class CountingRaw:
    def __init__(self, raw):
        self.raw = raw
        self.bytes_read = 0

    def read(self, n=-1):
        data = self.raw.read(n)
        self.bytes_read += len(data)
        return data


def tc_category(tc_header: str) -> str:
    """Lichess-style category from 'base+increment' (seconds)."""
    if not tc_header or tc_header == "-":
        return "classical"          # correspondence grouped with classical
    try:
        base, inc = tc_header.split("+")
        est = int(base) + 40 * int(inc)
    except ValueError:
        return "classical"
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    return "classical"


def elo_band(white: str, black: str):
    try:
        mean = (int(white) + int(black)) / 2
    except (TypeError, ValueError):
        return None
    if mean < 1400:
        return "<1400"
    if mean < 1800:
        return "1400-1799"
    if mean < 2200:
        return "1800-2199"
    return ">=2200"


def main():
    access_date = datetime.date.today().isoformat()
    dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)

    pool = collections.defaultdict(list)   # stratum -> [game, ...]
    n_seen = n_short = n_malformed = n_no_elo = 0
    pool_size = 0

    with requests.get(DUMP_URL, stream=True, timeout=120) as r:
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
            plies = sum(1 for _ in game.mainline_moves())
            if plies < MIN_PLIES:
                n_short += 1
                continue
            stratum = f"{tc_category(game.headers.get('TimeControl'))}|{band}"
            pool[stratum].append(game)
            pool_size += 1
            if n_seen % 10000 == 0:
                print(f"  seen {n_seen}, pool {pool_size}, {counting.bytes_read/1e6:.1f} MB", flush=True)
        bytes_fetched = counting.bytes_read

    print(f"pool complete: {pool_size} games in {len(pool)} strata; "
          f"{bytes_fetched/1e6:.1f} MB compressed; seen {n_seen} "
          f"(short {n_short}, malformed {n_malformed}, no-Elo {n_no_elo})")

    # proportional allocation with a floor, trimmed to N_TARGET exactly
    rng = random.Random(SEED)
    alloc = {}
    for s, games in pool.items():
        floor = min(STRATUM_FLOOR, len(games))
        prop = round(N_TARGET * len(games) / pool_size)
        alloc[s] = min(len(games), max(floor, prop))
    # trim/pad largest strata until the total is exactly N_TARGET
    def total():
        return sum(alloc.values())
    strata_by_size = sorted(alloc, key=lambda s: len(pool[s]), reverse=True)
    i = 0
    while total() != N_TARGET:
        s = strata_by_size[i % len(strata_by_size)]
        if total() > N_TARGET and alloc[s] > min(STRATUM_FLOOR, len(pool[s])):
            alloc[s] -= 1
        elif total() < N_TARGET and alloc[s] < len(pool[s]):
            alloc[s] += 1
        i += 1

    sample = []
    for s in sorted(pool):
        chosen = rng.sample(pool[s], alloc[s])
        sample.extend(chosen)
    rng.shuffle(sample)
    print(f"sampled {len(sample)} games across {len(alloc)} strata")

    with open(rf"{OUT_DIR}/corpus.pgn", "w", encoding="utf-8") as f:
        for g in sample:
            g.headers["White"] = "Anonymous"
            g.headers["Black"] = "Anonymous"
            print(g, file=f)
            print(file=f)

    with open(rf"{OUT_DIR}/strata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stratum", "time_control", "elo_band", "pool", "sampled"])
        for s in sorted(pool):
            tc, band = s.split("|")
            w.writerow([s, tc, band, len(pool[s]), alloc[s]])

    meta = {
        "source_url": DUMP_URL,
        "access_date": access_date,
        "compressed_bytes_fetched": bytes_fetched,
        "games_seen": n_seen,
        "excluded_short": n_short,
        "excluded_malformed": n_malformed,
        "excluded_no_elo": n_no_elo,
        "pool_size": pool_size,
        "n_strata": len(pool),
        "sample_size": len(sample),
        "stratum_floor": STRATUM_FLOOR,
        "min_plies_filter": MIN_PLIES,
        "seed": SEED,
    }
    with open(rf"{OUT_DIR}/sample_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("written: corpus.pgn, strata.csv, sample_meta.json")


if __name__ == "__main__":
    main()
