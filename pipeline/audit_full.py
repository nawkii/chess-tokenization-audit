"""Full tokenization audit over the stratified corpus: computes fertility,
tokens per character, split rate, and unknown rate per tokenizer and
notation, plus supporting analyses, and writes all outputs to data/.
"""

import io
import json
import re
import sys
import csv
import platform
import datetime
import collections
import importlib.metadata as md

import numpy as np
import pandas as pd
import requests
import chess
import chess.pgn
import tiktoken
from transformers import AutoTokenizer
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA = str(Path(__file__).resolve().parents[1] / "data")
SEED = 42
FEN_INTERVAL_K = 10
N_BOOTSTRAP = 1000
PREFIXES = [100, 500, 1000, 2000, 5000, 8000]

GPT2_REPO = "openai-community/gpt2"
GPT2_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
MISTRAL_REPO = "mistralai/Mistral-7B-v0.1"
MISTRAL_REV = "27d67f1b5f57dc0953326b2601d68371d40ea8da"
PGN_REPO = "InterwebAlchemy/PGNTokenizer"
PGN_REV = "3341d84e70a1d221cbe0f0d4edc96559845b96d9"


# ---------------------------------------------------------------- tokenizers

class HFByteLevel:
    def __init__(self, name, repo, revision=None):
        self.name = name
        kw = {"revision": revision} if revision else {}
        self.tok = AutoTokenizer.from_pretrained(repo, **kw)
        self.unk = self.tok.unk_token or (
            "[unknown]" if "[unknown]" in self.tok.get_vocab() else None)

    def encode_unit(self, unit):
        return self.tok.tokenize(" " + unit)

    def encode_units(self, units):
        enc = self.tok([" " + u for u in units], add_special_tokens=False)
        return enc["input_ids"]

    def encode_bare(self, unit):
        return self.tok.tokenize(unit)

    def encode_text(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def unit_has_unk(self, unit):
        return self.unk is not None and self.unk in self.encode_unit(unit)


class TiktokenTok:
    unk = None

    def __init__(self, name):
        self.name = name
        self.enc = tiktoken.get_encoding(name)

    def encode_unit(self, unit):
        ids = self.enc.encode(" " + unit)
        return [self.enc.decode([i]) for i in ids]

    def encode_units(self, units):
        return self.enc.encode_batch([" " + u for u in units])

    def encode_bare(self, unit):
        return [self.enc.decode([i]) for i in self.enc.encode(unit)]

    def encode_text(self, text):
        return self.enc.encode(text)

    def unit_has_unk(self, unit):
        return False


class HFSentencePiece:
    unk = None  # byte-fallback: every string encodable

    def __init__(self, name, repo, revision=None):
        self.name = name
        kw = {"revision": revision} if revision else {}
        self.tok = AutoTokenizer.from_pretrained(repo, **kw)

    def encode_unit(self, unit):
        return self.tok.tokenize(unit)

    def encode_units(self, units):
        enc = self.tok(list(units), add_special_tokens=False)
        return enc["input_ids"]

    def encode_bare(self, unit):
        return self.tok.tokenize(unit)

    def encode_text(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def unit_has_unk(self, unit):
        return False


class GrammarBaseline:
    name = "grammar"
    unk = None
    SAN_RE = re.compile(r"O-O-O|O-O|[KQRBN]|[a-h]|[1-8]|x|\+|#|=")

    def encode_unit(self, unit):
        if any(c in unit for c in "/ ") or unit in ("w", "b", "-", "KQkq") or re.fullmatch(r"\d+", unit):
            return list(unit)
        if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", unit):
            return list(unit)
        toks = self.SAN_RE.findall(unit)
        if "".join(toks) == unit:
            return toks
        return list(unit)

    def encode_units(self, units):
        return [self.encode_unit(u) for u in units]

    def encode_bare(self, unit):
        return self.encode_unit(unit)

    def encode_text(self, text):
        out = []
        for w in text.split(" "):
            out.extend(self.encode_unit(w))
        return out

    def unit_has_unk(self, unit):
        return False


# ---------------------------------------------------------------- extraction

def tc_category(tc_header):
    if not tc_header or tc_header == "-":
        return "classical"
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


def elo_band(white, black):
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


DISAMB_RE = re.compile(r"^[KQRBN][a-h1-8]x?[a-h][1-8]")


def extract():
    per_game = []
    with open(rf"{DATA}/corpus.pgn", encoding="utf-8") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            sans, ucis, fens = [], [], []
            constructs = collections.Counter()
            board = g.board()
            ply = 0
            for mv in g.mainline_moves():
                san = board.san(mv)
                sans.append(san)
                ucis.append(mv.uci())
                if board.is_en_passant(mv):
                    constructs["en_passant"] += 1
                board.push(mv)
                ply += 1
                if ply % FEN_INTERVAL_K == 0:
                    fens.append(board.fen())
                if san.startswith("O-O-O"):
                    constructs["castle_long"] += 1
                elif san.startswith("O-O"):
                    constructs["castle_short"] += 1
                if "=" in san:
                    constructs["promotion"] += 1
                if san.endswith("+"):
                    constructs["check"] += 1
                if san.endswith("#"):
                    constructs["mate"] += 1
                if "x" in san:
                    constructs["capture"] += 1
                if DISAMB_RE.match(san):
                    constructs["disambiguated"] += 1
            movetext = " ".join(
                (f"{i//2 + 1}. {s}" if i % 2 == 0 else s) for i, s in enumerate(sans))
            stratum = f"{tc_category(g.headers.get('TimeControl'))}|{elo_band(g.headers.get('WhiteElo'), g.headers.get('BlackElo'))}"
            per_game.append({"san": sans, "uci": ucis, "fen": fens,
                             "movetext": movetext, "stratum": stratum,
                             "constructs": constructs})
    return per_game


# ---------------------------------------------------------------- measurement

def unit_counts_for(tk, per_game):
    """Token count and UNK flag per unit, cached over distinct units."""
    out = {}
    for notation in ("san", "uci"):
        distinct = sorted({u for pg in per_game for u in pg[notation]})
        cache = {}
        for u in distinct:
            toks = tk.encode_unit(u)
            cache[u] = (len(toks), tk.unk is not None and tk.unk in toks)
        out[notation] = cache
    fen_distinct = sorted({f for pg in per_game for f in pg["fen"]})
    if isinstance(tk, GrammarBaseline):
        out["fen"] = {f: (len(f), False) for f in fen_distinct}
    else:
        ids_list = tk.encode_units(fen_distinct)
        if tk.unk is not None:
            unk_id = tk.tok.get_vocab()[tk.unk]
            out["fen"] = {f: (len(ids), unk_id in ids)
                          for f, ids in zip(fen_distinct, ids_list)}
        else:
            out["fen"] = {f: (len(ids), False)
                          for f, ids in zip(fen_distinct, ids_list)}
    return out


def main():
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
    n_san = sum(len(pg["san"]) for pg in per_game)
    n_fen = sum(len(pg["fen"]) for pg in per_game)
    print(f"{n_games} games, {n_san} SAN/UCI units, {n_fen} FEN records", flush=True)

    # rare constructs, overall and per stratum
    overall = collections.Counter()
    by_stratum = collections.defaultdict(collections.Counter)
    games_per_stratum = collections.Counter()
    for pg in per_game:
        overall.update(pg["constructs"])
        by_stratum[pg["stratum"]].update(pg["constructs"])
        games_per_stratum[pg["stratum"]] += 1
    keys = ["capture", "check", "mate", "castle_short", "castle_long",
            "promotion", "en_passant", "disambiguated"]
    with open(rf"{DATA}/constructs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["construct", "count"])
        for k in keys:
            w.writerow([k, overall.get(k, 0)])
    with open(rf"{DATA}/constructs_by_stratum.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stratum", "games"] + keys)
        for s in sorted(by_stratum):
            w.writerow([s, games_per_stratum[s]] + [by_stratum[s].get(k, 0) for k in keys])

    san_freq = collections.Counter(u for pg in per_game for u in pg["san"])
    top20 = [u for u, _ in san_freq.most_common(20)]
    pd.DataFrame(san_freq.most_common(20), columns=["san", "count"]).to_csv(
        rf"{DATA}/top20_san.csv", index=False)

    metrics_rows, conv_rows, boot_rows, seq_rows, err_rows = [], [], [], [], []
    fen_field_rows = []
    arrays = {}
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, n_games, size=(N_BOOTSTRAP, n_games))

    fields = ["placement", "side", "castling", "en_passant", "halfmove", "fullmove"]

    for tk in tokenizers:
        print(f"measuring {tk.name}...", flush=True)
        counts = unit_counts_for(tk, per_game)

        for notation in ("san", "uci", "fen"):
            cache = counts[notation]
            g_tokens = np.array([sum(cache[u][0] for u in pg[notation]) for pg in per_game], dtype=np.int64)
            g_units = np.array([len(pg[notation]) for pg in per_game], dtype=np.int64)
            g_chars = np.array([sum(len(u) for u in pg[notation]) for pg in per_game], dtype=np.int64)
            g_split = np.array([sum(1 for u in pg[notation] if cache[u][0] >= 2) for pg in per_game], dtype=np.int64)
            g_unk = np.array([sum(1 for u in pg[notation] if cache[u][1]) for pg in per_game], dtype=np.int64)

            T, U, C, S, K = g_tokens.sum(), g_units.sum(), g_chars.sum(), g_split.sum(), g_unk.sum()
            metrics_rows.append({
                "tokenizer": tk.name, "notation": notation, "units": int(U),
                "fertility": T / U, "tokens_per_char": T / C,
                "split_rate": S / U, "unk_unit_rate": K / U,
            })

            for n in PREFIXES:
                n = min(n, n_games)
                conv_rows.append({
                    "tokenizer": tk.name, "notation": notation, "games": n,
                    "tokens_per_char": g_tokens[:n].sum() / g_chars[:n].sum(),
                })

            bt = g_tokens[boot_idx].sum(axis=1)
            bu = g_units[boot_idx].sum(axis=1)
            bc = g_chars[boot_idx].sum(axis=1)
            fert = bt / bu
            tpc = bt / bc
            boot_rows.append({
                "tokenizer": tk.name, "notation": notation,
                "fertility": T / U,
                "fertility_lo": float(np.percentile(fert, 2.5)),
                "fertility_hi": float(np.percentile(fert, 97.5)),
                "tpc": T / C,
                "tpc_lo": float(np.percentile(tpc, 2.5)),
                "tpc_hi": float(np.percentile(tpc, 97.5)),
            })

            if notation in ("san", "uci"):
                arrays[f"{tk.name}_{notation}_game_tokens"] = g_tokens

        # sequence length: movetext per game, and tokens per FEN record
        seq = np.array([len(tk.encode_text(pg["movetext"])) for pg in per_game])
        fen_counts = np.array([counts["fen"][f][0] for pg in per_game for f in pg["fen"]])
        arrays[f"{tk.name}_movetext_tokens"] = seq
        arrays[f"{tk.name}_fen_record_tokens"] = fen_counts
        seq_rows.append({
            "tokenizer": tk.name,
            "movetext_mean": seq.mean(), "movetext_sd": seq.std(),
            "movetext_p50": float(np.percentile(seq, 50)),
            "movetext_p90": float(np.percentile(seq, 90)),
            "fen_record_mean": fen_counts.mean(),
            "fen_record_p50": float(np.percentile(fen_counts, 50)),
            "fen_record_p90": float(np.percentile(fen_counts, 90)),
        })

        # per-field FEN metrics
        field_units = {f: [] for f in fields}
        for pg in per_game:
            for fen in pg["fen"]:
                parts = fen.split(" ")
                for f, p in zip(fields, parts):
                    field_units[f].append(p)
        row_f = {"tokenizer": tk.name}
        row_u = {"tokenizer": tk.name}
        for f in fields:
            distinct = sorted(set(field_units[f]))
            fc = {}
            for u in distinct:
                toks = tk.encode_unit(u)
                fc[u] = (len(toks), tk.unk is not None and tk.unk in toks)
            tot_t = sum(fc[u][0] for u in field_units[f])
            tot_unk = sum(1 for u in field_units[f] if fc[u][1])
            row_f[f] = tot_t / len(field_units[f])
            row_u[f] = tot_unk / len(field_units[f])
        row_f["kind"] = "fertility"
        row_u["kind"] = "unk_rate"
        fen_field_rows.extend([row_f, row_u])

        # error analysis: most fragmented frequent SAN types
        cache = counts["san"]
        frequent = [(u, c) for u, c in san_freq.items() if c >= 20]
        worst = sorted(frequent, key=lambda uc: (-cache[uc[0]][0] / len(uc[0]), -cache[uc[0]][0]))[:8]
        for u, c in worst:
            err_rows.append({"tokenizer": tk.name, "unit": u, "count": c,
                             "tokens": cache[u][0],
                             "tokens_per_char": cache[u][0] / len(u)})

    pd.DataFrame(metrics_rows).to_csv(rf"{DATA}/metrics_main.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(rf"{DATA}/convergence.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(rf"{DATA}/bootstrap_ci.csv", index=False)
    pd.DataFrame(seq_rows).to_csv(rf"{DATA}/seqlen_stats.csv", index=False)
    pd.DataFrame(fen_field_rows).to_csv(rf"{DATA}/fen_fields.csv", index=False)
    pd.DataFrame(err_rows).to_csv(rf"{DATA}/error_analysis.csv", index=False)
    np.savez_compressed(rf"{DATA}/arrays.npz", **arrays)

    # emission coverage over the curated, version-pinned symbol set
    squares = [f + r for f in "abcdefgh" for r in "12345678"]
    symbol_set = {
        "castling": ["O-O", "O-O-O"],
        "piece_letters": ["K", "Q", "R", "B", "N"],
        "markers": ["x", "+", "#", "="],
        "squares": squares,
        "san_promotions": ["e8=Q", "e8=N", "a1=Q", "h1=Q"],
        "uci_moves": ["e2e4", "g1f3", "e7e8q", "a2a1q"],
        "fen_atoms": ["w", "b", "KQkq", "-"],
        "top20_san": top20,
    }
    cov_rows = []
    for tk in tokenizers:
        for cat, units in symbol_set.items():
            for u in units:
                toks = tk.encode_bare(u) if cat == "markers" else tk.encode_unit(u)
                is_unk = tk.unk is not None and tk.unk in toks
                cov_rows.append({"tokenizer": tk.name, "category": cat, "unit": u,
                                 "n_tokens": len(toks),
                                 "single_token": len(toks) == 1 and not is_unk})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(rf"{DATA}/emission_detail.csv", index=False)
    (cov.groupby(["tokenizer", "category"])["single_token"].mean().rename("coverage")
        .reset_index().to_csv(rf"{DATA}/emission_by_category.csv", index=False))
    (cov.groupby("tokenizer")["single_token"].mean().rename("coverage")
        .reset_index().to_csv(rf"{DATA}/emission_summary.csv", index=False))

    pins = {
        "python": sys.version,
        "platform": platform.platform(),
        "run_date": datetime.date.today().isoformat(),
        "n_games": n_games, "n_san_units": int(n_san), "n_fen_records": int(n_fen),
        "seed": SEED, "fen_interval": FEN_INTERVAL_K, "n_bootstrap": N_BOOTSTRAP,
        "packages": {p: md.version(p) for p in
                     ["transformers", "tokenizers", "tiktoken", "sentencepiece",
                      "zstandard", "chess", "pandas", "numpy", "matplotlib"]},
        "hf_repos": {},
    }
    for repo in [GPT2_REPO, MISTRAL_REPO, PGN_REPO]:
        try:
            r = requests.get(f"https://huggingface.co/api/models/{repo}", timeout=30)
            pins["hf_repos"][repo] = r.json().get("sha")
        except Exception as e:
            pins["hf_repos"][repo] = f"lookup failed: {e}"
    with open(rf"{DATA}/pinned_versions.json", "w", encoding="utf-8") as f:
        json.dump(pins, f, indent=2)

    print("\n=== MAIN METRICS ===")
    print(pd.DataFrame(metrics_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nall outputs written to", DATA)


if __name__ == "__main__":
    main()
