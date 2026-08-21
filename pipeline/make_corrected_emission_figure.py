"""Generate the word-initial emission coverage figure (Figure 4.4)."""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "data")
FIGS = str(ROOT / "figures")

ORDER = ["gpt2", "cl100k_base", "o200k_base", "mistral", "pgn_tokenizer", "grammar"]
LABEL = {"gpt2": "GPT-2", "cl100k_base": "cl100k", "o200k_base": "o200k",
         "mistral": "Mistral", "pgn_tokenizer": "PGN-specific", "grammar": "Grammar"}

COLOR = {"gpt2": "#2a78d6", "cl100k_base": "#1baf7a", "o200k_base": "#eda100",
         "mistral": "#d1495b", "pgn_tokenizer": "#4a3aa7", "grammar": "#8a8a8a"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SEQ = ["#eaf2fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
       "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
       "#104281", "#0d366b"]
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.dpi": 200,
})


def style_ax(ax, xgrid=False, ygrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xgrid:
        ax.grid(axis="x", zorder=0)
    if ygrid:
        ax.grid(axis="y", zorder=0)
    ax.tick_params(length=3, width=0.8)


def heat_grid(ax, M, mask=None):
    """Thin white separators so adjacent cells never bleed together."""
    ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)


def fig_tpc():
    m = pd.read_csv(rf"{DATA}/metrics_main.csv")
    notations = [("san", "SAN (PGN)"), ("uci", "UCI"), ("fen", "FEN")]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.6), sharey=True)
    for ax, (nt, title) in zip(axes, notations):
        sub = m[m.notation == nt].set_index("tokenizer").loc[ORDER]
        y = np.arange(len(ORDER))[::-1]
        for yi, tk in zip(y, ORDER):
            v = sub.loc[tk, "tokens_per_char"]
            lossy = sub.loc[tk, "unk_unit_rate"] > 0
            ax.barh(yi, v, height=0.62, color=COLOR[tk], zorder=3,
                    hatch="///" if lossy else None,
                    edgecolor="white" if lossy else "none", linewidth=0.5)
            txt = f"{v:.3f}" + (" lossy" if lossy else "")
            ax.text(v + 0.02, yi, txt, va="center", ha="left",
                    fontsize=7, color=INK2 if lossy else INK)
        ax.set_yticks(y, [LABEL[t] for t in ORDER], fontsize=8.5)
        ax.set_title(title, fontsize=9, color=INK, pad=4)
        ax.set_xlim(0, 1.42)
        ax.set_xticks([0, 0.5, 1.0])
        style_ax(ax, xgrid=True)
    axes[1].set_xlabel("tokens per character (lower is more compact)", fontsize=8.5)
    fig.tight_layout(w_pad=1.1)
    fig.savefig(rf"{FIGS}/fig_tpc.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_seqlen():
    arr = np.load(rf"{DATA}/arrays.npz")
    data = [arr[f"{tk}_movetext_tokens"] for tk in ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    y = np.arange(len(ORDER))[::-1]
    bp = ax.boxplot(data, positions=y, vert=False, widths=0.55,
                    showfliers=False, patch_artist=True, whis=(5, 95), zorder=3,
                    medianprops=dict(color="white", linewidth=1.3))
    for patch, tk in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLOR[tk])
        patch.set_alpha(0.9)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.5)
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(AXIS)
            line.set_linewidth(0.9)
    xmax = 700
    for yi, d in zip(y, data):
        ax.text(xmax - 8, yi, f"median {np.median(d):.0f}", va="center",
                ha="right", fontsize=8.2, color=INK2)
    ax.set_yticks(y, [LABEL[t] for t in ORDER], fontsize=8.5)
    ax.set_xlabel("tokens per game (SAN movetext; whiskers 5th to 95th percentile)",
                  fontsize=8.5)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600])
    style_ax(ax, xgrid=True)
    fig.tight_layout()
    fig.savefig(rf"{FIGS}/fig_seqlen.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_fen_fields():
    """Two panels on independent colour scales."""
    df = pd.read_csv(rf"{DATA}/fen_fields.csv")
    fert = df[df.kind == "fertility"].set_index("tokenizer").loc[ORDER]
    unk = df[df.kind == "unk_rate"].set_index("tokenizer").loc[ORDER]
    short = ["side", "castling", "en_passant", "halfmove", "fullmove"]
    slabel = ["side to\nmove", "castling\nrights", "en-passant\ntarget",
              "halfmove\nclock", "fullmove\nnumber"]

    P = fert[["placement"]].to_numpy(dtype=float)
    PU = unk[["placement"]].to_numpy(dtype=float)
    S = fert[short].to_numpy(dtype=float)
    SU = unk[short].to_numpy(dtype=float)

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(6.4, 2.5), gridspec_kw=dict(width_ratios=[1, 5], wspace=0.08))

    for ax, M, U, cols, labels, title in (
            (axl, P, PU, ["placement"], ["piece\nplacement"],
             "dominates the record"),
            (axr, S, SU, short, slabel, "the five short fields")):
        vmin = np.nanmin(np.where(U > 0, np.nan, M))
        vmax = np.nanmax(np.where(U > 0, np.nan, M))
        span = vmax - vmin if vmax > vmin else 1.0
        ax.imshow(np.where(U > 0, np.nan, M), cmap=SEQ_CMAP,
                  vmin=vmin - 0.45 * span, vmax=vmax + 0.06 * span, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if U[i, j] > 0:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               color="#ecebe4", zorder=2))
                    ax.text(j, i, "lossy", ha="center", va="center",
                            fontsize=6.5, style="italic", color=MUTED, zorder=3)
                else:
                    frac = (M[i, j] - (vmin - 0.45 * span)) / (span * 1.51)
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=8.2, zorder=3,
                            color="white" if frac > 0.62 else INK)
        ax.set_xticks(range(len(cols)), labels, fontsize=7)
        ax.set_title(title, fontsize=8.2, color=MUTED, style="italic", pad=5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        heat_grid(ax, M)

    axl.set_yticks(range(6), [LABEL[t] for t in ORDER], fontsize=8.5)
    axr.set_yticks([])
    fig.text(0.5, -0.10, "tokens per field (each panel on its own colour scale; "
             "darker is more tokens)", ha="center", fontsize=8.2, color=INK2)
    fig.tight_layout()
    fig.savefig(rf"{FIGS}/fig_fen_fields.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_emission():
    df = pd.read_csv(rf"{DATA}/emission_by_category_word_initial.csv")
    cats = ["castling", "piece_letters", "markers", "squares",
            "san_promotions", "uci_moves", "fen_atoms", "top20_san"]
    clabel = ["castling idioms (2)", "piece letters (5)", "markers x + # = (4)",
              "64 squares", "SAN promotions (4)", "UCI moves (4)",
              "FEN atoms (4)", "top-20 SAN moves (20)"]
    P = df.pivot(index="category", columns="tokenizer", values="coverage")
    M = P.loc[cats, ORDER].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.imshow(M, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = 100 * M[i, j]
            txt = "." if v == 0 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.2,
                    color="white" if M[i, j] > 0.6 else (MUTED if v == 0 else INK))
    ax.set_xticks(range(len(ORDER)), [LABEL[t] for t in ORDER], fontsize=8.5)
    ax.set_yticks(range(len(cats)), clabel, fontsize=8.2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    heat_grid(ax, M)
    ax.set_xlabel("per cent of units emitted as exactly one token "
                  "(a dot marks none)", fontsize=8.2, labelpad=7)
    fig.tight_layout()
    fig.savefig(rf"{FIGS}/fig_emission_word_initial.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_convergence():
    """Left: absolute level. Right: per-cent deviation against the criterion."""
    df = pd.read_csv(rf"{DATA}/convergence.csv")
    sub = df[df.notation == "san"]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.4, 2.9),
                                   gridspec_kw=dict(wspace=0.30))

    MARK = dict(zip(ORDER, ["o", "s", "^", "D", "v", "P"]))
    DASH = dict(zip(ORDER, ["-", (0, (4, 1.5)), (0, (1, 1.2)), "-",
                            (0, (5, 1, 1, 1)), (0, (3, 1))]))
    handles = []
    for tk in ORDER:
        d = sub[sub.tokenizer == tk].sort_values("games")
        ln, = axl.plot(d.games, d.tokens_per_char, color=COLOR[tk], linewidth=1.4,
                       marker=MARK[tk], linestyle=DASH[tk], markersize=3.4,
                       zorder=3, label=LABEL[tk])
        handles.append(ln)
        final = d.tokens_per_char.iloc[-1]
        axr.plot(d.games, 100 * (d.tokens_per_char - final) / final,
                 color=COLOR[tk], linewidth=1.4, marker=MARK[tk],
                 linestyle=DASH[tk], markersize=3.4, zorder=3)

    axl.set_xscale("log")
    axl.set_xticks([100, 500, 1000, 2000, 5000, 8000],
                   ["100", "500", "1k", "2k", "5k", "8k"])
    axl.set_xlim(85, 9400)
    axl.set_ylim(0.25, 1.05)
    axl.set_ylabel("SAN tokens per character", fontsize=8.5)
    axl.set_xlabel("games in prefix (log scale)", fontsize=8.5)
    axl.set_title("absolute level", fontsize=8.2, color=MUTED,
                  style="italic", pad=5)
    style_ax(axl, ygrid=True)

    axr.axhspan(-1, 1, color="#eaf2fd", zorder=0)
    axr.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    axr.text(8000, 1.02, "one per cent diagnostic band", fontsize=6.5,
             ha="right", va="bottom", color=INK2)
    axr.set_xscale("log")
    axr.set_xticks([100, 500, 1000, 2000, 5000, 8000],
                   ["100", "500", "1k", "2k", "5k", "8k"])
    axr.set_xlim(85, 9400)
    axr.set_ylim(-1.35, 1.35)
    axr.set_ylabel("deviation from full-corpus value (%)", fontsize=8.5)
    axr.set_xlabel("games in prefix (log scale)", fontsize=8.5)
    axr.set_title("full-corpus deviation", fontsize=8.2, color=MUTED,
                  style="italic", pad=5)
    style_ax(axr, ygrid=True)

    fig.tight_layout()
    leg = fig.legend(handles=handles, fontsize=8.2, frameon=False, ncols=6,
                     loc="lower center", bbox_to_anchor=(0.5, -0.09),
                     handlelength=1.3, columnspacing=1.5)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.savefig(rf"{FIGS}/fig_convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_seeds():
    """Seed sensitivity had only a table. The point is visual: three draws land."""
    s = pd.read_csv(rf"{DATA}/seed_sensitivity.csv")
    s = s[s.notation == "san"]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    seeds = [42, 7, 2024]
    marks = ["o", "s", "^"]
    x = np.arange(len(ORDER))
    for si, (seed, mk) in enumerate(zip(seeds, marks)):
        devs = []
        for tk in ORDER:
            row = s[s.tokenizer == tk].set_index("seed")["fertility"]
            devs.append(100 * (row.loc[seed] - row.loc[42]) / row.loc[42])
        ax.scatter(x + (si - 1) * 0.13, devs, s=26, marker=mk, zorder=3,
                   color=[COLOR[t] for t in ORDER],
                   edgecolor="white", linewidth=0.5,
                   label=f"seed {seed}" + (" (released)" if seed == 42 else ""))
    ax.axhspan(-1, 1, color="#eaf2fd", zorder=0)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.text(len(ORDER) - 0.55, 1.02, "one per cent reference band", fontsize=6.5,
            ha="right", va="bottom", color=INK2)
    ax.set_xticks(x, [LABEL[t] for t in ORDER], fontsize=8.5)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylabel("SAN fertility, deviation\nfrom the seed-42 draw (%)", fontsize=8.5)
    leg = ax.legend(fontsize=7, frameon=False, ncols=3, loc="lower center",
                    handlelength=1.0, columnspacing=1.4,
                    bbox_to_anchor=(0.5, -0.02))
    for t, mk in zip(leg.legend_handles, marks):
        t.set_color(MUTED)
    for t in leg.get_texts():
        t.set_color(INK2)
    style_ax(ax, ygrid=True)
    fig.tight_layout()
    fig.savefig(rf"{FIGS}/fig_seeds.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_emission()
    print("corrected emission figure written to", FIGS)
