# A Reproducible Audit of Tokenization Fragmentation in Chess Notations

Companion artifact for the BSc dissertation *A Reproducible Audit of Tokenization
Fragmentation in Chess Notations for General-Purpose Large Language Models*
(Yusuf Noori, BSBI / University for the Creative Arts, 2026).

The audit measures how six tokenizers (GPT-2, cl100k_base, o200k_base,
Mistral-7B, an open PGN-specific BPE tokenizer, and a rule-based grammar-aware
baseline) fragment the three standard chess notations (PGN/SAN, FEN, UCI) over
a seeded, stratified corpus of 8,000 rated Lichess games (532,673 moves,
49,674 board states). Every number reported in the dissertation traces to a
named file and column in `data/`.

## Layout

```
pipeline/   the executed audit code
data/       every measurement the dissertation cites, as CSV/JSON,
            plus the anonymised corpus (corpus.pgn) and stored arrays
figures/    the dissertation's figures as vector PDFs
```

## Reproducing the results

Python 3.14 with the pinned package versions in `requirements.txt`
(exact versions and tokenizer revisions are also recorded in
`data/pinned_versions.json`). All computation is CPU-only; the full audit
runs in under twenty minutes on an eight-core laptop.

Recompute every metric from the stored corpus (offline):

```
python pipeline/audit_full.py
python pipeline/emission_word_initial.py
python pipeline/supplementary_outputs.py
python pipeline/make_figures.py
python pipeline/make_corrected_emission_figure.py
```

Regenerate the corpus itself from the public Lichess dump (needs internet;
streams ~20 MB of the May 2026 export):

```
python pipeline/sample_corpus.py
python pipeline/seed_sensitivity.py
```

`audit_full.py` recomputes fertility, tokens per character, split rate and
unknown-token rates, with by-game clustered bootstrap intervals for
fertility and tokens per character;
`emission_word_initial.py` recomputes the deduplicated single-token emission
coverage under the registered word-initial convention;
`supplementary_outputs.py` recomputes the split-rate bootstrap intervals and
the per-game and per-record token distributions.

## Data and licensing

- **Code:** MIT License (see `LICENSE`).
- **Corpus:** derived from the [Lichess open database](https://database.lichess.org/)
  (CC0). Player usernames are anonymised; game URLs are retained for
  provenance, since the records remain public at source.
- Tokenizer artifacts are not redistributed; they are pinned by revision in
  `data/pinned_versions.json` and fetched from their canonical sources.

## Provenance

Developed for the dissertation named above.
