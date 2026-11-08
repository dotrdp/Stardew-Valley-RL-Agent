# Papers

Typst sources (Typst ≥ 0.12) for the two papers. Build from the repository root:

```bash
typst compile --root papers papers/furrow/main.typ     # Furrow benchmark paper
typst compile --root papers papers/speedrun/main.typ   # learning-to-speedrun paper
```

`*/generated/` (tables, figures, `keys.typ` with headline numbers) is written by
`python -m svrl.pipeline --stages report` from `runs/v1/results.json`; numbers are never typed by
hand. `common/refs.bib` contains only references that were checked (see `research/`).
