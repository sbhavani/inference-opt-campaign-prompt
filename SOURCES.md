# Sources

Everything in this repository was built from these, and nothing else.

## 1. The technical report

**"Accelerating open-source biomolecular models with Claude."** Claude Science, Richard Shuai, Rohil Badkundri,
Vincent Fan, Kilian Fatras, Lukas Jarosch, Amir Shanehsazzadeh. Anthropic, 17 September 2026 (last updated
21 September 2026). 140 pages.
<https://www-cdn.anthropic.com/e96b5807039a88168733d9687afe41dfbbd5de13.pdf>

Taken from it:

- The `default` vs `base` baseline distinction, and the instruction to benchmark against `default` so that
  reported speed-ups "reflect our changes, not configuration choices a knowledgeable user could make without
  our code."
- The measurement protocol: untimed warm-up, three replicates per input size, each mode's pass bracketed
  between two `default` passes required to agree within 5%, median of remaining repeats, H100 80 GB.
- The two-clock convention (the model's own forward/prediction call as headline; whole-pass wall time as
  companion) and the peak-memory definition (NVML high-water mark above an idle baseline).
- The bit-identity practice, e.g. OpenFold3 `exact` "verified bit-identical to default under a deterministic
  recipe… confirmed on all 435 predictions of the identity-check set."
- The `fast` accuracy guard expressed as the reference's own seed-to-seed spread.
- The optimization menu: memoizing recomputed intermediates, constant-folding branches with fixed predicates,
  CUDA-graph capture and replay, fusing chains of small operations, avoiding host–device stalls, chunking and
  multi-GPU sharding of the largest representations.
- The reach-without-accuracy reporting pattern (inference completed at up to 70,320 residues; predictions not
  accurate).
- The supervision claim: 36 model implementations in just under four weeks, supervised primarily by two staff
  "with experience in biomolecular modeling but none in inference optimization or kernel engineering."

## 2. The code release

**`anthropics/uplifting-biomolecular-modeling`** — a reference release accompanying the blog post; marked not
maintained and not accepting contributions. <https://github.com/anthropics/uplifting-biomolecular-modeling>

Taken from it:

- The **mode vocabulary** and its definitions: `off` (pinned upstream exactly as published), `exact` ("outputs
  identical to `off`, faster"), `fast` (documented numeric differences within the tool's own seed-to-seed
  variation), `big` (lowest peak memory).
- The **no-silent-fallback contract**: one `[<kit>-opt] ACTIVE mode=…` line, or `NOT ACTIVE: <reason>` and
  exit 3. "A kit never falls back to stock silently."
- The **never-edit-`stock`-in-place** rule and the per-kit layout (`stock/`, `opt/`, `environment/`,
  `configs/`, `run.sh`, `README.md`, `STOCK.md`, `CHANGES.md`, `upstream_issues/`).
- The `run.sh check` dry-run idea.
- Kernel technique, from `common/opt_core/README.md`: triangle-attention members using warp-specialized wgmma
  and TMA on `sm_90a` selected by sequence range, `mma.sync` and `cp.async` on `sm_80`, with bf16/fp16 operands
  and fp32 softmax and accumulation; dispatch between multiple members per operation.

## 3. The binder-design campaign prompts

**`Anthropic/claude-protein-binder-design`**, `prompts/` — CC BY 4.0.
<https://huggingface.co/datasets/Anthropic/claude-protein-binder-design>

A real published artifact from the same team, for a different task (de novo protein binder design). Supplied
the section grammar, the register, the system-prompt/kickoff split, and the verification-to-task length ratio.
Reused wording is itemized in [`ATTRIBUTION.md`](ATTRIBUTION.md).

## 4. Method references cited by the report

- Dao, T. et al. "FlashAttention: fast and memory-efficient exact attention with IO-awareness." NeurIPS 2022.
  arXiv:2205.14135.
- Tillet, P., Kung, H. T. & Cox, D. "Triton: an intermediate language and compiler for tiled neural network
  computations." MAPL 2019.
