# BioNeMo ESM-2 campaign kickoff (the T0 user turn)

> Companion to `inference_opt_campaign_prompt.md`. Fill the three resource placeholders before sending.
> This message starts an optimization run; editing these prompt files does not start one.

Execute the BioNeMo Recipes ESM-2 embedding-inference campaign defined in the attached campaign prompt.
Carry the prompt and the repository instructions into any delegated work. You are the orchestrator.
Proceed autonomously within the assigned resources; do not request routine implementation approvals.

This is a fresh campaign in a fresh candidate workspace with a separate immutable reference checkout.
Repository: https://github.com/NVIDIA-BioNeMo/bionemo-recipes
Commit: `fe8385e4cf4695dd15aca5665ae39aac51892a6f`.
Primary checkpoint: `nvidia/esm2_t33_650M_UR50D`; smoke checkpoint: `nvidia/esm2_t6_8M_UR50D`.
Use local model source at that commit and resolve immutable Hub revisions before loading weights.

Assigned device: one dedicated NVIDIA H100 80 GB, or the explicitly assigned substitute recorded at T0.
Budget: `<GPU_HOURS>` GPU-hours. Wall-clock deadline: T0 + `<HOURS>` hours. Update every `<CADENCE>`.
T0 is receipt of this completed kickoff. Record its timestamp and the deadline in UTC.
If placeholders are unresolved or no GPU is assigned, report the missing resources and perform only local
inspection; do not start the clock, provision compute, or infer spending authority.
Other resident GPU workloads invalidate the quiet-device protocol; report contention before benchmarking.

First actions, in order:

1. Record the model identifier, root run identifier (or an explicitly recorded locally generated identifier),
   and campaign tag `optkit-esm2-<YYYY-MM-DD>-<runid8>`. Read root and closer `AGENTS.md` files.
2. Verify the BioNeMo pin, immutable checkpoint revisions, checksums, image digest, and dependency versions.
   Freeze the 100-sequence minimum corpus, a distinct held-out correctness set, masks, and batch manifests.
   Use total token lengths 64/128/256/512/1024 including BOS/EOS; fixed batches 1/4/16 and mixed batches of 16.
3. Establish the reference ledger for BF16 TE workloads before optimizing. An 8M smoke run is not a 650M
   reference. Record OOM cells and unavailable checks. Block optimization on missing reference evidence.
4. Establish `base` (unmodified BF16 TE defaults) and `default` (best applicable existing TE kernels, layout,
   packing/bucketing, and runtime settings for the same output contract). Measure configuration-only gains
   separately. Keep precision and optional vLLM tracks separate. Never use stock HF as the headline denominator.
5. Freeze the campaign prompt's numerical gates: per-sequence nonpadding embedding NRMSE <= 0.01 and
   per-token cosine >= 0.999, with its near-zero-vector rule. Use bitwise comparison for `exact`.
   Seed spread does not define a tolerance for deterministic ESM-2. Preserve masked-LM regressions when
   shared code changes and the existing vLLM last-token L2-normalized pooling contract if that track is used.
6. Profile small, medium, large, and mixed-length workloads. Post a ranked list of measured opportunities
   in the chat or external results bundle, not a planning document committed to the repository. Then begin.

Investigate repeated RoPE/mask setup, packed token-dropout scaling, remaining pack/unpack costs, and graph or
compiler opportunities only as the profiles justify them. Existing THD/low-precision support is a baseline
feature. There is no triangle attention in this encoder and no assumed speed-up target.

At each update report budget and time remaining, accepted changes with measured gains, rejected approaches
with reasons, correctness failures, and the next measured opportunity. Serialize GPU benchmarks even when
CPU investigation is delegated. Never increase concurrency or spend to compensate for a deadline without
remaining assigned resources.

Finish with a scoped BioNeMo patch or draft PR, focused tests, regenerated mapped copies, reproducible
benchmark commands, and the clean final results bundle. Preserve the immutable reference. Report unavailable
GPU checks and unsupported modes honestly. Do not provision compute or publish issues/messages without
separate authorization. Begin when the resource fields and dedicated device are resolved.
