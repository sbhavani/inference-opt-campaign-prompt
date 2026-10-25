# BioNeMo ESM-2 campaign kickoff (setup, then the timed hour)

> Companion to `inference_opt_campaign_prompt.md`. Configured for a one-hour Brev H100 pilot.
> This message authorizes setup; the optimization clock starts after readiness. Editing these files does not start a run.

Execute the BioNeMo Recipes ESM-2 embedding-inference campaign defined in the attached campaign prompt.
Carry the prompt and the repository instructions into any delegated work. You are the orchestrator.
Proceed autonomously within the assigned resources; do not request routine implementation approvals.

This is a fresh campaign in a fresh candidate workspace with a separate immutable reference checkout.
Repository: https://github.com/NVIDIA-BioNeMo/bionemo-recipes
Commit: `fe8385e4cf4695dd15aca5665ae39aac51892a6f`.
Primary checkpoint: `nvidia/esm2_t33_650M_UR50D`; smoke checkpoint: `nvidia/esm2_t6_8M_UR50D`.
Use local model source at that commit and resolve immutable Hub revisions before loading weights.

Assigned device: one dedicated Brev H100 80 GB, single GPU. Prefer the quoted `gpu_1x_h100_pcie` at $3.948/hour
after rechecking its current price and capacity. Instance name: `esm2-opt-1h-20261024`.
Budget: **60 minutes of optimization after setup**, on one H100. Update every **10 minutes**.
Provisioning, environment installation, checkpoint download/checksum verification, and the 8M setup smoke
check happen before T0. Start T0 only once these are ready, record it in UTC, and set the optimization
deadline to T0 + 60 minutes. Reference checks, profiling, implementation, and final validation count against
that hour. Report setup time/cost separately; the instance still accrues charges during setup.
Stop GPU experiments at the deadline, export and verify results, and promptly perform authorized deletion.
Report cleanup time and actual provider billing separately. Do not promise an exact total charge.

This kickoff authorizes one single-H100 allocation for setup plus the one-hour optimization campaign, at the displayed accepted rate;
no multi-GPU or higher-cost fallback. Run `brev ls --json` first. If logged out, have the user run
`brev login`. Show the selected cost/type before creation. Confirm authorization to delete only this
named instance after saving results, or an explicit user choice to retain it with continuing billing,
before allocating; the quoted type does not support stop. Never touch unrelated instances.

Other resident GPU workloads invalidate the quiet-device protocol; report contention before benchmarking.

First actions, in order:

1. Record the model identifier, root run identifier (or an explicitly recorded locally generated identifier),
   and campaign tag `optkit-esm2-<YYYY-MM-DD>-<runid8>`. Read root and closer `AGENTS.md` files.
2. Verify the BioNeMo pin, immutable checkpoint revisions, checksums, image digest, and dependency versions.
   Freeze the 100-sequence minimum corpus, a distinct held-out correctness set, masks, and batch manifests.
   For this pilot, use total token lengths 128/512/1024 including BOS/EOS; batches 1 and 4 plus a mixed
   batch of 4. Keep the campaign prompt's 100-sequence held-out minimum. Defer the broader matrix explicitly.
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
   in the chat or external results bundle, not a planning document committed to the repository. Then attempt at most one focused change, reserving optimization minutes 40–55 for final verification
   and minutes 55–60 for analysis and results. Setup does not reduce this hour. If setup cannot finish,
   report its blocker and perform authorized cleanup without starting the optimization clock.

Investigate repeated RoPE/mask setup, packed token-dropout scaling, remaining pack/unpack costs, and graph or
compiler opportunities only as the profiles justify them. Existing THD/low-precision support is a baseline
feature. There is no triangle attention in this encoder and no assumed speed-up target.

At each update report budget and time remaining, accepted changes with measured gains, rejected approaches
with reasons, correctness failures, and the next measured opportunity. Serialize GPU benchmarks even when
CPU investigation is delegated. Never increase concurrency or spend to compensate for a deadline without
remaining assigned resources.

Finish with a scoped BioNeMo patch or draft PR, focused tests, regenerated mapped copies, reproducible
benchmark commands, and the clean final results bundle. Preserve the immutable reference. Report unavailable
GPU checks and unsupported modes honestly. Provision only the authorized single Brev instance; do not publish issues/messages without separate
authorization. Verify result copies and final instance state after authorized deletion. A job timeout alone
does not stop billing. Begin once Brev authentication and the named-instance cleanup choice are resolved.
