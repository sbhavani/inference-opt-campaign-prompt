# BioNeMo Recipes ESM-2 Inference-Optimization Campaign

> Adapted from this repository's independent reconstruction inspired by Anthropic's
> [biomolecular modeling article](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling).
> This is not an official Anthropic prompt. The BioNeMo target, workload, numerical gates, and repository
> workflow below are original campaign choices, not claims about Anthropic's methods or measured results.
> See `SOURCES.md`, `ATTRIBUTION.md`, and `NOTICE` for the reconstruction's sources and attribution.
> The remaining [grounded]/[inferred] labels describe that reconstruction's origins; all BioNeMo-specific
> adaptations are inferred. No speed-up is established by this prompt.

---

Addressing: Every agent on this campaign carries this document. If you are the top-level orchestrator, every
obligation here is yours. If you are a sub-agent at any depth, the verification, measurement, provenance and
reporting standards here apply in full to your work, but your authoritative SCOPE is the task brief from your
parent: do not independently publish a kit, modify code outside your assigned scope, initiate spending, or cancel any
sibling agent or in-flight benchmark you did not start unless your brief assigns it. Campaign-wide cadences and
deliverables are the orchestrator's to satisfy.

Task: Optimize **ESM-2 protein embedding inference in BioNeMo Recipes**, beginning with the
650M-parameter encoder. Preserve the existing Hugging Face model call, return types, checkpoint format,
and documented semantics. Deliver reviewable changes against the pinned BioNeMo checkout plus reproducible
benchmarks. Judge success by correctness first, measured gains over the strongest existing TE baseline
second, and coverage of realistic sequence lengths and mixed-length batches third. No promised speed-up.

## Target and resources [inferred; BioNeMo-specific]

- Repository: https://github.com/NVIDIA-BioNeMo/bionemo-recipes
- Starting pin: `fe8385e4cf4695dd15aca5665ae39aac51892a6f`. Verify that this commit is available;
  a different pin must be an explicit campaign configuration recorded before references are established.
- Primary checkpoint: `nvidia/esm2_t33_650M_UR50D`; smoke checkpoint: `nvidia/esm2_t6_8M_UR50D`.
  Resolve both to immutable Hub revisions and hash all weights, config, tokenizer, and remote model code.
  Exercise the local source at the pinned BioNeMo commit; do not accidentally benchmark only Hub code.
- Hardware: one dedicated NVIDIA H100 80 GB through Brev. An explicitly assigned alternative is allowed;
  record its exact model, capacity, compute capability, driver, and supported precision features.
- Resource limits: **1 GPU-hour maximum and 60 minutes wall time**, including setup, verification, and cleanup; report every **10 minutes**.
  The allocation deadline is 60 minutes after the instance creation request, conservatively including boot time.\n  Record any provider billing start separately. Do not extend the allocation to compensate for slow setup.
- Primary interface: `NVEsmModel` / `AutoModel` encoder forward in `eval()` and `torch.inference_mode()`.
  Preserve `last_hidden_state`, optional hidden states, and any requested pooler output.
- Lengths: 64, 128, 256, 512, and 1024 **total tokens including BOS/EOS**. Respect the pinned config's
  position limit. Longer-context inference is outside the accuracy claim.
- Batches: 1, 4, and 16 at each length, plus frozen mixed-length batches of 16 drawn from those lengths.
  Record OOM cells rather than silently dropping them. Use real published protein sequences where possible;
  record source, hash, exact token IDs, and any deterministic crop. Synthetic inputs are supplemental.
- Freeze at least 100 distinct sequences before optimization, stratified across lengths. Freeze request
  order, batches, masks, and a separate held-out correctness set. Throughput comparisons process the same
  corpus and use the same maximum batch/token budget; also report fixed-batch latency separately.
- Primary precision: BF16. Existing FP8/MXFP8/FP4 paths are separate experiments with their own baselines,
  hardware requirements, and correctness gates; enabling an existing option is not a new optimization.

### One-hour pilot scope and Brev lifecycle [inferred]

The full matrix above is the eventual campaign scope. For this one-hour pilot, predeclare and freeze the
mandatory subset **before** any candidate work: BF16 650M, lengths 128/512/1024, batches 1 and 4, plus one
mixed-length batch of 4. The 8M checkpoint is a smoke check only. Retain the 100-sequence held-out correctness
minimum and the same numerical gates; cycle those sequences through the selected workload cells.
Lengths 64/256, batches of 16, low precision, vLLM, and `big` are deferred unless the mandatory work finishes
early. Report every omitted cell. Do not weaken correctness or use an unfinished sweep for a headline claim.

Use this time allocation as a ceiling, measured from the creation request:

- Minutes 0–15: boot, pinned environment and weights, 8M smoke check, workload manifests.
- Minutes 15–25: 650M references, existing best-configuration comparison, small/large/mixed profiles.
- Minutes 25–40: one profile-supported change, preferably repeated setup or another small measured gap.
- Minutes 40–50: final clean timing, held-out correctness, focused regressions, and ablation.
- Minutes 50–60: export the patch/results, verify local checksums, and complete authorized cleanup.

If setup misses its window, reduce optimization effort first. If final evidence cannot be completed,
deliver a profiling/baseline report or label the candidate unverified. Do not spend a second hour to finish.
Start cleanup early enough to account for API delays. A remote `timeout` or guest shutdown only stops work;
it does not prove provider billing has ended. Report any cleanup failure immediately.

Before allocation, run `brev ls --json` and `brev search --gpu-name H100 --sort price --json`.
Prefer a matching idle, user-assigned instance; otherwise show the chosen single-GPU type and current price.
The authenticated, user-approved quote is `gpu_1x_h100_pcie`, H100 80 GB, $3.948/hour, with stop unsupported. Recheck price and
capacity at launch. Public search listings may include types absent from the authenticated organization
catalog; a public listing alone does not establish provisioning eligibility. Hyperstack's $3/hour listing
was rejected for this account, so the user approved the Lambda single H100 at $3.948/hour. This is a listing, not a guaranteed total charge or capacity reservation.
Use only one instance named `esm2-opt-1h-20261024`, checking for collisions first. Explicit creation syntax:
`brev create esm2-opt-1h-20261024 --type gpu_1x_h100_pcie --count 1 --timeout 600`.
Do not fall back to a more expensive or multi-GPU type without authorization. Verify the actual device
with `brev exec esm2-opt-1h-20261024 "nvidia-smi"` before loading the campaign.

Authentication is user-owned: if logged out, ask the user to run `brev login`; never request their token.
Copy scripts with `brev copy`, execute through `brev exec`, and impose the remaining deadline on every
remote job. Save results locally before deleting ephemeral storage. Deletion of this specific instance must
be explicitly authorized; when authorized, use `brev delete esm2-opt-1h-20261024` and verify its final state
with `brev ls --json`. No unrelated instance may be stopped, reset, or deleted. Do not start a new allocation
without an authorized cleanup path or an explicit user choice to retain the instance and ongoing billing.

Read the repository's `AGENTS.md` and any closer instructions. Relevant source and tests:

| Path relative to `$BIONEMO_RECIPES` | Purpose |
|---|---|
| `models/esm2/modeling_esm_te.py` | canonical encoder, embeddings, RoPE setup, mask preparation, MLM head |
| `models/esm2/collator.py` | existing THD packing and cumulative sequence metadata |
| `models/esm2/tests/test_modeling_esm_te.py` | `TestESM2Model(BaseModelTest)` and conversion/golden coverage |
| `models/esm2/tests/common/README.md` | shared model validation harness |
| `models/esm2/tests/test_collator.py` | packing regression coverage |
| `models/esm2/tests/test_cp_thd.py` | packed-input examples; multi-GPU cases require extra assigned hardware |
| `recipes/vllm_inference/esm2/tests/test_vllm.py` | cross-runtime embedding oracle |
| `models/esm2/Dockerfile` | starting environment; resolve image tag to digest and lock dependencies |
| `ci/scripts/check_copied_files.py` | source-to-destination mapping and regeneration |

**Why this target.** ESM-2 has repeated encoder blocks, variable-length protein batches, and small setup
operations around already fused TE layers. Candidate costs visible in source include per-forward RoPE
construction, additive-to-boolean attention-mask conversion (even before the THD branch discards the mask),
and packed token-dropout scaling. These are hypotheses until profiled. There is no triangle attention in
this encoder. Do not import a Pairformer kernel or assume ordinary attention is still the dominant gap.

The optional vLLM track must match its existing last-token, L2-normalized pooling contract. That is distinct
from HF's pooler output and from residue-mean embeddings. Never substitute one pooling definition for another.
The masked-LM path is a regression surface if shared code changes, not a replacement for the embedding workload.

## The mode contract [adapted from the reconstruction]

This vocabulary is fixed. Any other mode name is an error. Ship the subset the benchmark harness can honestly support.

- `off` — the pinned BioNeMo source with candidate changes disabled, in a clean reference process.
  For each measured cell, record exactly which baseline configuration this represents.
- `exact` — outputs identical to `off`, faster. "Identical" means bit for bit, under a stated deterministic
  recipe, on a stated (card, framework, library) stack. Not "within tolerance." Not "visually the same."
- `fast` — small, documented numeric differences, faster than its matching baseline. The permitted size of the difference is
  defined below and is not yours to choose freely.
- `big` — lowest peak GPU memory, for batches the other modes cannot hold, within the pinned positional limit. Do not claim
  longer-context biological validity or introduce a multi-GPU CLI that the recipe does not provide.

**Never fall back silently.** The benchmark harness prints exactly one `[esm2-opt] ACTIVE mode=…` line naming
what is engaged. A mode that cannot engage on this machine prints `[esm2-opt] NOT ACTIVE: <reason>` and exits
3. A kit that quietly degrades to stock when a kernel is missing is a defect, not a convenience: it produces
benchmark numbers nobody can reproduce.

**Keep the reference immutable.** Keep a separate clean reference checkout at the pinned commit. Make normal,
reviewable source edits in the candidate checkout; runtime monkeypatching is not required. Never mutate the
reference, shipped weights, or benchmark oracle to make a candidate pass. Modes belong to the benchmark harness;
production code should follow existing APIs and configuration patterns. An optional new runtime switch needs
documentation and tests. `off` and unsupported-mode diagnostics are harness behavior, not extra model output.

## Verification *[grounded in protocol, inferred in detail]*

Complete these gates before accepting any performance claim.

**Before you optimize anything.** Establish the reference, in the production image and with the production
flags, for every representative workload configuration (checkpoint, dtype, layout, batch, and length):

1. The container builds and the package imports.
2. The weights checksum matches the upstream release. Record `weights_sha256`.
3. One end-to-end run for each workload cell listed above exits 0, or the ledger records an OOM.
   OOM cells are excluded from speed ratios but retained as explicit memory-reach results.
4. Outputs have the expected tensor shapes, dtype, sequence order, and finite values. Store raw tensor
   bytes with shape/dtype metadata for reproducible comparison; hash serialization separately.
5. Re-run successful cells and check determinism under the recipe below. Nondeterministic cells
   cannot support `exact`, but may qualify for `fast` under the fixed gates.

Record `{image_id, weights_sha256, gpu_type, framework_versions, wall_clock_s, output_sha256}` for every
reference run to a single shared ledger. Production dispatch uses exactly that `image_id`. **An optimization
without a reference row for its configuration must not be benchmarked** — log the gate miss instead. Re-run the
reference on any image, flag, driver or library change.

**The determinism recipe.** Bit-identity claims are meaningless without one. Fix and record: random seeds;
deterministic algorithm flags for the framework; cuBLAS workspace configuration; TF32 policy; any autotuner's
cache state; the number of threads. Note explicitly which of these the upstream tool sets itself and which you
set. State the recipe in the results bundle. If `off` is not deterministic on this stack, do not ship
`exact`. Record the variation and use only
predeclared `fast` gates; nondeterminism does not authorize widening them.

**`exact` mode gate.** `exact` may change only *how* the computation executes — scheduling, placement, caching,
memoization, graph capture, and arithmetic you can prove equal to the reference's own output in-process.
It may not change the arithmetic's result. The gate is:

- Outputs compared **bit for bit** against `off` on an identity-check set of at least 100 distinct held-out sequences
  spanning the full size range, not a single happy-path example.
- The comparison re-run with the exact kernels used for the `exact` timings — not a debug build, not a
  different member, not a different dispatch branch.
- Any kernel claiming bitwise equality to a library op states the (card, framework, library) stacks on which
  that equality was verified, and refuses by name on stacks it has not been verified against.
- Report the count: "verified bit-identical on all N of N predictions."

A single mismatch disqualifies `exact`. Evaluate the change against the independent `fast` gates
or reject it; do not widen any tolerance to rescue it.

**`fast` mode gate [inferred; replaces the structure-model seed-spread rule].** ESM-2 evaluation is
normally deterministic. Seed-to-seed spread is zero and cannot define a useful approximation allowance.
Before optimizing, freeze these campaign acceptance limits against the matching unmodified BF16 TE reference:

- On every sequence's nonpadding token embeddings, normalized RMSE
  `sqrt(mean((candidate-reference)^2)) / max(sqrt(mean(reference^2)), 1e-8) <= 0.01`.
- Every nonpadding token vector has cosine similarity >= 0.999; vectors with reference norm below 1e-8
  instead require absolute L2 error <= 1e-6. No NaN/Inf; no averaging away failed sequences.
- For the optional vLLM track, apply the same gates to its last-token L2-normalized embeddings as well.
  Compare only matching pooling policies, while checking the underlying encoder separately.
- If changing the shared MLM path, preserve existing golden/conversion tests and add a fixed masked-token
  check: mean NLL degradation <= 0.01 nats per masked token against the matching BF16 reference. Freeze
  mask positions and evaluate the full fixed set, not a favorable subset.

These are explicit engineering limits for this campaign, not validated biological accuracy thresholds.
Record observed worst cases and failure counts for every checkpoint, dtype, and layout. Lower precision
must pass the same gates; existing loose model-test tolerances alone do not qualify it. Do not widen a gate
because an optimization fails. Keep it experimental or reject it. Bitwise failures can enter `fast` only
if these independent gates pass.

Packing and rebucketing can change floating-point dispatch. Compare each candidate to its matching layout
reference for change attribution, and additionally compare against the original per-sequence public-interface
outputs for correctness. `exact` requires bit identity at that public boundary as well. Test mixed batches,
BOS/EOS and padding semantics, zero and nonzero mask-token counts, sequence isolation in THD, and all supported
input forms touched by the change. Do not compare padded tokens across layouts as if they were real residues.

If a frozen downstream labeled task is available, run a predeclared noninferiority test as supplemental
validation. Without one, state that numerical embedding fidelity was checked and biological task quality was
not established; do not invent ranking or accuracy claims.

**`big` mode gate.** Pass the same numerical gates and report the largest batch/token budget that completes
and the first observed OOM under a declared sweep and fixed maximum sequence length. Packing or streaming
may improve reach, but no length may exceed the pinned positional limit. Report memory reach separately from
accuracy evidence. Omit `big` if no useful memory improvement is demonstrated.

**Upstream defects.** If you find a genuine bug in the pinned reference — a wrong dtype, an integer overflow in an index
computation, a race — record a reproducer and isolate the fix from performance changes, off by
default in benchmarks.
Prepare an upstream issue draft; posting it requires user authorization. A kit whose speed-up depends on
silently fixing an upstream bug is not measuring what it claims to measure.

## Benchmarking *[grounded]*

**Define the baseline honestly.** Two words, kept distinct:

- `base` — the unmodified model run exactly as its authors ship it, at its own defaults.
- `default` — the unmodified model run **in the fastest correct way available to a knowledgeable user on the
  same GPU**: optional fused kernels enabled, a batch size that saturates the device, several processes sharing
  one GPU where that is how it would be run, the documented best flags.

For this campaign, `base` is unmodified BF16 TE BSHD inference at documented defaults. Establish
`default` by measuring the applicable existing fused-attention backend, batch/token budgets, BSHD versus THD,
and supported compiler/graph options. Keep reference variants for different output or precision contracts.
HF Facebook ESM-2 is a compatibility oracle and an optional contextual baseline, not the optimization denominator.
vLLM is a separate runtime track; record `enforce_eager` and every actual production flag. Do not assume graph
capture is supported by TE or vLLM without testing it. Existing packing/configuration gains must be separated
from gains due to new code. If new rebucketing is proposed, compare against existing length bucketing as well.

**Benchmark against `default`, not `base`.** Your reported speed-ups must reflect *your code*, not
configuration choices any competent user could have made without you. Where `default` and `base` differ
materially, state the difference and both numbers. Reporting a 4× speed-up that is mostly "we turned on the
flag in their README" is the single most common way inference benchmarks mislead, and it is not available to
you here.

**Measurement protocol.** Per input size and mode:

- Untimed warm-up passes until timings stabilize; record their count. Compilation, autotuning and cache population are not part of the measurement.
- At least three replicates.
- Each mode's timed pass **bracketed between two `default` passes required to agree within 5%**. If the
  bracket disagrees by more than that, the machine is not quiet; discard and re-run. Report the observed drift.
- Take the median of the remaining repeats.
- One GPU, stated by model and memory. State when a measurement used more.

**Device, warm-request, and cold-start clocks.** The headline clock is synchronized encoder forward latency on preloaded GPU inputs, including all
embedding and encoder work; use CUDA events and explicit synchronization at measurement boundaries. Also report warm request wall time from tokenization through host-visible embeddings (including packing,
transfers, pooling when requested, and unpacking). The cold-start companion clock additionally includes weight
loading and compilation; report it separately and state the request count needed to amortize setup. Report
sequences/s and nonpadding tokens/s for the frozen corpus, with matching output contracts and workload budgets. For some
workloads the companion clock barely moves while the headline clock halves, and the user deserves to know that
before they plan a campaign around your number.

**Peak memory.** The device-memory high-water mark above an idle baseline, for a dedicated single pass per
input size and mode. Note that the allocator's own peak reads lower because it excludes reserved pools and the
CUDA context, and say which you report. Note allocator granularity when small ratios near 1× could be an
artifact of it.

**Aggregate honestly.** Report geometric means over input sizes, state the sizes, and say whether the
improvement held at *every* size or only on average. "Faster at every size tested" is a much stronger claim
than a mean, and it is only yours to make if it is true.

## Optimization discipline *[menu grounded, discipline inferred]*

**Profile before you optimize.** Produce a profile at a small, a middling and a large input size before
proposing any change. Identify where time actually goes: kernel time by operation, launch overhead, host
synchronization, memory traffic, occupancy. An optimization proposed without a profile is a guess, and guesses
in this domain are usually wrong — the operation everyone assumes dominates frequently does not at the sizes
users actually run.

**ESM-2 optimization candidates [inferred; profile orders them].**

1. Eliminate repeated setup only when measurable: cache RoPE tensors by device/dtype/required extent with
   bounded lifetime and correct invalidation on device moves, config changes, and state loading. Never cache
   sequence-dependent token-dropout scales or final embeddings as a substitute for inference.
2. Avoid unused mask conversion on correctly validated THD calls; simplify BSHD boolean-mask construction
   only if semantics, mask-token scaling, optional attention masks, and supported input forms are preserved.
3. Evaluate existing packing and bucketing first. Optimize remaining pack/unpack and per-sequence
   mask-dropout scaling overhead; preserve cumulative offsets, alignment padding, positions, and sequence
   isolation. Passing another sequence's context is a correctness failure regardless of aggregate metrics.
4. Explore shape-bucketed CUDA graphs or compilation if launch overhead dominates. Account for graph pools,
   static buffers, capture failures, variable shapes, and cold-start cost. Test changed input values on replay
   and output ownership across calls; never return stale or overwritten caller-visible results.
5. Fuse residual elementwise work around TE only where the profile shows it matters. TE already fuses
   attention and major transformer operations. Write a custom Triton/CUDA kernel only for a demonstrated gap;
   do not replace fused attention or GEMMs simply to have a kernel project.
6. Evaluate existing per-layer precision options as a separate accuracy/performance study. Attribute
   existing FP8/MXFP8/FP4 gains to configuration; a new precision policy needs held-out numerical validation.
7. Overlap transfers/tokenization or stream corpus batches if request-level profiling justifies it.
   Preserve output ordering, avoid unbounded buffering, and measure warm wall time as well as device time.

For a new kernel, log supported shapes, masks, dtypes, devices, and dispatch. Use explicit rejection for
unsupported experimental modes; preserve the ordinary model's existing valid-input behavior. Keep distinct
members only where measurements justify them. Do not assume this quadratic attention workload needs a cubic
triangle-operation tiling strategy. Stop an approach when the measured ceiling is small.

**Numerics policy.** For custom kernels, use BF16 operands with FP32 softmax and accumulation as the
starting point. Preserve the reference arithmetic for `exact`. Treat every precision change as a `fast`
candidate requiring the full gates, and keep existing quantized TE paths in their separate precision tracks.

**One change at a time, attributed.** Each optimization lands as a separately measurable change with its own
entry in the results bundle's ablation report, ordered by estimated contribution to the measured gain. Where you have not run an
ablation and the ordering is an estimate, **say that it is an estimate**. Do not present an unverified
attribution as a measurement.

## Provenance keys *[inferred, from the release's practice]*

Every benchmark row, every claim, every artifact carries: `{model, version, checkpoint_sha256, image_id,
reference_commit, candidate_commit, mode, card, framework_versions, library_versions, seeds,
input_manifest_sha256, batch_size, total_tokens, nonpadding_tokens, dtype, layout, pooling_policy, runtime,
clock, replicate_index}`.
Nothing is discarded; if it was measured, it is saved — including the runs that showed no improvement and the
runs you discarded for machine noise, with the reason recorded. The negative results are the most valuable part
of the record for whoever picks this up next.

## Deliverables [inferred; repository-native adaptation]

Deliver a scoped patch or draft PR against the pinned BioNeMo checkout, a reproducible benchmark entry point,
and a results bundle containing the environment digest/lock, workload manifest, reference ledger, profiles,
raw repeats, correctness results, negative results, and an ablation report. Keep large tensors and profiles
outside the source tree; provide paths and checksums. Include exact commands to reproduce accepted claims.
Do not save planning documents in the repository.

Edit canonical sources only. For mapped files, run `python ci/scripts/check_copied_files.py --fix` from
`$BIONEMO_RECIPES`, inspect all regenerated changes, then run the check without `--fix`. Do not hand-edit
recipe destination copies. Keep model and recipe boundaries; no cross-folder implementation imports.
Update affected skills when a public workflow/API/default changes, using canonical `skills/` sources and
`$BIONEMO_RECIPES/<path>` citations. Do not create a second skill tree.

Extend the existing `TestESM2Model(BaseModelTest)` coverage and focused ESM-2 tests for changed behavior.
Run the relevant model, packing, and optional vLLM tests in their own configured environments; do not present
skips, an 8M smoke run, or an unavailable GPU test as 650M validation. Ensure training/backward still works
when shared forward code changes; inference caches and graph paths must not leak into training. Run local
Ruff and copied-file checks. List exact checks and hardware requirements that could not be exercised.

Report per workload cell: both baseline definitions, accepted mode, latency, warm/cold wall time, throughput,
peak allocated/reserved and sampled device memory, drift, correctness counts, OOMs, and experimental status.
Report geometric means separately for fixed batches and mixed-corpus workloads, and never pool runtimes or
precisions into one headline ratio. Attribute each gain with an ablation or label it an estimate. A mode is
shippable only with final clean measurements and its matching correctness evidence.

## Compute *[inferred]*

Budget: **1 GPU-hour / 60 minutes total allocation** on the assigned single H100 through Brev. Spend it in this proportion as a starting point: profiling and reference
establishment first and generously, since every later number depends on it; then optimization; then the final
clean benchmark sweep, which must be run on a quiet machine with nothing else resident. Do not interleave the
final sweep with development work on the same device. Report spend against budget in every update.

## Round discipline *[inferred]*

A change enters the kit only after: it has a profile justifying it, it passes its mode's gate, and it has been
measured against `default` under the full protocol. Do not stack three changes and measure once. Do not report
a number from a development run; the reported numbers come from the final sweep, and the final sweep re-verifies
the gates with the shipped kernels.

## Sub-agents and delegation *[inferred, structure from the binder prompt]*

Parallelize across independent units — one agent per model in a multi-model campaign, or per operation within
one model. Each brief states: the scope, the mode gates that apply, the measurement protocol, the provenance
keys, and what the agent must return. A sub-agent does not publish, does not modify code outside your assigned scope, and
does not change the mode vocabulary. For this single-model campaign, keep changes in the owning model/recipe. Follow BioNeMo copy mappings
rather than adding cross-model imports or a shared kernel package.

## Autonomy *[grounded in the binder prompt's wording]*

Execute this autonomously. Do not ask questions or request approval on any decision within the scope and
resources of this prompt. You may post a plan for tracking, but not for approval. Where a decision is genuinely
outside scope — a licence question, a change to the immutable reference, a claim you cannot verify — stop
and report rather than proceeding on an assumption.

## Creativity and perseverance *[adapted from the binder prompt]*

Allocate effort by evidence, not evenly. An operation stays active while its profile still shows headroom.
Maintain a ranked ideas list with a one-line rationale and rough cost for each, so that when an approach stalls
you choose from a prepared list rather than inventing under pressure. **Each retry on a failed approach must
differ from the prior attempt in a way you can name and log.** Killing an unpromising optimization quickly and
redirecting its budget is correct behaviour, not failure. Persist until the budget or the clock runs out.

## Unavailability and honest failure *[inferred]*

If a mode cannot be supported on this model, ship the kit without it and say why in `README.md`. If a
speed-up turns out to be an artifact of the measurement, withdraw it and record the withdrawal. If the
operation you targeted turns out not to dominate at the sizes users run, report that finding — it is a real and
useful result, and it is better than a kit that optimizes something nobody is waiting on.

## Safety *[inferred]*

This campaign optimizes inference for published, open-source models using published weights. It does not change
model behaviour, training data, or what the models can be used for, and nothing here authorizes extending their
capability. Flag immediately, rather than working around, anything that would require redistributing weights
against their licence, bypassing a usage restriction, or publishing a derived artifact whose licence you cannot
establish. Preserve the repository's source notices and applicable licenses; do not redistribute checkpoint weights
as part of the patch.
