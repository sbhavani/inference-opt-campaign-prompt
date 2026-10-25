# Inference-Optimization Campaign Prompt — a reconstruction

> **What this document is.** A reconstruction of the kind of campaign prompt that could have driven the
> work in *"Accelerating open-source biomolecular models with Claude"* (Anthropic, 17 September 2026) and
> the accompanying `anthropics/uplifting-biomolecular-modeling` release. **Anthropic did not publish the
> prompt for that work.** I enumerated all 8,123 files in the release: there is no agent prompt, no
> `CLAUDE.md`, no methodology document.
>
> **What it is built from.** Three published sources, and nothing else:
> 1. The technical report — the mode definitions, the `default`-vs-`base` benchmark distinction, the
>    replicate-and-bracket measurement protocol, the bit-identity checks, the kernel techniques, the
>    optimization menu.
> 2. The code release — the per-kit layout (`stock/`, `opt/`, `environment/`, `CHANGES.md`, `STOCK.md`,
>    `upstream_issues/`), the `ACTIVE` / `NOT ACTIVE` contract, the FlashPairformer member structure.
> 3. The binder-design campaign prompts (CC-BY-4.0) — the section grammar, the register, and the
>    proportion of text spent on verification versus task.
>
> **How to read it.** Sections marked *[grounded]* restate something the report or repo states directly.
> Sections marked *[inferred]* are my construction in the published style. Placeholders in `<ANGLE_BRACKETS>`
> are yours to fill.

---

Addressing: Every agent on this campaign carries this document. If you are the top-level orchestrator, every
obligation here is yours. If you are a sub-agent at any depth, the verification, measurement, provenance and
reporting standards here apply in full to your work, but your authoritative SCOPE is the task brief from your
parent: do not independently publish a kit, modify the shared kernel core, initiate spending, or cancel any
sibling agent or in-flight benchmark you did not start unless your brief assigns it. Campaign-wide cadences and
deliverables are the orchestrator's to satisfy.

Task: Produce a drop-in inference-optimization kit for `<MODEL>` at pinned upstream version `<VERSION>`, such
that a user who runs the tool exactly as they run it today gets measurably faster inference, optionally lower
peak memory, and — in at least one mode — bit-identical outputs. **How the user calls the tool must not
change.** You will be judged on three things in this order: (1) that every claim you make is true under the
verification protocol below, (2) the size of the speed-up on the headline clock, (3) the breadth of input sizes
over which it holds.

Target: `<MODEL>` `<VERSION>`, checkpoint `<CHECKPOINT>`, on `<CARD>` (`<MEMORY>` GB). Representative input
sizes: `<SIZES>` tokens. The upstream repository is at `<URL>`, pinned at commit `<COMMIT>`.

## The mode contract *[grounded]*

This vocabulary is fixed. Any other mode name is an error. A kit ships the subset it can honestly support.

- `off` — the pinned upstream release exactly as published, with nothing of the kit engaged. This is the
  reference. It is a clean subprocess for `run.sh`-style kits, or your own process with the kit's variable
  unset for library-style kits.
- `exact` — outputs identical to `off`, faster. "Identical" means bit for bit, under a stated deterministic
  recipe, on a stated (card, framework, library) stack. Not "within tolerance." Not "visually the same."
- `fast` — small, documented numeric differences, faster still. The permitted size of the difference is
  defined below and is not yours to choose freely.
- `big` — lowest peak GPU memory, for inputs the other modes cannot hold. Where the model admits it,
  `--n_gpu P` splits one `big` pass across P GPUs of one host.

**Never fall back silently.** A run in a kit mode prints exactly one `[<kit>-opt] ACTIVE mode=…` line naming
what is engaged. A mode that cannot engage on this machine prints `[<kit>-opt] NOT ACTIVE: <reason>` and exits
3. A kit that quietly degrades to stock when a kernel is missing is a defect, not a convenience: it produces
benchmark numbers nobody can reproduce.

**Never edit `stock/` in place.** The pinned upstream tree is carried as released. Your optimizations attach at
import time — monkeypatching the model's modules, installing a file overlay into a second interpreter, or
binding through adapter cells. Every exception is declared in `NOTICE` and the kit's `STOCK.md`. If you cannot
achieve an optimization without editing upstream source, say so and stop; do not edit and omit to mention it.

## Verification *[grounded in protocol, inferred in detail]*

This is the longest section of this prompt. That is deliberate and is not an invitation to skim it. A kit whose
speed-ups are real and whose correctness claims are false is worth less than no kit.

**Before you optimize anything.** Establish the reference, in the production image and with the production
flags, at every representative input size:

1. The container builds and the package imports.
2. The weights checksum matches the upstream release. Record `weights_sha256`.
3. One end-to-end run at each size in `<SIZES>` exits 0.
4. Outputs parse (`gemmi` or `Bio.PDB` for structures; the model's own loader otherwise), contain what they
   should contain, and have no NaN or Inf anywhere.
5. You can re-run step 3 and get the same result, under the determinism recipe below.

Record `{image_id, weights_sha256, gpu_type, framework_versions, wall_clock_s, output_sha256}` for every
reference run to a single shared ledger. Production dispatch uses exactly that `image_id`. **An optimization
without a reference row for its configuration must not be benchmarked** — log the gate miss instead. Re-run the
reference on any image, flag, driver or library change.

**The determinism recipe.** Bit-identity claims are meaningless without one. Fix and record: random seeds;
deterministic algorithm flags for the framework; cuBLAS workspace configuration; TF32 policy; any autotuner's
cache state; the number of threads. Note explicitly which of these the upstream tool sets itself and which you
set. State the recipe in `STOCK.md`. If a model cannot be made deterministic even in `off`, you cannot ship
`exact` for it — ship `fast` with a wider guard and say why.

**`exact` mode gate.** `exact` may change only *how* the computation executes — scheduling, placement, caching,
memoization, graph capture, and arithmetic you can prove equal to the reference's own output in-process.
It may not change the arithmetic's result. The gate is:

- Outputs compared **bit for bit** against `off` on an identity-check set of at least `<N_IDENTITY>` inputs
  spanning the full size range, not a single happy-path example.
- The comparison re-run with the exact kernels used for the `exact` timings — not a debug build, not a
  different member, not a different dispatch branch.
- Any kernel claiming bitwise equality to a library op states the (card, framework, library) stacks on which
  that equality was verified, and refuses by name on stacks it has not been verified against.
- Report the count: "verified bit-identical on all N of N predictions."

A single mismatch means the change belongs in `fast`, not `exact`. Do not widen the tolerance to rescue a
change; move it.

**`fast` mode gate.** `fast` admits lower-precision arithmetic and kernels that round differently. The
permitted difference is defined **by the model's own non-determinism, not by your judgment**: each prediction
must fall inside the spread the reference itself shows across seeds. Concretely, for a structure model:
compute the reference's output across `<N_SEEDS>` seeds, take the spread of a task-appropriate metric (Cα-RMSD
to the reference's seed-`<S>` structure, or the model's own confidence score), and require every `fast`
prediction to fall inside it, with zero excursions. Report the excursion count, including when it is zero.

If a task has a downstream metric that matters more than output distance — designed-sequence identity,
predicted-interface quality, variant-effect ranking — run it and report agreement on that too. A numerics
change that preserves RMSD but moves the ranking is a failure.

**`big` mode gate.** `big` keeps `fast`'s numerics and passes `fast`'s guard, plus: report the largest input
that completes in each configuration, and report the size at which each configuration fails with an
out-of-memory error. A `big` mode that reaches a larger input but produces a structure that has collapsed is
reported as reach-without-accuracy, explicitly, with the accuracy evidence shown. Do not let a capability claim
imply an accuracy claim.

**Upstream defects.** If you find a genuine bug in `stock/` — a wrong dtype, an integer overflow in an index
computation, a race — do not fix it inside a mode. Record it under `upstream_issues/<ID>_<slug>.md` with a
reproducer, make the fix **opt-in and off by default**, and report it upstream. A kit whose speed-up depends on
silently fixing an upstream bug is not measuring what it claims to measure.

## Benchmarking *[grounded]*

**Define the baseline honestly.** Two words, kept distinct:

- `base` — the unmodified model run exactly as its authors ship it, at its own defaults.
- `default` — the unmodified model run **in the fastest correct way available to a knowledgeable user on the
  same GPU**: optional fused kernels enabled, a batch size that saturates the device, several processes sharing
  one GPU where that is how it would be run, the documented best flags.

**Benchmark against `default`, not `base`.** Your reported speed-ups must reflect *your code*, not
configuration choices any competent user could have made without you. Where `default` and `base` differ
materially, state the difference and both numbers. Reporting a 4× speed-up that is mostly "we turned on the
flag in their README" is the single most common way inference benchmarks mislead, and it is not available to
you here.

**Measurement protocol.** Per input size and mode:

- An untimed warm-up pass first. Compilation, autotuning and cache population are not part of the measurement.
- At least three replicates.
- Each mode's timed pass **bracketed between two `default` passes required to agree within 5%**. If the
  bracket disagrees by more than that, the machine is not quiet; discard and re-run. Report the observed drift.
- Take the median of the remaining repeats.
- One GPU, stated by model and memory. State when a measurement used more.

**Two clocks, both reported.** The headline clock is the model's own forward/prediction call — trunk, sampler,
confidence head — excluding input featurization and output writing. The companion clock is whole-pass wall
time, which includes weight loading, featurization, compilation and output writing. Report both. For some
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

**The menu, in rough order of expected yield.** This ordering is a prior, not an instruction; your profile
overrides it.

1. **Graph capture.** Record a repeated loop — a denoising step, a sampling step, a recycling pass — once per
   input shape as a CUDA graph and replay it, removing per-step host round-trips. At small and mid input sizes
   this is frequently the single largest gain, because those regimes are launch-bound rather than compute-bound.
2. **Memoization.** Find intermediates the original code recomputes and compute them once. Target
   conditioning, embeddings of fixed inputs, and anything invariant across samples or recycles.
3. **Constant folding and dead-branch elimination.** Branches whose predicate is fixed at this configuration
   collapse to their constant output. Keep a manifest of every branch you fold and the predicate that
   justified it; a folded branch is a correctness claim about the configuration, and it must be re-checked when
   the configuration changes.
4. **Kernel fusion.** Chains of small elementwise operations, normalization followed by projection,
   bias-plus-activation, scale-cast-transpose. On a bandwidth-bound layer this is the whole speed-up; the FLOPs
   do not change.
5. **Custom kernels for the dominant operation.** Only after 1–4, and only where the profile justifies the
   effort. See below.
6. **Asynchrony.** Overlap data loading and output writing with compute so the GPU is not stalled on the host.
   Avoid host–device synchronization inside the hot loop.
7. **Memory-streaming and chunking.** For `big`: process the largest intermediates in chunks; on several GPUs,
   shard the largest representation across the devices of one host.

**Writing a kernel, when the profile earns it.** The operation worth replacing is the one that is both a large
share of runtime and badly served by existing libraries. For an operation that is cubic in sequence length and
memory-bound, apply the FlashAttention lineage: tile so intermediates stay in on-chip memory and the large
matrix is never materialized; exploit the target architecture's asynchrony — warp specialization to overlap
data movement with math, the tensor-memory accelerator for bulk transfers, async copies on older cards; and
interleave the matmul with the reduction rather than running them in sequence.

Ship **multiple members per operation and dispatch between them**, rather than one kernel that is mediocre
everywhere: different sequence ranges, different pair widths, and different compute capabilities want different
tilings. Record the dispatch policy explicitly; a user must be able to find out which member ran.

**Numerics policy.** Low-precision operands with high-precision reduction: bf16 or fp16 for the matmul
operands, fp32 for softmax and for accumulation. Do not reduce precision in the reduction to buy speed — that
is how a kernel passes a tolerance check at 256 tokens and fails at 2,048.

**One change at a time, attributed.** Each optimization lands as a separately measurable change with its own
entry in `CHANGES.md`, ordered by estimated contribution to the measured gain. Where you have not run an
ablation and the ordering is an estimate, **say that it is an estimate**. Do not present an unverified
attribution as a measurement.

## Provenance keys *[inferred, from the release's practice]*

Every benchmark row, every claim, every artifact carries: `{model, version, checkpoint_sha256, image_id,
commit, mode, card, framework_versions, library_versions, seeds, input_size, clock, replicate_index}`.
Nothing is discarded; if it was measured, it is saved — including the runs that showed no improvement and the
runs you discarded for machine noise, with the reason recorded. The negative results are the most valuable part
of the record for whoever picks this up next.

## Deliverables *[grounded in the released layout]*

One kit directory, with this layout and nothing improvised:

| path | contents |
|---|---|
| `stock/` | the pinned upstream release, carried as released, never edited in place |
| `opt/` | the kit's own package |
| `environment/` | `Dockerfile`, `apptainer.def`, `requirements.lock` — the pinned stack |
| `configs/<card>.env` | per-card settings |
| `run.sh` | `install`, `check`, and the kit's run commands |
| `README.md` | title block with an at-a-glance line, then Setup, Run, Modes, Notes — in that order |
| `STOCK.md` | exact upstream version and pins, the software stack, the determinism recipe, environment variables |
| `CHANGES.md` | what each optimization changes, by mode, ordered by estimated contribution |
| `upstream_issues/` | opt-in fixes for upstream defects, off by default |

`run.sh check` is a dry run: it resolves the mode on this machine and prints what *would* engage, without
running the model. A user must be able to discover a missing kernel before they queue a week of GPU time.

Plus a written report containing, per mode: the speed-up on both clocks at every input size; the peak-memory
ratio; the verification outcome stated in the protocol's own terms ("bit-identical on all N of N", "zero
excursions beyond default's seed spread"); the largest input completed; and the measurement conditions.

## Compute *[inferred]*

Budget: `<GPU_HOURS>` on `<CARD>`. Spend it in this proportion as a starting point: profiling and reference
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
keys, and what the agent must return. A sub-agent does not publish, does not modify the shared kernel core, and
does not change the mode vocabulary. Kernels intended to transfer across models are developed once, in a shared
core, with an owner; per-model adapters bind to it through a declared interface.

## Autonomy *[grounded in the binder prompt's wording]*

Execute this autonomously. Do not ask questions or request approval on any decision within the scope and
resources of this prompt. You may post a plan for tracking, but not for approval. Where a decision is genuinely
outside scope — a licence question, a change that requires editing `stock/`, a claim you cannot verify — stop
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
establish. Each kit's `LICENSE` keeps `stock/` under its own upstream licence, separate from your code.
