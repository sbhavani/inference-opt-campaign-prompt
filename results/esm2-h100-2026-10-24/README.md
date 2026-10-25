# ESM-2 H100 pilot — October 24, 2026 (Pacific)

A scoped BioNeMo source patch enables CUDA graph capture for packed THD mask-dropout scaling.
On one frozen mixed-length batch, it measured **2.07× faster encoder forward** and **1.83× faster warm
requests** than the existing unmodified BSHD graph configuration. These results describe this pilot workload
and recorded stack.

Unmodified THD capture failed at `torch.repeat_interleave`, which synchronized to infer the output length.
Supplying `output_size=input_ids.numel()` removes that synchronization without changing the arithmetic.
[PyTorch documents this behavior](https://docs.pytorch.org/docs/2.14/generated/torch.repeat_interleave.html).
The gain comes from combining existing packing and graphs after fixing their compatibility; it is not a new
attention kernel. The reviewable source patch, regenerated recipe copies, and tests are saved in the BioNeMo
working tree and local task artifacts.

| Total-token workload | Existing BSHD graph (ms) | Patched THD graph (ms) | Speed-up |
|---|---:|---:|---:|
| l128-b1 | 3.778 | 3.040 | 1.24× |
| l128-b4 | 4.910 | 4.156 | 1.18× |
| l512-b1 | 5.073 | 4.151 | 1.22× |
| l512-b4 | 11.890 | 11.650 | 1.02× |
| l1024-b1 | 7.666 | 7.119 | 1.08× |
| l1024-b4 | 23.917 | 23.662 | 1.01× |
| mixed | 22.085 | 10.646 | 2.07× |

Fixed-batch geometric means across 128/512/1024 tokens were **1.18× for batch 1** and **1.07× for batch 4**.
Gains at the largest uniform batches are small. The mixed batch was [128, 512, 1024, 128], including BOS/EOS.

Mixed warm request time was **28.84 → 15.71 ms**, including tokenization, transfers, packing, replay,
unpacking, and host-visible embeddings. Cold requests remained around **18 seconds** for both configurations;
no meaningful cold-start improvement is claimed. Three candidate cold requests were bracketed by four
baseline requests, with checkpoint files already cached. Cold measurements include imports, loading,
warm-up, capture, and output writing; container startup and weight downloads are excluded.

The final before/candidate/after sweep used five repeats, each averaging 20 forward calls, after at least
100 warm-up calls and stabilization. Maximum repeat-level baseline drift was **0.81%**, below the 5% gate.
All warm-up and cold-bracket gates passed. Existing scratch-fill settings and ordinary BSHD graph capture
were evaluated as baseline configurations and receive no new-code credit. A separate `torch.compile`
reduce-overhead trial was stopped after initial 55–58 ms cells and TE graph-break/eager-fallback warnings;
its full numerical verification was not completed.

A dtype/shape/raw-byte audit passed **219/219 candidate outputs**: 100 held-out single predictions,
100 predictions from those same proteins in 25 masked mixed batches, and 19 timed-case outputs.
All cold outputs also matched. **11 focused GPU tests passed**, covering new capture/replay, changing masks,
alignment padding, and gradients, plus inherited forward/backward/golden checks. Ruff, copied-file
consistency, patch applicability, and whitespace checks passed.

Timing inputs use the first four of 200 frozen reviewed UniProt proteins with deterministic N-terminal crops.
The last 100 proteins were held out for verification. This is a shape-focused pilot; no broad corpus-throughput,
downstream biological-quality, low-precision, multi-GPU, or vLLM runtime claim is made. Allocator memory metrics
are recorded separately from whole-device memory; no capacity improvement is claimed.

Setup took **17 minutes 20 seconds** and was excluded from the optimization budget. The optimization clock
started at **2026-10-25 05:41:48 UTC**, with a 60-minute deadline. Verification and analysis finished at
**2026-10-25 06:13:53 UTC**, about 32 minutes into that budget.

See [summary.json](summary.json) for final aggregates and verification counts, and
[environment.json](environment.json) for the recorded pins. Byte identity is verified on that stack;
a rebuild must re-establish references. The complete 1.55 GB raw archive is saved in local Codex task
artifacts, with SHA-256 `2b4d65e2306298f2f064f322f5930566d5cf49f0589e6336d75259388c8a871a`. It includes tensors, profiles, manifests, the source patch,
harnesses, environment locks, logs, and negative findings. Checkpoint weights are excluded.
