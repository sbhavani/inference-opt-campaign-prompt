# BioNeMo Recipes ESM-2 Inference-Optimization Campaign

Two prompts for an autonomous campaign to optimize **ESM-2 650M protein embedding inference** in
[NVIDIA BioNeMo Recipes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes).

This adapts the repository's independent reconstruction inspired by
[Anthropic's biomolecular modeling article](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling).
**It is not an official, leaked, or recovered Anthropic prompt.** The target, workload, acceptance limits,
and repository workflow are our campaign choices. No performance result is claimed here.

| File | Purpose |
|---|---|
| [Campaign prompt](prompts/inference_opt_campaign_prompt.md) | Full target, verification, benchmark, and delivery contract |
| [Kickoff message](prompts/kickoff_message.md) | User message that starts a configured run |
| [Sources](SOURCES.md) | Original reconstruction sources and BioNeMo adaptation references |
| [Attribution](ATTRIBUTION.md) | CC BY 4.0 attribution for reused material |

## Why ESM-2

The encoder already uses Transformer Engine and supports packed inputs. The useful question is how much
headroom remains around that optimized baseline: repeated RoPE/mask setup, packed token-dropout scaling,
pack/unpack costs, and launch overhead. Profiles decide which opportunities deserve implementation.
These source-level observations are hypotheses, not established bottlenecks. This target has ordinary
encoder attention rather than triangle attention.

The campaign uses BF16 TE as its primary baseline, fixed lengths of 64–1024 total tokens, fixed batches
of 1/4/16, and mixed-length batches. Existing packing and precision options count as baseline configuration,
so their gains cannot be passed off as new code improvements. vLLM embedding inference is an optional
separate track with its existing last-token L2-normalized pooling contract.

## Using the prompts

1. Supply the campaign prompt as the run's governing instructions, subject to the host's instruction hierarchy.
2. Fill `<GPU_HOURS>`, `<HOURS>`, and `<CADENCE>` in the kickoff and matching campaign resource fields.
   Assign one dedicated H100 80 GB, or record an explicitly assigned substitute. These prompts do not
   authorize provisioning or unbounded spending.
3. Verify the pinned BioNeMo commit and resolve immutable checkpoint revisions and the container digest.
4. Send the completed kickoff. Merely editing these files does not start a campaign.

The campaign requires reference establishment before optimization, a quiet GPU, bracketed timing repeats,
and a final clean benchmark sweep. It produces a scoped BioNeMo patch or draft PR, focused regression tests,
and an external results bundle; canonical source changes regenerate mapped recipe copies.

`exact` requires bitwise equality. `fast` uses fixed numerical embedding gates, since deterministic ESM-2
has no useful seed-to-seed spread. These are engineering limits, not evidence of unchanged biological task
accuracy. `big` tests memory reach within the pinned positional limit. Unsupported modes are omitted.

## Licence

[CC BY 4.0](LICENSE). See [ATTRIBUTION.md](ATTRIBUTION.md) and [NOTICE](NOTICE).
