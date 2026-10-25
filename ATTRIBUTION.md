# Attribution

## Reused material

`prompts/inference_opt_campaign_prompt.md` adapts wording and structure from:

> **Anthropic, "De novo miniprotein binder design campaign: prompt release bundle"**, in the dataset
> `Anthropic/claude-protein-binder-design`, <https://huggingface.co/datasets/Anthropic/claude-protein-binder-design>.
> Licensed **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>). Retrieved October 2026.
> Changes were made.

That dataset accompanies *"Autonomous de novo protein binder design with Claude"* (Claude Science and Amir
Shanehsazzadeh, Anthropic technical report, August 2026).

### What was reused, specifically

The adapted document targets a different task (GPU inference optimization, not protein design), so almost all
content is original. These passages are verbatim or near-verbatim from the source prompt, and are the reason
this repository carries CC BY 4.0:

| in this repo | source |
|---|---|
| "Every agent on this campaign carries this document. If you are the top-level orchestrator, every obligation here is yours. If you are a sub-agent at any depth, the verification, measurement, …" | the `Addressing:` paragraph of the multi-target campaign prompt |
| "…apply in full to your work, but your authoritative SCOPE is the task brief from your parent" | same paragraph |
| "Each retry on a failed approach must differ from the prior attempt in a way you can name and log." | `Creativity and perseverance:` |
| "choose from a prepared list rather than inventing under pressure" | `Creativity and perseverance:` |
| "You may post a plan for tracking, but not for approval." | `Autonomy:` |
| "This is a FRESH campaign in a FRESH workspace. It is not a resume of any prior run; at kickoff there is no prior campaign state of yours to reconcile against." (kickoff) | `single_target_kickoff.md` |

Also adapted at the structural level, without verbatim text: the labelled-section grammar
(`Task:`, `Verification:`, `Compute:`, `Deliverables:`, `Autonomy:`, `Safety:` …), the split between a long
system-context prompt and a short kickoff user message, and the deliberate choice to make verification the
longest section.

## Not reused

The source prompts themselves are **not** redistributed here. Read them at the dataset link above.

The technical report and code release described in [`SOURCES.md`](SOURCES.md) informed the *content* of this
reconstruction — the mode definitions, the benchmark protocol, the optimization menu — but no text was copied
from either. The report is © Anthropic; the code release is Apache-2.0 with `stock/` subtrees under their own
upstream licences.

## Trademarks

"Claude" and "Anthropic" are used nominatively, to identify the published work this reconstruction is about.
This repository is not affiliated with, endorsed by, or produced by Anthropic.
