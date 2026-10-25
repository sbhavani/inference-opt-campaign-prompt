# Inference-Optimization Campaign Prompt (a reconstruction)

> ### ⚠️ This is not an Anthropic document
>
> This repository contains a **reconstruction** — an independent, inferred prompt written by reading
> Anthropic's *published* technical report and code release. **It is not a leaked, recovered, or official
> Anthropic artifact.** Anthropic did not publish the prompt that drove the work described in
> ["How Claude is uplifting biomolecular modeling"](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling);
> all 8,123 files in the accompanying code release contain no agent prompt, no `CLAUDE.md`, and no methodology
> document. Nothing here should be cited as evidence of how that work was actually prompted.

## What this is

A ~3,200-word campaign prompt for driving an autonomous **GPU inference-optimization** project — profile a
model, optimize its inference path, and prove you did not change its outputs — plus the short kickoff message
that would start a run.

It exists because the published report describes a striking result (36 model implementations optimized in just
under four weeks, supervised by two people with no prior kernel-engineering experience) without describing the
instructions that produced it. The interesting question is what such a prompt would have to *contain*. This is
an attempt to answer that from the evidence available.

## Contents

| file | what it is |
|---|---|
| [`prompts/inference_opt_campaign_prompt.md`](prompts/inference_opt_campaign_prompt.md) | the campaign prompt (system context) |
| [`prompts/kickoff_message.md`](prompts/kickoff_message.md) | the T0 user message that starts a run |
| [`SOURCES.md`](SOURCES.md) | every source, with what was taken from each |
| [`ATTRIBUTION.md`](ATTRIBUTION.md) | CC-BY-4.0 attribution for reused material |

Sections inside the prompt are tagged **[grounded]** where they restate something the report or repo states
directly, and **[inferred]** where they are original construction in the published style. Roughly 60% is
grounded.

## The structural argument

The thesis is in the proportions. In Anthropic's published protein-binder campaign prompt — a real artifact,
released under CC-BY-4.0 — the `Verification:` section runs **2,205 words**, longer than the task description
by a wide margin. This reconstruction keeps that ratio deliberately: the task statement is one paragraph and
verification is the longest section in the document.

The claim that follows is that **the prompt is mostly an oracle and a provenance contract, not a task
description.** Three gates carry the weight, and each is written so the agent cannot negotiate with it:

1. **`exact` mode** demands bit-for-bit identity against unmodified upstream, re-verified with the shipped
   kernels, and explicitly forbids widening a tolerance to rescue a change — the change moves to `fast` instead.
2. **`fast` mode** defines permitted error *by the model's own seed-to-seed spread*, not by the agent's
   judgment. The tolerance is derived from the system under test, so it cannot be argued upward.
3. **`big` mode** must report reach and accuracy separately, so a capability claim never implies an accuracy
   claim.

Two things were added that the report's methodology implies but does not spell out, because they are where this
kind of campaign actually goes wrong: **profile before optimizing** (the operation everyone assumes dominates
frequently does not, at the input sizes users actually run), and **say when an attribution is an estimate** —
the report does this itself, noting "no ablation is available to confirm this order."

## Using it

Fill the `<ANGLE_BRACKET>` placeholders (target model and pins, GPU, representative input sizes, identity-check
set size, seed count, budget). Two cautions learned from writing it:

- The measurement protocol requires a **quiet, dedicated** device. The rule that each timed pass be bracketed
  between two baseline passes agreeing within 5% exists to detect contention; on a shared or bursting instance
  it will fail constantly, and correctly.
- Pick a target whose profile you already know, so reference establishment does not consume the budget. That
  step is deliberately expensive.

## Licence

[CC BY 4.0](LICENSE). See [`ATTRIBUTION.md`](ATTRIBUTION.md).
