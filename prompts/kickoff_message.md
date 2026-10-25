# Kickoff message (the T0 user turn) — reconstruction

> Companion to `inference_opt_campaign_prompt.md`. In the published binder campaigns the long prompt is the
> *system* context carried by every agent at every depth, and the kickoff is a short *user* message that starts
> the clock. This mirrors that split.

Execute the `<GPU_HOURS>`-hour inference-optimization campaign for `<MODEL>` exactly as specified in the
campaign prompt. The prompt is in your system context and is also attached to this message as a markdown file;
the system-context copy is authoritative and is what every sub-agent at every depth carries. You are the
top-level orchestrator. Do not ask me any questions or wait for approval.

This is a FRESH campaign in a FRESH workspace. It is not a resume of any prior run; at kickoff there is no
prior campaign state of yours to reconcile against. The GPU is dedicated to you for the duration — if you
observe other processes resident on the device, that is an anomaly worth reporting, not an expected condition,
because it invalidates the measurement protocol.

Clock: T0 is the timestamp of this message. End time is T0 + `<HOURS>` hours.

Your first actions, in this order:

1. Record your model identifier, your orchestrator root frame id, and the campaign tag
   `optkit-<model>-<YYYY-MM-DD>-<frameid8>`. Use it verbatim in every artifact name.
2. Bring up the pinned environment and run the full reference-establishment protocol from the Verification
   section at every size in `<SIZES>`, in the production image with production flags. Write the reference
   ledger before any optimization work begins. **Block on this.** Nothing downstream is meaningful without it.
3. Establish `default` — the fastest correct configuration a knowledgeable user could reach without your code
   — and record how it differs from `base`. State both.
4. Produce profiles at a small, a middling and a large input size, and post a ranked optimization plan with
   estimated yield and cost per item. Post it for tracking, not approval.
5. Then begin.

Report at `<CADENCE>`: spend against budget, what landed, what each landed change measured, what was killed and
why, and the current ranked list. If the clock rather than the budget becomes binding, say so in the next
update and scale concurrency rather than silently cutting planned work.

Begin now.
