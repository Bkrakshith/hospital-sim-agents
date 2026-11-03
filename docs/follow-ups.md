# Follow-ups from the Plan 1 build (2026-11-03)

Known minor issues that reviewers found and deliberately deferred. None blocks running the system.

## Before the first real BONES-SEED download (scripts/fetch_bones_seed.sh in hospital-sim-humans)
- converter exit 0 with silent skips still deletes BVH (count mismatch caught but BVH lost); BATCH=500 unused; subshell OOM-kill without DONE marker hangs loop; strict M==N fails on stale outputs; generic `cough`/`walk_slow limp` patterns unanchored; HF_TOKEN env inherited by children; failed_batches mv nests on rerun

## Deferred minors by task
- Task 1: minor (deferred): loader never checks persona dict key == Persona.id
- Task 1: minor (deferred): _find_root walks up for personas.yaml implicitly; weak error outside a config tree
- Task 2: minor (deferred): NaN/Infinity accepted as floats; AudioEvent.source_kind plain str; extra="forbid" strict across processes (fine within wz.v1)
- Task 3: minor (deferred): secret regex sk-[...]{16,} false positives inside words (plan-mandated); OAuth/AUTH_TOKEN shapes not covered by redact; _file_key stat-then-open TOCTOU
- Task 3: minor (deferred): .bak overwritten on repeated hook changes; "Missing credentials" wording lacks a dedicated test; make_client catches broadly
- Task 4: minor (deferred): budget checked only before a call, a single call can overshoot the cap
- Task 4: minor (deferred): no empty-content test for the OpenAI backend; fake mode accepts any provider string
- Task 5: minor (deferred): fence toggle ignores marker type/length; "C#" trailing # stripped; heading-only sections give no chunk; missing research_notes yields no chunks silently
- Task 5: minor (deferred): atomic save has no rollback if the final rename fails after the old dir is moved aside; skipped bad-encoding notes are absent from the source hash
- Task 6: minor (deferred): pid reuse can keep a stale lock alive; reset/restore check-then-act window; begin_run MAX+1 without unique constraint; ensure_seed SELECT-then-INSERT race; snapshot json not atomic
- Task 6: minor (deferred): two acquirers can both judge a lock stale (small TOCTOU); restore_snapshot doesn't compare snapshot embedder meta with live one
- Task 7: minor (deferred): look_at may target PA/alarm/cough sources (caller should filter non-spatial); clock reversal blocks gaze; Entity.kind plain str; real signs.yaml facing_yaw are placeholders and could hide signs
- Task 7: minor (deferred): background cough schedule re-anchors to first tick after a bout (tick-dependent there)
- Task 8: minor (deferred): validate_decision does not match target type to intent kind (sit_down g1 passes); identical intents re-published every decision (dedupe in sim/agent node); summarise uses run_id while recall uses deps.run_index (Task 13 must pass the index from begin_run)
- Task 8: minor (deferred): _consider raising in _run tail makes drain re-raise; summaries runlog write on failed-summary path unguarded; same utterance re-reported on consecutive ticks could duplicate in held list; LTM write failing mid-summary loses remaining memories
- Task 9: minor (deferred): redacted key collisions merge dict entries; storyboard/deck sharing a basename would overwrite in copy_configs; iter_stream reads whole file
- Task 9: minor (deferred): bad-t test covers iter_stream only, not the two script classes
- Task 10: minor (deferred): AudioEvent ids uuid4 (non-deterministic across runs); shared rng makes partial-word drops call-order dependent
- Task 11: minor (deferred): chaos/b4 new_arrival with persona None always spawns the stub default persona (not seeded/visible); panel-supplied text with braces gets templated; when-beat with absent person but no target bypasses presence check
- Task 12: minor (deferred): Origin scheme not compared; no tests for Origin "null"/missing port/no-Origin; map labels small at 400 px; page flows not exercised in a browser
- Task 13: minor (deferred, carried into Task 14): per-person try in agent tick; lockstep treats silent agent heartbeat (~5 s) as pending 0; configurable ready timeout; CLI signals reception_bot; e2e ROS_DOMAIN_ID 1-101
- Task 13: minor (deferred): untracked create_task(save()) in director; agent cleanup gaps if open_shortterm/make_gateway/_write_meta raise; --intents replay doesn't record intents/says/world in the new run; de-dup key ignores point/addressee/style; exact replay timeline at sim-time 1 s
- Task 14: minor (deferred): grandchildren not signalled if group leader already reaped; copied scenario loses YAML comments; panel URL printed twice
- Task 16: minor (deferred): README uses internal "Plan 2/3" terms; Motion data section lists script fixes inline; cost/rates unmeasured (labelled)
- Final fix wave: re-review all addressed (9/9); deferred: breaker ignores 5xx/max_tokens streaks; long lines in agent_node; pre-restore name overwrite harmless

## Deferred from the final review
- Cost is priced on the requested model even when a fallback model served the reply (use msg.model and usage.iterations).
- The agent node's Counter diff of heard events can miss a repeated identical sound at the 2 s window edge.
- A rerun starts at run_index N+1, so long-term memory recency weights differ slightly from the original run.
