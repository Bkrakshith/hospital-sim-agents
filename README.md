# hospital-sim-agents

Simulated hospital patients and visitors who think for themselves. Each person is a
language-model "mind" with a persona, a memory and a body. They talk to a Unitree G1 robot
over ROS 2: they ask it the way to the clinic, cough, get lost and get impatient, and the
robot has to cope.

You use it to test a robot's reception behaviour against people who are not scripted.

This version runs the people on a lightweight stand-in simulator (the "stub sim"). Isaac Sim
comes later. The Python package is `wardzero_agents` and the command is `hhumans`.

What v1 covers: patients and visitors asking for directions in a waiting area. There are 12
starter cards and an 8-beat storyboard (both in `config/`). There are no crisis scenarios yet.

## Setup

```bash
source /path/to/Hospital_G1/env.sh        # Python 3.10 venv and ROS 2 Humble
cd hospital-sim-agents
pip install -e .                          # add ".[test]" to run the tests
scripts/install_hooks.sh                  # pre-commit secret scan, once per clone
hhumans --help
```

Run the unit tests with `python -m pytest -q -m "not ros"`.

## Accounts and API keys

The minds call a hosted model, so a run needs an API key.

- You need a Claude Console account or an OpenAI Platform account with pay-as-you-go billing.
- A Claude Pro or Max subscription cannot be used for API calls. It is a different product.
- Set a monthly spend limit in the console before you start.

Store the key in one of these ways, in this order of preference:

1. `ant auth login` (the Anthropic CLI).
2. `hhumans keys set claude` or `hhumans keys set openai`. The key goes in your OS keyring.
3. On a headless server with no keyring, put the key in a file with mode 600 and point
   `WARDZERO_KEY_FILE` at it.

Check with `hhumans keys status claude`. Remove a key with `hhumans keys clear claude`.
Before a run starts, the key is checked against each model the run uses: decide, summarise,
and vision when perception is `full`.

Only the agent service gets the key. The other processes (the stub sim, the director and the
robot) start without `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or
`WARDZERO_KEY_FILE` in their environment.

No key, no run. The one exception is `WARDZERO_TEST_FAKE_LLM=1`, which swaps in a fake model.
It is for tests only, prints `TEST MODE: no AI provider` and marks the run as a test.

Keys never go in ROS parameters, topics, logs, run folders or git. The pre-commit hook from
`scripts/install_hooks.sh` scans for secrets before each commit.

### Models

Claude (the default provider):

| Job | Model |
|---|---|
| decide (what to do next) | `claude-opus-5-5`, effort low |
| summarise (memory) | `claude-haiku-5-5` |
| vision | `claude-opus-5-5` |
| cheaper option | `claude-sonnet-5-5` |

OpenAI has no defaults. The scenario must name its models.

Requests use structured outputs and a 1-hour prompt cache. On Opus and Sonnet a refusal
falls back automatically.

## Running

```bash
hhumans run config/scenarios/waiting_area_v1.yaml
```

This starts the stub sim, the agent service, the director, the control panel and a
placeholder robot (`reception_bot`). Stop it with Ctrl-C. Output goes to a run folder under
`runs/` (override with `WARDZERO_RUNS_DIR`).

Useful flags:

- `--time-scale N` runs the sim N times faster than the wall clock.
- `--lockstep` holds the sim clock while any person is still deciding, so slow model calls
  do not change what happens. On the RTX 3070, slower than real time is fine.
- `--no-robot` runs the people alone. `--bot none` leaves the robot slot empty so you can
  start your own.
- `--no-panel` skips the control panel. `--panel-port PORT` moves it.
- `--ready-timeout SECONDS` is how long the sim waits for the other services. The first
  knowledge base build can be slow.
- `--sim isaac` is accepted but not available yet.

### Your own robot

The robot fairness rule: the robot sees the world only through its own senses.

- The robot hears people only on `/g1/audio/heard`.
- It speaks by publishing plain text on `/g1/speech/say`.
- Robot speech is 72 dB at 1 m. Text with no addressee goes to the nearest person within 2.5 m.
- In the stub the robot gets no lidar or camera. People's bodies in the sensors come in Plan 2
  with Isaac.

To test a real G1 stack, point its speech output at `/g1/speech/say` and its speech input at
`/g1/audio/heard`, then run with `--bot none`. `wardzero_agents/examples/reception_bot.py` is
a small placeholder that shows the pattern. It is not the real robot's behaviour.

The robot's microphone cannot tell who spoke, so on `/g1/audio/heard` the `source_id` is
always `unknown`. `source_kind` still says whether it was a person, the PA, a cough or an alarm.

The audit `hhumans audit` checks the live ROS graph once. It flags any node whose name does
not start with `wardzero_` that is subscribed to `/humans/`, `/scenario/`,
`/sim/audio/events`, `/sim/world_event` or `/agents/pending`. Use `--wait SECONDS` to allow
more discovery time.

Topic names, types and rates are in `docs/interfaces.md` ("Persona agents").

## Control panel

`hhumans run` prints the address: <http://127.0.0.1:8765/>.

The panel binds to 127.0.0.1 only. On a remote machine, tunnel it:

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

Then open the same address in your own browser. Pages from other sites are refused (the
panel checks the Origin header).

From the panel you can start, pause and stop the run, fire any of the cards on a chosen
person, switch chaos on or off and see the cast, the timeline, the cost against the budget,
and each person's mood and last line.

- **Cards** are scripted nudges to one person: `lost_visitor`, `coughing_fit`,
  `hard_of_hearing`, `name_called`, `impatient_patient`, `blocked_path`, `wet_floor`,
  `new_arrival`, `clinic_delay`, `ward_visitor`, `toilet_request`, `way_out`.
- **Storyboard** (`config/storyboards/waiting_area_day.yaml`): 8 beats that fire cards at set
  sim times, for example a clinic delay at 5 minutes and a new visitor at 8.
- **Chaos** fires random cards at a rate you set, within a cap.

## Memory

Each person has a long-term memory that carries across runs. A person who was lost last run
remembers it.

- Memory lives in `$WARDZERO_DATA_DIR` (default `~/.local/share/wardzero-agents/`).
- Reset people to their starting memories:
  `hhumans memory reset --person P2 --scenario config/scenarios/waiting_area_v1.yaml`,
  or `--persona NAME`, or `--all`. It asks first unless you pass `--yes`. A reset refuses to
  run while a run is active.
- Snapshots: `hhumans memory snapshot save|restore|list [NAME]`.
- Every run saves `run-<id>-start` and `run-<id>-end` snapshots on its own.
- A restore replaces the whole memory db, for everyone. So before any restore (a `rerun`, a
  scenario with `memory_snapshot`, or `hhumans memory snapshot restore`) the current memory is
  saved first, as `pre-restore-<run id>` (or `pre-restore-manual-<time>` for a restore by
  hand). The name is printed. Restore it to get back what the later runs had learned.
- Names starting with `run-` or `pre-restore-` are reserved for these automatic snapshots.
  `snapshot save` and the panel refuse them.

## Knowledge base

The people draw on a small knowledge base built from `research_notes/` and the config.

- The default embedder is a local hash embedder. It needs nothing and is repeatable, but
  its retrieval is weak.
- `pip install -e .[embed]` adds fastembed. It downloads a model on first use and retrieves
  better. Use it with `hhumans kb build --embedder fastembed` and set
  `WARDZERO_EMBEDDER=fastembed` so runs match.

`hhumans kb build` rebuilds the index. A run rebuilds it on its own when it is missing or
out of date.

## Re-running and replay

Three ways to go again, from a recorded run folder:

| Command | What it does | Cost |
|---|---|---|
| `hhumans replay RUN --exact` | Republishes the recorded world. Read-only panel. | free |
| `hhumans replay RUN --intents` | Re-sends the recorded intents to a live stub sim and robot, so a new robot meets the same behaviour. | free |
| `hhumans rerun RUN` | Runs the same scenario again from the run's start memory. The minds decide again. | full |

Exact physical repeat holds only on the same machine and build.

`rerun` restores the run's start snapshot over the long-term memory, so memories made by any
later run would be lost. It saves them first as `pre-restore-<new run id>` and prints that
name; `hhumans memory snapshot restore pre-restore-<new run id>` brings them back.

`replay --intents` records into a new run folder: the sim's streams, and the intents, speech
and world events it re-sent. That folder can itself be replayed.

## Perception modes

- **Oracle (now).** People get exact facts from the simulator: who is near, what is said,
  what signs they can see.
- **Full (Plan 3).** People see through head cameras and a vision model, hear through local
  speech recognition and speak through text to speech. It will need: head camera streams, a
  vision model service, a local ASR service, a TTS service and an audio route into the sim.

## Motion data

v1 uses placeholder clips (`hospital-sim-humans/motion/clips_v1.yaml`).

Real clips come from BONES-SEED, a 45.2 GB archive:

1. Free the disk space.
2. Accept the licence on Hugging Face.
3. Run `hospital-sim-humans/scripts/fetch_bones_seed.sh --dry-run` first, then without it.

Fix these before the first real download (known issues in the script):

- A converter that exits 0 with silent skips still deletes the BVH.
- `BATCH=500` is unused.
- A subshell killed for memory leaves no DONE marker and the loop hangs.
- The strict M==N check fails on stale outputs.
- The generic `cough` and `walk_slow limp` patterns are not anchored.
- `HF_TOKEN` is inherited by child processes.
- `failed_batches` nests on a rerun.

Licence: BONES-SEED and clips made from it are not redistributed. Never commit them.

## Costs

These are estimates, to be measured.

| Run (1 hour, oracle mode) | Estimate |
|---|---|
| Opus 5.5 | about $10-15 |
| Sonnet 5.5 | about $5-8 |
| Haiku summaries | under $0.50 |
| Replays | free |

`budget_usd` in the scenario caps spend. When it is reached, the bodies carry on with their
reflexes and the minds stop deciding. The panel shows the cost against the budget.

## Environment variables

| Variable | Meaning |
|---|---|
| `WARDZERO_DATA_DIR` | memory, snapshots and the knowledge base index |
| `WARDZERO_RUNS_DIR` | where run folders go |
| `WARDZERO_KEY_FILE` | key file for headless servers (mode 600) |
| `WARDZERO_EMBEDDER` | `hash` (default) or `fastembed` |
| `WARDZERO_TEST_FAKE_LLM` | `1` for tests only |

## Troubleshooting

If the AI provider rejects requests (a bad key, a model the account cannot use, a request that
is too large), the agent service logs a warning with the HTTP status the first time each kind
happens and shows it on the panel. After 5 rejected requests in a row it stops calling the AI
for the rest of the run: the panel shows "AI provider rejecting requests: <kind> (status N)".
The people carry on with their reflexes and what they were already doing. Check the key with
`hhumans keys status`, check the model names in the scenario, and start a new run. The kinds
are `auth` (401/403: the key), `model_not_found` (404: the model name or account access),
`bad_request` (400) and `too_large` (413). Network errors and 5xx are not counted.

ROS 2 installs pytest plugins that clash with pytest 8. `pyproject.toml` switches them off.
If you still see plugin errors, run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -m "not ros"`.

## Licence

Apache License, Version 2.0 (see `LICENSE`). This is part of WardZero
([hospital-sim](https://github.com/Bkrakshith/hospital-sim)). WardZero is a simulation. Nothing here has run
with a physical robot or in a real hospital.
