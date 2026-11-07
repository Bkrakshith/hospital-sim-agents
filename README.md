# WardZero Studio

Test your robot against simulated hospital patients, visitors and staff before a real trial.

Part of [WardZero](https://bkrakshith.github.io/wardzero/), a simulated hospital for testing robots before they meet patients. Start from the main repository, [hospital-sim](https://github.com/Bkrakshith/hospital-sim).

Each person is run by an AI model (Claude or OpenAI). They have a role, a mood and a memory. They ask the robot
for help, get impatient when it does not help, and tell each other how it went. After a run, a report scores the
robot against the scenario's success criteria: problems solved, near misses, contacts, forbidden areas,
unanswered requests, response time and mood changes.

Bring your own robot (its URDF, its sensors and a ROS 2 controller), or start with the built-in Unitree G1. Run a
scenario in a quick 2D simulator, or in the 3D hospital in Isaac Sim 6.0.

### New in v3.1

- **The G1 walks in Isaac Sim.** NVIDIA's GEAR-SONIC whole-body controller drives the built-in G1. It walks people
  to where they need to go and comes back to its post. Quick 2D runs need no controller.
- **Doors swing by physics.** People and the robot push the hinged doors open; a door closer shuts them again.
- **A receptionist at the desk.** Reception scenes add a live receptionist, seated behind the desk. The robot can
  walk over and tell her about a person who needs staff (`tell_staff`). She can be removed from the cast.
- **Placement in the Studio.** Drag people and the robot on the map, turn them, and tick **Start seated** to put a
  person in a waiting-area seat at the start.
- **Forbidden areas.** Draw rectangles the robot must keep out of. The robot's route avoids them, and the report
  counts any entry.
- **The camera follows the robot** in every Isaac scenario.

The Python package is `wardzero_agents` and the command is `hhumans`. This repository is part of
[WardZero](https://github.com/Bkrakshith/hospital-sim) (project page: <https://bkrakshith.github.io/wardzero/>).

## Quick start

### What you need

- Ubuntu 22.04 (x86_64) with ROS 2 Humble and Python 3.10.
- For real runs, an API account with pay-as-you-go billing: a Claude Console account (the default) or an OpenAI
  Platform account. A Claude Pro or Max subscription does not work for API calls. You can try the Studio for free
  first, without a key (step 3).
- For 3D runs only: Isaac Sim 6.0 and an NVIDIA RTX GPU with 16 GB of graphics memory (RTX 4080 or better)
  (step 6). Quick 2D runs need no GPU.

### 1. Install

```bash
sudo apt install ros-humble-desktop ros-humble-vision-msgs ros-humble-rmw-cyclonedds-cpp python3-venv git
git clone --recurse-submodules -b v3.1 https://github.com/Bkrakshith/hospital-sim.git WardZero
cd WardZero
python3 -m venv --system-site-packages .venv   # system site-packages make ROS 2's rclpy visible
source env.sh                                  # ROS 2 and the venv; do this in every new terminal
pip install -e "hospital-sim-agents[voice]"
python hospital-sim-agents/scripts/fetch_voices.py   # optional: spoken voices in recorded runs
```

`env.sh` also removes conda libraries from the library path, because they break ROS 2 and Isaac Sim.

### 2. Check the computer

```bash
hhumans doctor
```

Each line says `ok`, `note` or `FIX`. Each `FIX` comes with numbered steps that fix it.

### 3. Try the Studio for free

```bash
WARDZERO_TEST_FAKE_LLM=1 hhumans studio
```

This opens the Studio at <http://127.0.0.1:8770/> with a fake model in place of the AI. The people follow simple
fixed behaviour, nothing is billed, and each run is marked as a test. Use it to learn the four pages.

### 4. Add your key and run the demo scenario

```bash
hhumans keys set claude        # stored in your OS keyring; see "Accounts and API keys"
hhumans studio
```

In the Studio:

1. **Scene.** Choose **Reception and guide**. The people, their roles and moods, and their start positions are on
   the map. Drag them to move them, tick **Start seated** for anyone who should start in a waiting-area seat, and
   use **Draw area** to mark places the robot must keep out of.
2. **Robot.** Choose **Built-in G1 (Claude)**. Its camera, microphone and speaker are already set up.
3. **Run.** Choose **Quick 2D**, a duration of 1 minute and a **Spending limit** of $5. Leave **Word of mouth**
   on. Press Start. The live cost shows during the run; a one-minute run with the defaults costs well under $1.
4. **Report.** Read pass or fail for each success criterion, what each person said and asked, how their mood
   changed, and the suggestions for the robot.

Then set **Repeats** to 2 or more and tick **Carry memory between repeats**. The people now remember the earlier
run, so a robot that let someone down starts the next run with less trust.

The built-in G1 is an example robot. It can see, hear and talk. In Quick 2D it moves along its route directly, and
in Isaac Sim it walks with NVIDIA's GEAR-SONIC controller (step 6). It can walk a person to a room, or walk over
and tell the receptionist. The Studio is there to test your robot, so replace it with yours when you are ready.

### 5. Test your own robot

1. On the **Robot** page, make a new robot and give the folder with its URDF. Press **Check the URDF**.
2. Add or edit its sensors: camera, lidar, microphone and speaker. Each sensor decides which topics the robot gets.
3. Choose the ROS 2 domain and namespace. The page lists the topics your controller publishes and subscribes to,
   with three lights that turn on as it connects. The full contract is in `docs/robot-topics.md`, and
   `wardzero_agents/examples/reception_bot.py` is a small controller to start from.
4. Place the robot on the map, then run the same scenario in Quick 2D. When it behaves there, run it in Isaac Sim.

The built-in robot (`--bot claude`) uses the same contract, so it is a worked example. Details are in
"Your own robot" below.

### 6. Optional: Isaac Sim

1. Install Isaac Sim 6.0 in `~/isaacsim`, or set `ISAAC_SIM_ROOT` to where it is.
2. Install the NVIDIA asset packs and connect them to the hospital scene. The
   [hospital-sim-env README](https://github.com/Bkrakshith/hospital-sim-env) gives the steps. On a GPU with less
   than 12 GB of graphics memory, use the reduced asset set (`python3 hospital-sim-env/isaac/scene/select_assets.py lite`).
3. Build the collision file once. It keeps the people and the robot inside the walls:

   ```bash
   pip install usd-core numpy
   python hospital-sim-agents/scripts/build_collision_usd.py   # writes hospital-sim-agents/assets/hospital_collision.usd
   ```

4. Download the walking controller for the built-in G1 (about 1 GB, NVIDIA Open Model License; the weights are
   kept outside this repository):

   ```bash
   python hospital-sim-agents/scripts/fetch_sonic.py
   ```

   Without it the G1 stands still in Isaac Sim. Your own robot does not need it.
5. Run `hhumans doctor` again, then choose **Isaac Sim** on the Run page. A window or headless with a live preview
   are both available.

Isaac Sim starts slowly the first time, because it compiles shaders and loads the hospital. Wait for it. Recorded
runs with a walking robot are slower than real time (physics runs at 200 Hz for the robot's balance); the
recorded video plays back at normal speed. Isaac Sim keeps a large cache of derived data (tens of GB); to keep it on another disk, set
`WARDZERO_KIT_ARGS="--/UJITSO/datastore/localCachePath=<folder>"` before you start the Studio.

### For developers

```bash
pip install -e "hospital-sim-agents[test]"
cd hospital-sim-agents && scripts/install_hooks.sh   # pre-commit secret scan, once per clone
python -m pytest -q -m "not ros"
```

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
| decide (what to do next) | `claude-sonnet-5-5`, effort low |
| summarise (memory) | `claude-haiku-5-5` |
| vision | `claude-sonnet-5-5` |
| dearer option | `claude-opus-5-5` (set it in the scenario `models:`) |

OpenAI has no defaults. The scenario must name its models.

Requests use structured outputs and prompt caching: the instructions and persona card are cached for 1 hour, and a person's day so far is cached as a second, 5-minute breakpoint ahead of the per-tick text. On Opus and Sonnet a refusal
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
  do not change what happens. The run is then slower than real time, which is fine.
- `--no-robot` runs the people alone. `--bot none` leaves the robot slot empty so you can
  start your own.
- `--no-panel` skips the control panel. `--panel-port PORT` moves it.
- `--ready-timeout SECONDS` is how long the sim waits for the other services. The first
  knowledge base build can be slow.
- `--sim isaac|stub` picks the simulator (see "Isaac Sim mode").

## Studio

The Studio is a browser tool for setting up a test without editing YAML. Start it with:

```bash
hhumans studio            # --port 8770 by default; --no-browser to skip opening a tab
```

It has four pages, in order:

1. **Scene** pick a scenario, place and edit the people (role, mood, start position),
   and see them on the hospital map.
2. **Robot** bring your own robot: its URDF, sensors and ROS 2 connection (below).
3. **Run** start the run, watch it live with the cost so far, and stop it.
4. **Report** read the finished run: pass or fail against the scenario's criteria, what the
   people said and rated, and suggestions.

The scenarios you can start from:

- Reception and guide: people ask the way at a hospital entrance.
- A&E waiting room: long waits, and a patient who gets worse.
- Meal round: bring the right tray to the right bed.
- Specimen run: carry a sample from the ward to the lab.
- Call bell at night: a patient presses the bell and nobody is near.
- Visiting-hours rush: a crowded ward entrance.
- Wandering patient: someone leaves the ward alone.
- Ward fire: a small fire in the ward corridor.
- Fire evacuation guide: lead people out safely.
- and more to come.

Each scenario has success criteria (for example "fire out within 2 minutes" or "no contact with
people"). A run is scored against them and the report says which passed. A criterion that a run gives nothing to
measure shows as n/a, for example a response time when nobody asked the robot anything, or a safety criterion when the
robot never moved: it neither passes nor fails, and it is left out of the score.

### Your own robot

On the Studio's Robot page you give the Studio a folder with your robot's URDF, then:

1. **Check the URDF.** The Studio reads the links and joints and finds the base and the head.
2. **Sensors.** Add or edit the camera, lidar, microphone and speaker. Each sensor decides
   which topics the robot gets: no microphone, no `audio/heard`.
3. **Connect over ROS 2.** Choose the ROS domain your robot runs on (`--robot-domain`) and its
   namespace. The page shows three lights (robot controller seen, sensors subscribed, commands seen) and
   the list of topics your robot should publish and subscribe to.
4. **Place it on the map.** Pick where the robot starts and which way it faces.

The robot's side of the contract is in `docs/robot-topics.md`: what it hears, sees and
drives, and what it publishes (`speech/say`, `goal_pose`, `cmd_vel` and task actions). The
robot lives on its own ROS domain. The sim publishes only the robot topics and `/clock` there,
so the robot cannot read the people's minds or the scenario. To see how a result comes back
for each task action and each walk goal (`arrived`, `stuck`, `no_route`, `superseded`,
`cancelled`), read that file. The built-in Claude robot (`--bot claude`) follows the same
contract, so it is a worked example.

#### Connecting by hand

The robot fairness rule: the robot sees the world only through its own senses.

- The robot hears people only on `/g1/audio/heard`.
- It speaks by publishing plain text on `/g1/speech/say`.
- Robot speech is 72 dB at 1 m. It is addressed to the person whose speech the robot last heard (within 20 s, once); otherwise to the nearest person within 2.5 m. The robot itself never sees who spoke.
- In the stub the robot gets anonymous person boxes on `camera/bbox_3d` and the objects in view
  on `perception/objects`, but no camera images or lidar. Those come with Isaac.

To test a real G1 stack, point its speech output at `/g1/speech/say` and its speech input at
`/g1/audio/heard`, then run with `--bot none`. `wardzero_agents/examples/reception_bot.py` is
a small placeholder that shows the pattern. It is not the real robot's behaviour.

The robot's microphone cannot tell who spoke, so on `/g1/audio/heard` the `source_id` is
always `unknown`. `source_kind` still says whether it was a person, the PA, a cough or an alarm.

The audit `hhumans audit` checks the live ROS graph once. It flags any node whose name does
not start with `wardzero_` that is subscribed to `/humans/`, `/scenario/`,
`/sim/audio/events`, `/sim/world_event` or `/agents/pending`. Use `--wait SECONDS` to allow
more discovery time.

Topic names, types and rates are in `docs/interfaces.md` ("Persona agents"). The robot's
own topics are in `docs/robot-topics.md`.

### Isaac Sim mode

`hhumans run SCENARIO --sim isaac` runs the same people and robot contract in Isaac Sim 6.0, with
`--headed` to watch in a window or `--headless` (the default, with a live preview in the Studio) and
`--record` to save the videos and a rosbag. `--sim stub` is the fast 2D stand-in without rendering.
In both, only the robot's topics and `/clock` are on the robot's domain (`--robot-domain N`, shown on the
Studio's Robot page); the people are on a private domain the robot never joins. With a window, the view starts at
the robot's post, in front of it and below the ceiling, so the robot and the people who come to it are in view.

**What you need** (setup steps in "Quick start", step 6).

- Isaac Sim 6.0 installed, with its `python.sh` (set `ISAAC_SIM_ROOT` if it is not in `~/isaacsim`) and an NVIDIA RTX
  GPU. The hospital with three people and the robot uses about 6.5 GB of graphics memory with the reduced assets
  and a window; headless uses less.
- ROS 2 Humble sourced for the lab's own nodes (`env.sh`; set `WARDZERO_ROS_ENV_SH` to your own file). Without
  it the run goes ahead with no rosbag, and the run log says so.
- Two USD files, which are not part of this repository:
  - the hospital stage, from the `hospital-sim-env` checkout (`WARDZERO_STAGE_USD` overrides the path);
  - the collision USD that keeps people and robots inside the walls. Make it once with
    `python scripts/build_collision_usd.py` (needs `pip install usd-core numpy`). It writes
    `assets/hospital_collision.usd`; `WARDZERO_COLLISION_USD` overrides the path.

  If either file is missing, `hhumans run --sim isaac` stops with exit code 2 and says which one, and the
  Studio's Isaac option says what is missing instead of starting a run that would fail.

**Voices.** Recorded runs can have spoken voices (a separate Piper process; the voice models are downloaded,
not shipped):

```bash
pip install '.[voice]'
python scripts/fetch_voices.py
hhumans voice RUN_FOLDER      # add or redo the voices of an existing recording
```

The voice step runs after the simulator has exited and never fails a run; `--no-voice` skips it.

**From the Studio.** The Run page's Simulator group chooses Isaac Sim or Quick 2D, a window or headless with a
preview, recording and its video size, and voices. Settings are saved per project. The robot domain is on the
Robot page.

**Spending limit.** The limit on the Run page is for the whole Start: the people, the robot, and the debrief and
report, over every repeat and the baseline run. Each run is given what remains divided by the runs left, and a
repeat batch stops, with its report, when the limit is reached.

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

In Isaac Sim the people are NVIDIA's animated characters from the Isaac Sim asset pack, so there is nothing extra to
download. Motion-capture clips (BONES-SEED) are not part of this repository and are not needed for a run.

## Costs

These are estimates, to be measured.

| Run (1 hour, oracle mode) | Estimate |
|---|---|
| Sonnet 5.5 (the default) | about $5-8 |
| Opus 5.5 | about $10-15 |
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

**Before a run: `hhumans doctor`.** It checks ROS 2, the Claude key (stored, not spent), Isaac Sim and its two USD
files, another Isaac Sim already on the GPU, free graphics memory, disk space, the voices and your saved robots. Each
problem comes with the steps that fix it, and it exits non-zero when something must be fixed.

**When a run fails.** A run is several processes. When one fails, the others are stopped, and their stop messages (a
`KeyboardInterrupt` traceback, "There was an error running python") come after the line that matters. The Studio's
Run page reads the whole run log (and Isaac Sim's own log when Isaac stopped) and shows each problem it recognises,
most likely cause first, with numbered steps to fix it. From a terminal, `hhumans diagnose` does the same for the
Studio's latest run, or `hhumans diagnose FILE` for a saved log (`hhumans run ... 2>&1 | tee run.log`). It knows:

- a robot with no URDF, a URDF that does not convert, robot settings that are not valid;
- a scenario mistake, a missing hospital stage or collision USD, Isaac Sim not installed;
- no key, a rejected key, no credit left, a model the account cannot use, rate limits, provider outages, the keyring,
  an old Brotli;
- conda libraries on the path, ROS 2 not sourced, a missing Python package;
- Isaac Sim crashes (out of graphics memory, the fire effect, no GPU or driver, no display for a window), a process
  killed for RAM, a port in use, another run or another Isaac Sim already going;
- the people or director not coming up, a missing memory snapshot, and the after-run notes (no voices, no report).

Some of these are refused before anything starts: an Isaac run with a robot that has no URDF, or while another Isaac
Sim is running, stops at Start with the fix.

**Fire scenarios and graphics memory.** The realistic fire and smoke (NVIDIA Flow) needs about 1.5 GB of graphics
memory when the fire starts. With less than 2 GB free, the fire is drawn as a simple glowing stand-in instead, and the run log
says so (without this, Isaac Sim crashed). Headless runs leave more room for the full fire. `WARDZERO_FIRE=proxy`
always uses the stand-in; `WARDZERO_FIRE=none` draws no fire.

If the AI provider rejects requests (a bad key, a model the account cannot use, a request that
is too large), the agent service logs a warning with the HTTP status the first time each kind
happens and shows it on the panel. After 5 rejected requests in a row it stops calling the AI
for the rest of the run: the panel shows "AI provider rejecting requests: <kind> (status N)".
The people carry on with their reflexes and what they were already doing. Check the key with
`hhumans keys status`, check the model names in the scenario, and start a new run. The kinds
are `auth` (401/403: the key), `model_not_found` (404: the model name or account access),
`bad_request` (400) and `too_large` (413). Network errors and 5xx are not counted.

If the key check says "failed (APIConnectionError)" although the network works, check
`python -c "import brotli; print(brotli.__version__)"`. Ubuntu 22.04 ships Brotli 1.0.9, which the SDK HTTP
client cannot use. Run `pip install "Brotli>=1.2"` in the environment.

ROS 2 installs pytest plugins that clash with pytest 8. `pyproject.toml` switches them off.
If you still see plugin errors, run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -m "not ros"`.

## Licence

Apache License, Version 2.0 (see `LICENSE`). This is part of WardZero
([hospital-sim](https://github.com/Bkrakshith/hospital-sim)). WardZero is a simulation. Nothing here has run
with a physical robot or in a real hospital.
