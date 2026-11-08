# nemo_gym_furrow — Furrow as an NVIDIA NeMo Gym environment

LLM agents play **segments** of a Furrow speedrun through four tools. A task replays the scripted
expert on seed `S` up to day `D`. From there the model has `horizon_days` in-game days (default 7) and
`max_tool_calls` tool calls (default 60) to complete **any one new restoration request**. Faster is
better, measured against the expert's time from the same state. The design is in DESIGN.md §9; the
facts verified about NeMo Gym 0.7.0 are in [`docs/NEMO_GYM.md`](../../docs/NEMO_GYM.md).

| Path | What |
|---|---|
| `core.py` | Start states, tools, lexicographic reward, thread-safe `SessionStore` (does not import NeMo Gym) |
| `resources_servers/furrow/app.py` | The NeMo Gym resources server (`SimpleResourcesServer`): `/observe` `/list_actions` `/act` `/status` + `/seed_session` `/verify` `/close_session` |
| `resources_servers/furrow/configs/furrow.yaml` | Resources server + `simple_agent` (`furrow_simple_agent`) + `legacy_agent` environment server |
| `resources_servers/furrow/data/example.jsonl` | 5 example tasks |
| `configs/policy_model.yaml` | Model server `policy_model` (Gym's `openai_model` adapter) |
| `configs/env.fake.yaml` | `env.yaml` template for the fake model |
| `configs/nemo_rl_grpo_furrow.yaml` | NeMo RL GRPO recipe (**UNVERIFIED**) |
| `prompt.py` | System prompt (about 650 tokens), tool schemas, dataset-row builder |
| `tasks.py` | CLI that writes train/val/test JSONL files |
| `fake_openai.py` | Offline fake OpenAI-compatible model (`/v1/responses`, `/v1/chat/completions`, `/v1/models`) |
| `launch.py` | Points Gym's per-server venvs at one prepared venv, so runs work offline |
| `tests/` | `pytest` suite, including a real-harness end-to-end test |

## Setup (VERIFIED)

NeMo Gym 0.7.0 requires Python ≥ 3.13.14. On Python 3.14 the `gym` CLI crashes in hydra's argparse
(`ValueError: badly formed help string`), so use a 3.13 build. uv 0.11.11 offers CPython only up to
3.13.13; uv 0.13.7 has 3.13.18.

```bash
cd /path/to/Stardew-Valley-RL-Agent            # $REPO below
uvx uv@0.13.7 python install 3.13.18
# This also repoints uv's `cpython-3.13-linux-x86_64-gnu` link, which `.venv` uses, to 3.13.18.
# To keep `.venv` on 3.13.13: ln -sfn ~/.local/share/uv/python/cpython-3.13.13-linux-x86_64-gnu ~/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu
uv venv --python ~/.local/share/uv/python/cpython-3.13.18-linux-x86_64-gnu/bin/python3.13 .venv-nemo
uv pip install --python .venv-nemo/bin/python nemo-gym==0.7.0 pytest
. "$HOME/.cargo/env"
.venv/bin/maturin build --release -m crates/furrow-py/Cargo.toml -o /tmp/furrow-wheel -i .venv-nemo/bin/python
uv pip install --python .venv-nemo/bin/python /tmp/furrow-wheel/furrow-*.whl
```

Tests take about 35 s. They need no network and no GPU, and include two end-to-end runs through the
real Gym harness:

```bash
.venv-nemo/bin/python -m pytest -q python/nemo_gym_furrow
```

## 1. Generate tasks (VERIFIED)

```bash
PYTHONPATH=python .venv-nemo/bin/python -m nemo_gym_furrow.tasks \
    --out data/nemo_gym_furrow --train 2000 --val 100 --test 200 \
    --days-per-seed 2 --min-expert-days 0.25 --workers 8
```

- Seeds come from disjoint ranges: train `[0, 2^32)`, val `[2^32, 2^33)`, test `[2^33, 2^34)`.
- For each seed the expert plays the run once. Start days are sampled from that trajectory, but only
  days from which the expert completes its next request within the horizon.
- `--min-expert-days 0.25` drops near-trivial starts, such as "Deliver what you hold".
- Each row is `responses_create_params` (system prompt, user message with the start state, tools)
  plus `task_id`, `seed`, `start_day`, `horizon_days`, `max_tool_calls`, `expert_days` and `agent_ref`.
- `manifest.json` records counts, seed ranges, `SIM_VERSION` and `content_hash`.
- Other flags: `--horizon-days`, `--max-tool-calls`, `--env-config '{"restoration": "remixed"}'`.

## 2. Start the servers (VERIFIED, offline)

Gym starts each server from `<uv_venv_dir>/<type>/<name>/.venv`. `launch venvs` links all of those to
`.venv-nemo`, and `+skip_venv_if_present=true` makes Gym reuse them, so it never runs
`uv pip install`. Gym writes `cache/` and `results/` to the current directory, and Ray writes logs to
`$RAY_TMPDIR` (default `/tmp/ray`).

```bash
REPO=$PWD; WORK=/tmp/furrow-gym; mkdir -p $WORK && cd $WORK
PYTHONPATH=$REPO/python $REPO/.venv-nemo/bin/python -m nemo_gym_furrow.launch venvs --root $WORK/venvs
#  -> +skip_venv_if_present=true +uv_venv_dir=/tmp/furrow-gym/venvs

# terminal 1: the fake model (picks a random valid action from the list_actions output; seeded by the
# conversation, so repeats of one task are identical)
PYTHONPATH=$REPO/python $REPO/.venv-nemo/bin/python -m nemo_gym_furrow.fake_openai --port 8765

# terminal 2: resources server, agent, environment server, model server
$REPO/.venv-nemo/bin/gym env start --search-dir $REPO/python/nemo_gym_furrow \
    --config resources_servers/furrow/configs/furrow.yaml --config configs/policy_model.yaml \
    --model-url http://127.0.0.1:8765/v1 --model-api-key dummy --model furrow-fake-random \
    +skip_venv_if_present=true +uv_venv_dir=$WORK/venvs
# wait for "All 4 / 4 servers ready!"; stop with Ctrl-C (Gym also stops the Ray cluster it started)
```

## 3. Collect rollouts (VERIFIED with the fake model)

```bash
cd $WORK
$REPO/.venv-nemo/bin/gym eval run --no-serve --agent furrow_simple_agent \
    --input $REPO/python/nemo_gym_furrow/resources_servers/furrow/data/example.jsonl \
    --output results/furrow_rollouts.jsonl --num-repeats 2
```

This writes three files:

- `results/furrow_rollouts.jsonl`: one row per rollout, with the full Responses-API trajectory, `reward` and the metrics below.
- `results/furrow_rollouts_aggregate_metrics.json`: `mean/*` of every metric, plus `success_rate`, `mean_speed_ratio_on_success` and `beats_expert_rate`.
- `results/furrow_rollouts_materialized_inputs.jsonl`

**Any OpenAI-compatible endpoint.** Restart step 2 with a different model URL, key and name.

- **Responses API.** `configs/policy_model.yaml` (Gym's `openai_model`) calls `POST {url}/responses`.
  For example `--model-url https://api.openai.com/v1 --model-api-key $OPENAI_API_KEY --model gpt-4.1-mini`.
  Status: VERIFIED against the fake only, UNVERIFIED against OpenAI.
- **Chat completions.** For an endpoint that serves only `POST /v1/chat/completions`, replace
  `--config configs/policy_model.yaml` with `--model-type vllm_model`. Gym converts between the two
  formats. Status: VERIFIED against the fake.
- **vLLM.** For example `vllm serve Qwen/Qwen3-4B-Instruct-2507 --enable-auto-tool-choice --tool-call-parser hermes`
  with `--model-type vllm_model --model-url http://127.0.0.1:8000/v1`. Status: UNVERIFIED (no GPU was used).
- **Server settings.** You can also set `policy_base_url`, `policy_api_key` and `policy_model_name` in
  `$WORK/env.yaml`; `configs/env.fake.yaml` is a template.
- **Agent step limit.** `furrow_simple_agent.max_steps` (70) caps model turns. Keep it ≥ the tasks'
  `max_tool_calls`.

## Tools, episode end and reward

| Tool | Arguments | Returns |
|---|---|---|
| `observe` | none | `describe()` text (date, weather, location, energy, money, bag, crops, machines, open requests with items held) + `progress` |
| `list_actions` | none | valid macros, grouped: `GoTo(woods\|coast)`, `Buy(pea_seed,1\|5; corn_seed,1)`; `count` |
| `act` | `actions`: list of names (a bare string is accepted) | `executed` (outcome and clock per macro), `errors` (unknown or currently invalid name, `did_you_mean` via difflib), `not_run`, compact `state`, `progress` |
| `status` | none | compact `state` + `progress` (`now`, `day_ends_left`, `tool_calls_left`, `invalid_actions`, `done`, `end_reason`) |

How `act` and the budgets work:

- A day plan runs in order. It stops at the first unknown or currently invalid name, which is not
  executed and is counted in `invalid_actions`. It also stops when the day ends or the segment ends.
- Every tool call counts against `max_tool_calls`. A plan holds at most 40 macros.

The segment ends at the first of:

- `success`: a new request is completed.
- `horizon`: the day counter reaches the start day + `horizon_days`.
- `tool_budget`: the tool-call budget is exhausted.
- `env_finished`: Furrow's own caps end the run.

**Reward** (`core.segment_reward`, in [0, 1), lexicographic):

- **Success:** `0.5 + 0.5·(E+c)/((E+c)+(A+c))`. Here A is the agent's in-game days to the request
  (fractional, from `abs_minute`). E is the expert's days from the same state, computed when the
  session is seeded by rolling the expert forward on a `clone()`; the horizon is used if the expert
  fails. c = 0.25. The score is strictly decreasing in A, is 0.75 when A = E, and is always > 0.5.
- **Failure:** `0.45·progress`. `progress` is the largest gain on any still-open request: delivered
  slots count 1, held items toward an open slot count up to ½, and coins toward a payment count up
  to ½. So every failure scores below every success.

Each verify response also carries these metrics:

- `success`, `end_reason`, `days_to_success`, `expert_days`, `expert_success`, `speed_ratio` (E/A)
- `progress`, `days_played`, `requests_gained`, `slots_gained`
- `tool_calls`, `invalid_actions`, `macros_executed`

If verify finds no session, it returns `reward 0`, `mask_sample: true` and `failure_kind: session_lost`.

**Sessions.** Each rollout's state is keyed by Gym's signed session cookie
(`request.session["session_id"]`), as in Gym's `example_session_state_mgmt`. The agent's
`/seed_session` call builds one `furrow.Env`. The tools and `/verify` reuse it, and `/verify` scores
it and drops it; a repeated verify returns the cached result.

- The typed Environment-Server protocol (`ResourcesSeedSessionRequest` with `resources_session_id`,
  `/close_session`) is supported and passes Gym's `check_resources_session_contract`.
- The store is thread-safe: a lock per session and one for the map. Furrow work runs in a threadpool.
- Idle sessions expire after `session_ttl_seconds`, and `max_sessions` caps the count (LRU).

## GRPO with NeMo RL (UNVERIFIED)

Nothing here was run: no GPU and no NeMo RL install were used. The recipe follows NeMo RL main @
61bc275 (2026-11-08): `examples/nemo_gym/run_grpo_nemo_gym.py`, and a config that inherits
`grpo_workplace_assistant_nemotron_nano_v2_9b.yaml` the same way
`gdpo_example_tool_call_multireward.yaml` does.

1. Generate tasks (step 1). The rows already carry `agent_ref: furrow_simple_agent`; NeMo RL's
   "flat rows" route through the `furrow_environment_server` relay.
2. Build a venv with the Gym version that NeMo RL pins (submodule `3rdparty/Gym-workspace/Gym`), plus
   the furrow wheel. Link it with
   `python -m nemo_gym_furrow.launch venvs --root /ABS/venvs --venv /ABS/that-venv`.
   This is needed because `furrow` is not on PyPI, so Gym cannot build the server venv from
   `requirements.txt`.
3. Copy `configs/nemo_rl_grpo_furrow.yaml` to `<NeMo-RL>/examples/nemo_gym/grpo_furrow.yaml` and
   replace `/ABS/...`. It sets `config_paths` to `vllm_model_for_training.yaml` + our `furrow.yaml`,
   nulls the inherited workplace relay, and sets `skip_venv_if_present`/`uv_venv_dir`,
   `max_total_sequence_length: 32768`, and 16 prompts × 8 generations.
4. Run from `<NeMo-RL>`:
   `NEMO_GYM_EXTRA_ROOTS=/ABS/REPO/python/nemo_gym_furrow uv run examples/nemo_gym/run_grpo_nemo_gym.py --config examples/nemo_gym/grpo_furrow.yaml`

Still to check:

- Whether NeMo RL's pinned Gym honours `NEMO_GYM_EXTRA_ROOTS`, absolute `config_paths` and
  `skip_venv_if_present` in the same way as 0.7.0.
- Whether the inherited `cluster` (8 GPUs) and the model fit your hardware.
- Context length. With the fake policy, a 60-call segment measured about 12–13k tokens, before the
  model's own text.
