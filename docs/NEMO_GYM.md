# NeMo Gym integration (Furrow) — verified notes, design, tests

The code is in `python/nemo_gym_furrow/`; usage is in its `README.md`. This page records:

- §1: what was verified about **NVIDIA NeMo Gym 0.7.0** (PyPI `nemo-gym==0.7.0`, which matches
  upstream tag `v0.7.0`, commit `1f8afc71d45fac14872d86fe300374afa2001e7a`)
- §2: the design built on it
- §3: what was tested
- §4: what remains unverified

File references are relative to the installed package root
(`.venv-nemo/lib/python3.13/site-packages/`). The `nemo_gym/` sources there were diffed against the
v0.7.0 checkout and were identical. Built-in servers ship in the same wheel: `resources_servers/`,
`responses_api_agents/`, `responses_api_models/`, `environment_servers/`.

## 1. Verified facts about NeMo Gym 0.7.0

**Install / Python.**

- `requires_python >= 3.13.14` (PyPI metadata).
- On CPython 3.14.4 the `gym` CLI fails at startup. hydra 1.3.x adds `--shell-completion` with a
  `LazyCompletionHelp` help object, and 3.14's argparse rejects it (`ValueError: badly formed help
  string`, `hydra/_internal/utils.py`). On 3.13.18 it works.
- uv 0.11.11's download list stops at 3.13.13; `uvx uv@0.13.7 python install 3.13.18` works.
- Installing the wheel pulls in `ray[default]`, mlflow, fastapi, openai==2.44.0 and more. No GPU is
  needed for any of it.

**Resources server.** Base classes are in `nemo_gym/base_resources_server.py`.

- `SimpleResourcesServer.setup_webserver` (line ~243) installs the session middleware and the routes
  `/seed_session`, `/close_session`, `/verify`, `/aggregate_metrics` and `GET /reverify_mode`.
  Subclasses add tool routes with `app.post("/<tool>")(...)`.
- `verify` is abstract.
- `BaseVerifyResponse` (line 107) has these fields:
  - `reward` (float)
  - `mask_sample`: exclude the sample, because the environment failed rather than the model
  - `failure_kind`: shared vocabulary in `nemo_gym/failure_kinds.py`, e.g. `session_lost`
  - `failure_reason`
- `REVERIFY_MODE` is a ClassVar on the config: `stateless`, `unsupported` or `unknown`.
- The server config needs `domain` for resources servers (`config_types.py` `Domain`; `games` exists).
  `allowed_agents` optionally restricts which agent types may be paired with it.
- `ray_enabled = False` on the class keeps the server itself off Ray
  (`server_utils.py` `_server_uses_ray`). Leaving it undeclared warns and enables Ray.

**Per-rollout state** (`nemo_gym/server_utils.py`).

- `SimpleServer.setup_session_middleware` (line 1198) puts a uuid at `request.session["session_id"]`
  (`SESSION_ID_KEY`, line 863). It lives in a Starlette signed-cookie session named
  `<ServerClass>___<config name>`.
- The template is `resources_servers/example_session_state_mgmt/app.py`:
  - `seed_session(request, body)` stores state under that id;
  - the tools and `verify` read it back from the cookie.
- The newer Environment-Server protocol seeds with `ResourcesSeedSessionRequest`
  (`resources_session_id`, `episode_id`, `task_id`, `task_data`; `extra="forbid"`) and closes with
  `/close_session`. Its contract, enforced by
  `nemo_gym/testing/session_conformance.check_resources_session_contract`:
  - a repeated seed is idempotent;
  - close is idempotent;
  - closing a never-seeded id succeeds;
  - a seed after close must fail.

**Agent.** `responses_api_agents/simple_agent/app.py`.

- The legacy `/run` (line 420) does these steps:
  1. POSTs the whole dataset row (`body.model_dump()`) to the resources server's `/seed_session`;
  2. runs `/v1/responses` with the returned cookies;
  3. POSTs row + `response` to `/verify` with the same cookies.
- In the loop, each model `function_call` becomes `POST /<tool name>` with the JSON arguments. The
  raw response body, including non-2xx error bodies, is returned to the model as the
  `function_call_output`.
- `max_steps` caps model calls per rollout (line 336).
- The final `response.output` holds every function_call / function_call_output / message item.

**Environment server and routing.**

- Rollout collection reaches an agent only through an environment server.
- `environment_servers/legacy_agent` relays `/run` and `/aggregate_metrics` unchanged.
- If a config has an agent without one, Gym generates a relay and logs a deprecation warning
  (`nemo_gym/global_config.py` ~1136–1188). `error_on_agent_without_environment_server: true` makes
  this an error.

**Model servers.**

- `responses_api_models/openai_model` calls `POST {openai_base_url}/responses`
  (`nemo_gym/openai_utils.py` `create_response`). Its config resolves `${policy_base_url}`,
  `${policy_api_key}` and `${policy_model_name}`.
- `responses_api_models/vllm_model` converts Responses requests to `POST {base_url}/chat/completions`
  unless `is_responses_native` is set (`app.py` ~461, ~956).
- `gym env start` waits until each model endpoint accepts a connection
  (`model_endpoint_readiness_timeout_seconds`, default 600).

**Dataset rows.**

- Each JSONL row has `responses_create_params` (`input` messages and `tools`; Responses-API function
  tools with `strict`).
- Extra top-level fields are allowed. The agent forwards them to `seed_session` and `verify`.
- `agent_ref: {type: responses_api_agents, name: ...}` routes a row without `--agent`.
- Agent configs list `datasets` (`name`, `type`, `jsonl_fpath`). `license` is required for `train`
  and `validation` types (`config_types.py` `DatasetConfig`).

**Server discovery and venvs.**

- `nemo_gym/cli/env.py` `_resolve_server_dir` (line 330) resolves `<type>/<name>` under these roots,
  in order: `NEMO_GYM_EXTRA_ROOTS` / `--search-dir`, the cwd, then the install root.
- A directory counts as a server only if it has `requirements.txt` or `pyproject.toml`.
- Entrypoints must be relative; Gym runs `python app.py` from that directory with it on `PYTHONPATH`.
- `nemo_gym/cli/setup_command.py`:
  - `get_venv_path` (line 107) uses `<uv_venv_dir>/<type>/<name>/.venv`, defaulting to the cwd for
    wheel installs.
  - `setup_env_command` creates the venv and `uv pip install`s `nemo-gym==<version>` + the
    requirements.
  - With `skip_venv_if_present=true` and an existing `bin/python` + `bin/activate`, it only
    `source`s the venv (line 132). This makes fully offline runs possible.
- The cwd also receives `cache/` and `results/` (`nemo_gym/__init__.py` `WORKING_DIR`).

**CLI** (`nemo_gym/cli/main.py`; the old `ng_*` commands are deprecated aliases).

- `gym env start` starts the head server (default port 11000; `+head_server.port=` overrides it)
  and all configured servers.
  - It prints `All N / N servers ready!` when they are up.
  - It always initializes a local Ray cluster, even if every server sets `ray_enabled=False`
    (`RunHelper._start`, `cli/env.py` line 464).
  - Ray honours `RAY_TMPDIR`. SIGINT shuts down the servers and that Ray cluster.
- `gym eval run --no-serve --agent A --input I --output O [--limit N] [--num-repeats K]
  [--concurrency C]` collects rollouts against running servers.
  - These flags map to Hydra keys `+agent_name`, `+input_jsonl_fpath`, `+output_jsonl_fpath`, ...
  - Outputs: `O`, `O_materialized_inputs.jsonl`, `O_aggregate_metrics.json`, `quality_summary.json`.
- `gym eval run` without `--no-serve` is the end-to-end mode (`E2ERolloutCollectionConfig`). It
  rejects `--input` and runs the prepared dataset split (`train`/`validation`/`benchmark`) declared in
  the configs.

**Metrics** (`nemo_gym/reward_profile.py`).

- `/aggregate_metrics` computes mean/max/min/median/std of every top-level numeric or bool field of
  the verify responses. Masked samples are left out.
- `compute_metrics(tasks)` adds custom metrics.
- `get_key_metrics` picks the headline ones (default: all `mean/*`).

## 2. Design

The task is a segment (DESIGN.md §9).

- A row holds `seed`, `start_day` (0-based `info["day"]`), `horizon_days` (7), `max_tool_calls`
  (60), `expert_days` and the prompt.
- `seed_session` replays the deterministic expert to the first state with `day >= start_day` (one
  Sleep can end two days). It then computes the reference E by rolling the expert forward on
  `env.clone()` until the next request completes, or until the horizon passes (E = none).
- `tasks.py` keeps only start days where the expert succeeds within the horizon. It uses disjoint
  seed ranges: train `<2^32`, val `[2^32, 2^33)`, test `[2^33, 2^34)`.

The tools (`observe`, `list_actions`, `act`, `status`) are pure functions of the session.

- `act` takes one macro or a day plan. The plan stops at the first unknown or currently-invalid name,
  which is not executed; the error gives difflib close matches and is counted in `invalid_actions`.
  It also stops when the day ends or the segment ends.
- Every tool call counts against the budget. The budget, the horizon or the first new request ends
  the segment.

The reward is lexicographic, in [0, 1):

- success: `0.5 + 0.5·(E+c)/((E+c)+(A+c))`, c = 0.25 days, A and E in fractional in-game days;
- failure: `0.45·progress`, partial credit from request-slot fractions parsed from `describe()`.

The code is split in two layers:

- `core.py` holds the logic and does not import NeMo Gym.
- `resources_servers/furrow/app.py` is the HTTP layer:
  - session id = Gym's cookie session id (legacy `/run` path), or `resources_session_id` (typed
    path);
  - Furrow work runs via `run_in_threadpool`, with a lock per session plus one for the store;
  - TTL and LRU eviction;
  - `/verify` scores and drops the session and caches the result;
  - a missing session gives a masked sample (`session_lost`);
  - `REVERIFY_MODE = unsupported`, because the reward needs the live session.

Offline runs:

- `launch.prepare_shared_venvs` symlinks every `<uv_venv_dir>/<type>/<name>/.venv` to one venv
  (`.venv-nemo`).
- `fake_openai.py` serves `/v1/responses` and `/v1/chat/completions` with a seeded random policy:
  observe → (list_actions → act)* until `done`.

## 3. Tests

`.venv-nemo/bin/python -m pytest -q python/nemo_gym_furrow` gives **38 passed, 0 skipped** in about
35 s. It needs no network and no GPU.

- `test_core.py` (19):
  - grouping round-trip over valid sets along an expert run;
  - deterministic start states and expert reference (`stable_hash`);
  - identical actions give identical outputs and reward;
  - unknown and masked names give errors, close matches and no state change;
  - day-plan stop at day end; plan and budget caps; horizon end;
  - the expert used as the agent gets exactly reward 0.75 (A = E);
  - 4 threads on one session;
  - reward ordering over 2,000 random cases; monotonicity; partial-progress parsing;
  - store idempotency, close, result cache and eviction.
- `test_app.py` (9), FastAPI `TestClient` against the real app:
  - a rollout through the routes; the prompt state equals the server state;
  - per-cookie isolation; a masked verify when there is no session; 422 on bad rows; config defaults;
  - Gym's `check_resources_session_contract`; a typed seed reaching the tools; `/aggregate_metrics`.
- `test_tasks_prompt_fake.py` (8):
  - disjoint seed ranges; rows match the server's start state and expert reference; reproducibility;
  - the `--min-expert-days` filter;
  - prompt length ≤ 800 tokens (estimated: 2,646 chars / 413 words);
  - fake policy and reply validation (`NeMoGymResponse`, `NeMoGymChatCompletion`); the fake plays a
    session to the end.
- `test_integration.py` (2, parametrized over `openai_model` and `vllm_model`): real harness.
  1. Generate 3 tasks.
  2. Start the fake.
  3. `gym env start` (Furrow + `simple_agent` + `legacy_agent` + model adapter, shared venv, Ray in
     a temporary `RAY_TMPDIR`).
  4. `gym eval run --no-serve` collects 3 rollouts.
  5. Check: unmasked; reward in [0, 1) and consistent with `success`; tool calls in the trajectory
     equal the server's count; aggregate `mean/reward` matches; clean shutdown.

The README's step 1–3 commands were also run by hand.

## 4. Unverified

- Real models (OpenAI, vLLM, NIM) as the policy. Only the fake was used. Tool-call quality, context
  growth (about 12–13k tokens of prompt + tool output for a 60-call segment) and the reward
  distribution for real LLMs are unknown.
- The NeMo RL GRPO recipe (`configs/nemo_rl_grpo_furrow.yaml`). It was written from NeMo RL main @
  61bc275 docs and examples and has never been run. Open questions:
  - whether its pinned Gym honours `NEMO_GYM_EXTRA_ROOTS`, absolute `config_paths` and
    `skip_venv_if_present`;
  - GPU/cluster settings;
  - whether flat rows with `agent_ref` route as described.
- Gym building the Furrow server venv itself from `requirements.txt`. `furrow` is not on PyPI, so
  that path needs a wheel index; only the prepared-venv path was tested.
- The typed Environment-Server flow end to end (`single_agent_turn` etc.). Only the seed/close
  contract and cookie hand-off were tested, in process.
- Multi-worker servers (`num_workers > 1`). Sessions are process-local, so requests for one rollout
  must reach the same worker.
