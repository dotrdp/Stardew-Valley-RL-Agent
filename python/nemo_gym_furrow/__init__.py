"""NVIDIA NeMo Gym integration for Furrow (DESIGN.md §9).

- ``core``: segment tasks, tools, the lexicographic verify reward, thread-safe session store
  (no NeMo Gym imports).
- ``prompt``: system prompt, tool schemas, dataset-row builder.
- ``tasks``: CLI writing train/val/test task JSONL files.
- ``fake_openai``: tiny OpenAI-compatible model server for offline end-to-end runs.
- ``launch``: helpers to run Gym's servers offline from one prepared venv.
- ``resources_servers/furrow``: the NeMo Gym resources server (app, configs, example data).
"""

__all__ = ["core", "prompt", "tasks", "fake_openai", "launch"]
