"""Server entry point for storage, asset acceptance, result browsing, and runs.

Formal GPU batches remain disabled until the administrator-controlled GPU
policy is implemented and validated. The available ``smoke-drop`` and
``smoke-slope`` commands are single non-authoritative MuJoCo CPU cases.
"""

from experiment_runner.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
