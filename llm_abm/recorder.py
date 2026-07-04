"""TranscriptRecorder: shared append-only JSONL logging for any simulation.

General across simulations (Minority Game, Polis society, ...) — every backend
call's prompt/response can be logged here so a run is fully auditable and
re-analyzable without re-running (or re-paying for) anything.
"""

from __future__ import annotations

import json
from pathlib import Path


class TranscriptRecorder:
    """Append-only JSONL log of every prompt/response in a run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a")

    def log(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.close()
