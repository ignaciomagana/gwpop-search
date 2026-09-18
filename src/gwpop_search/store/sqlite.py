"""Append-only SQLite state store for models, evaluations, and promotions."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator, Mapping

from gwpop_search.grammar import ModelSpec
from gwpop_search.search.scheduler import (
    EvaluationRecord,
    PromotionDecision,
)


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS models (
    model_hash TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    run_id TEXT PRIMARY KEY,
    model_hash TEXT NOT NULL,
    fidelity TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    diagnostics_pass INTEGER NOT NULL,
    screen_value REAL,
    compute_cost REAL NOT NULL,
    run_config_json TEXT NOT NULL,
    artifact_path TEXT,
    FOREIGN KEY(model_hash) REFERENCES models(model_hash),
    UNIQUE(model_hash, fidelity, seed, run_config_json)
);

CREATE TABLE IF NOT EXISTS promotions (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    model_hash TEXT NOT NULL,
    from_fidelity TEXT NOT NULL,
    to_fidelity TEXT,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    scheduler_version TEXT NOT NULL,
    FOREIGN KEY(model_hash) REFERENCES models(model_hash)
);
"""


class ResultStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def register_model(self, model: ModelSpec) -> None:
        payload = model.canonical_json()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT spec_json FROM models WHERE model_hash = ?",
                (model.model_hash,),
            ).fetchone()
            if existing is not None:
                if existing["spec_json"] != payload:
                    raise ValueError(
                        f"model hash collision or spec mismatch for {model.model_hash}"
                    )
                return
            connection.execute(
                "INSERT INTO models(model_hash, spec_json) VALUES (?, ?)",
                (model.model_hash, payload),
            )

    def record_evaluation(
        self,
        run_id: str,
        record: EvaluationRecord,
        *,
        seed: int,
        run_config: Mapping[str, object],
        artifact_path: str | None = None,
    ) -> None:
        config_json = json.dumps(run_config, sort_keys=True, separators=(",", ":"))
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluations(
                    run_id, model_hash, fidelity, seed, status,
                    diagnostics_pass, screen_value, compute_cost,
                    run_config_json, artifact_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    record.model_hash,
                    record.fidelity.value,
                    int(seed),
                    record.status,
                    int(record.diagnostics_pass),
                    record.screen_value,
                    float(record.compute_cost),
                    config_json,
                    artifact_path,
                ),
            )

    def record_promotions(
        self,
        decisions: tuple[PromotionDecision, ...] | list[PromotionDecision],
    ) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO promotions(
                    model_hash, from_fidelity, to_fidelity, decision,
                    reason, scheduler_version
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.model_hash,
                        item.from_fidelity.value,
                        None if item.to_fidelity is None else item.to_fidelity.value,
                        item.decision,
                        item.reason,
                        item.scheduler_version,
                    )
                    for item in decisions
                ],
            )

    def promotion_history(self) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM promotions ORDER BY sequence"
            ).fetchall()
        return [dict(row) for row in rows]

    def evaluations(
        self,
        *,
        model_hash: str | None = None,
        fidelity: str | None = None,
    ) -> list[dict[str, object]]:
        clauses = []
        params = []
        if model_hash is not None:
            clauses.append("model_hash = ?")
            params.append(model_hash)
        if fidelity is not None:
            clauses.append("fidelity = ?")
            params.append(fidelity)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evaluations" + where + " ORDER BY run_id",
                params,
            ).fetchall()
        return [dict(row) for row in rows]
