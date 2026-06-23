import sqlite3
import json
from typing import List, Optional
from core.schemas import ExperimentRecord, ExperimentConfig, EvaluationReport, ResearchCritique

class MemorySystem:
    def __init__(self, db_path: str = "labmind_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    parent_experiment_id TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    runtime_seconds REAL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT,
                    critique_json TEXT
                )
            """)
            conn.commit()

    def save_experiment(self, record: ExperimentRecord):
        """Saves or updates an ExperimentRecord in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            metrics_json = record.metrics.model_dump_json() if record.metrics else None
            critique_json = record.critique.model_dump_json() if record.critique else None
            
            cursor.execute("""
                INSERT INTO experiments (experiment_id, parent_experiment_id, created_at, status, runtime_seconds, config_json, metrics_json, critique_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    parent_experiment_id=excluded.parent_experiment_id,
                    created_at=excluded.created_at,
                    status=excluded.status,
                    runtime_seconds=excluded.runtime_seconds,
                    config_json=excluded.config_json,
                    metrics_json=excluded.metrics_json,
                    critique_json=excluded.critique_json
            """, (
                record.experiment_id,
                record.parent_experiment_id,
                record.created_at,
                record.status,
                record.runtime_seconds,
                record.config.model_dump_json(),
                metrics_json,
                critique_json
            ))
            conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieves an ExperimentRecord by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT experiment_id, parent_experiment_id, created_at, status, runtime_seconds, config_json, metrics_json, critique_json FROM experiments WHERE experiment_id = ?", (experiment_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
                
            return self._row_to_record(row)

    def get_lineage(self, leaf_experiment_id: str) -> List[ExperimentRecord]:
        """Retrieves the full lineage of experiments ending with the given ID."""
        lineage = []
        current_id = leaf_experiment_id
        
        while current_id:
            record = self.get_experiment(current_id)
            if not record:
                break
            lineage.insert(0, record) # Insert at beginning so lineage is from oldest to newest
            current_id = record.parent_experiment_id
            
        return lineage
        
    def get_all_experiments(self) -> List[ExperimentRecord]:
        """Retrieves all stored experiments."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT experiment_id, parent_experiment_id, created_at, status, runtime_seconds, config_json, metrics_json, critique_json FROM experiments")
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row) -> ExperimentRecord:
        exp_id, parent_id, created_at, status, runtime_seconds, config_str, metrics_str, critique_str = row
        
        config = ExperimentConfig.model_validate_json(config_str)
        metrics = EvaluationReport.model_validate_json(metrics_str) if metrics_str else None
        critique = ResearchCritique.model_validate_json(critique_str) if critique_str else None
        
        return ExperimentRecord(
            experiment_id=exp_id,
            parent_experiment_id=parent_id,
            created_at=created_at,
            status=status,
            runtime_seconds=runtime_seconds,
            config=config,
            metrics=metrics,
            critique=critique
        )
