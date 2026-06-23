from pydantic import BaseModel
from typing import List, Dict, Optional

# --- 1. Dataset Analysis ---
class DatasetProfile(BaseModel):
    dataset_type: str  # e.g., "image"
    num_samples: int
    num_classes: int
    class_distribution: Dict[str, int]
    imbalanced: bool
    avg_width: int
    avg_height: int
    corrupted_images: int

# --- 2. Planning & Evolution ---
class ExperimentConfig(BaseModel):
    experiment_id: str
    model_name: str
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str
    augmentation: bool = False
    dropout: float = 0.0

class StopCondition(BaseModel):
    target_accuracy: float
    max_experiments: int
    patience: int

# --- 3. Training & Evaluation ---
class ExecutionResult(BaseModel):
    experiment_id: str
    status: str # "success", "failed"
    runtime_seconds: float
    train_loss: List[float]
    val_loss: List[float]
    model_weights_path: str
    error_message: Optional[str] = None

class EvaluationReport(BaseModel):
    experiment_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix_path: Optional[str] = None

# --- 4. Critique & Evolution ---
class ResearchCritique(BaseModel):
    experiment_id: str
    issue: str
    confidence: float
    recommendations: List[str]

class EvolutionProposal(BaseModel):
    base_experiment_id: str
    rationale: str
    new_config: ExperimentConfig

# --- 5. Memory & Lineage ---
class ExperimentRecord(BaseModel):
    experiment_id: str
    parent_experiment_id: Optional[str] = None
    created_at: str
    status: str
    runtime_seconds: Optional[float] = None
    config: ExperimentConfig
    metrics: Optional[EvaluationReport] = None
    critique: Optional[ResearchCritique] = None
