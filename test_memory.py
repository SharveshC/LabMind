from core.schemas import ExperimentConfig, ExperimentRecord
from core.memory import MemorySystem
import os

def test_memory():
    # Remove old test DB if exists
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")
        
    memory = MemorySystem("test_memory.db")
    
    config1 = ExperimentConfig(
        experiment_id="exp_001",
        model_name="resnet18",
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        optimizer="adam",
        augmentation=False,
        dropout=0.0
    )
    
    exp1 = ExperimentRecord(
        experiment_id="exp_001",
        parent_experiment_id=None,
        created_at="2026-06-20T12:00:00Z",
        status="completed",
        runtime_seconds=120.5,
        config=config1
    )
    
    memory.save_experiment(exp1)
    
    config2 = ExperimentConfig(
        experiment_id="exp_002",
        model_name="resnet18",
        learning_rate=0.0005, # lowered LR
        batch_size=32,
        epochs=10,
        optimizer="adam",
        augmentation=True, # Added augmentation
        dropout=0.0
    )
    
    exp2 = ExperimentRecord(
        experiment_id="exp_002",
        parent_experiment_id="exp_001",
        created_at="2026-06-20T12:05:00Z",
        status="running",
        runtime_seconds=None,
        config=config2
    )
    
    memory.save_experiment(exp2)
    
    lineage = memory.get_lineage("exp_002")
    assert len(lineage) == 2, f"Expected lineage length 2, got {len(lineage)}"
    assert lineage[0].experiment_id == "exp_001", "First item should be root"
    assert lineage[1].experiment_id == "exp_002", "Second item should be leaf"
    
    print("Memory tests passed successfully!")
    
    # Cleanup
    if os.path.exists("test_memory.db"):
        try:
            os.remove("test_memory.db")
        except PermissionError:
            pass # Windows file lock issue with SQLite, can be ignored in testing

if __name__ == "__main__":
    test_memory()
