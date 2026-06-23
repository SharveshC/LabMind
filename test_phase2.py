import os
import shutil
from PIL import Image
from core.schemas import ExperimentConfig
from services.dataset import DatasetService
from services.executor import ExecutorService
from services.evaluation import EvaluationService

def setup_dummy_dataset():
    """Sets up a tiny dummy dataset to test dataset.py."""
    base_path = "datasets/cats_dogs"
    os.makedirs(os.path.join(base_path, "cats"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "dogs"), exist_ok=True)
    
    # Create actual dummy images using PIL
    for i in range(3):
        img = Image.new("RGB", (224, 224), color="red")
        img.save(os.path.join(base_path, "cats", f"cat_{i}.jpg"))
    for i in range(4):
        img = Image.new("RGB", (224, 224), color="blue")
        img.save(os.path.join(base_path, "dogs", f"dog_{i}.jpg"))

def cleanup_dummy_dataset():
    if os.path.exists("datasets"):
        shutil.rmtree("datasets")

def test_phase2():
    print("--- Testing Step 1: Dataset Service ---")
    setup_dummy_dataset()
    dataset_service = DatasetService()
    profile = dataset_service.analyze("datasets/cats_dogs")
    print(profile.model_dump_json(indent=2))
    
    print("\n--- Testing Step 2, 3, & 4: Trainer, Executor, Evaluator ---")
    config = ExperimentConfig(
        experiment_id="exp_001",
        model_name="resnet18",
        learning_rate=0.001,
        batch_size=32,
        epochs=1,
        optimizer="adam"
    )
    
    executor = ExecutorService()
    evaluator = EvaluationService()
    
    print(f"Running executor for {config.experiment_id}...")
    result = executor.run(config)
    print(f"Execution Status: {result.status}")
    if result.status == "success":
        print(f"Train Loss: {result.train_loss}, Val Loss: {result.val_loss}")
        print(f"Weights saved at: {result.model_weights_path}")
    else:
        print(f"Error: {result.error_message}")
    
    print(f"\nRunning evaluation for {config.experiment_id}...")
    report = evaluator.evaluate(result)
    print(report.model_dump_json(indent=2))
    
    cleanup_dummy_dataset()
    print("\nPhase 2 tests passed successfully!")

if __name__ == "__main__":
    test_phase2()
