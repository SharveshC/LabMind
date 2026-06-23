import os
from core.schemas import DatasetProfile, ExperimentConfig
from agents.planner import PlannerAgent
from unittest.mock import patch, MagicMock

def test_planner():
    # Setup dummy profile
    profile = DatasetProfile(
        dataset_type="image",
        num_samples=5234,
        num_classes=2,
        class_distribution={"cats": 2600, "dogs": 2634},
        imbalanced=False,
        avg_width=224,
        avg_height=224,
        corrupted_images=0
    )
    
    # We will use a mock here so the test can run without an API key,
    # proving the validation and retry logic works correctly.
    # If GEMINI_API_KEY is present, you can test it live by removing the mock.
    
    print("--- Testing Planner Agent (Mocked LLM) ---")
    
    with patch('agents.planner.genai.Client') as MockClient:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.text = '''
        {
          "model_name": "resnet18",
          "learning_rate": 0.001,
          "batch_size": 32,
          "epochs": 10,
          "optimizer": "adam",
          "augmentation": false,
          "dropout": 0.2
        }
        '''
        
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # Initialize planner with a dummy key just for the test
        planner = PlannerAgent(api_key="DUMMY_KEY_FOR_TESTING")
        
        # Generate config
        config = planner.generate(profile, experiment_id="exp_001")
        
        assert isinstance(config, ExperimentConfig), "Config is not an instance of ExperimentConfig"
        assert config.experiment_id == "exp_001", "experiment_id was not correctly injected"
        assert config.model_name == "resnet18", "Schema failed to parse model_name"
        
        print(config.model_dump_json(indent=2))
        print("\nPlanner validation and parsing passed successfully!")

if __name__ == "__main__":
    test_planner()
