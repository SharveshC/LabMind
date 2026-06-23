import os
import json
from google import genai
from core.schemas import DatasetProfile, ExperimentConfig
from core.prompts import PLANNER_PROMPT

class PlannerAgent:
    def __init__(self, api_key: str = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=key)

    def generate(self, profile: DatasetProfile, experiment_id: str, max_retries: int = 3) -> ExperimentConfig:
        """
        Takes a dataset profile, asks Gemini to plan an experiment, 
        and strict-validates the response into a Pydantic ExperimentConfig.
        Retries upon validation failure.
        """
        
        prompt = PLANNER_PROMPT.format(dataset_profile=profile.model_dump_json(indent=2))
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-pro',
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.4,
                        response_mime_type="application/json"
                    )
                )
                
                raw_json = response.text
                # Sometimes LLMs wrap JSON in markdown even when told not to. Clean it just in case.
                raw_json = raw_json.strip()
                if raw_json.startswith("```json"):
                    raw_json = raw_json[7:]
                if raw_json.startswith("```"):
                    raw_json = raw_json[3:]
                if raw_json.endswith("```"):
                    raw_json = raw_json[:-3]
                raw_json = raw_json.strip()
                
                # Parse JSON
                data = json.loads(raw_json)
                
                # Force inject the experiment_id as per rules (LLM should not generate it)
                data["experiment_id"] = experiment_id
                
                # Validate against schema
                config = ExperimentConfig.model_validate(data)
                return config
                
            except Exception as e:
                print(f"[PlannerAgent] Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Planner failed to generate valid configuration after maximum retries.") from e

        raise RuntimeError("Planner failed unexpectedly.")
