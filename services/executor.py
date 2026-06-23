from core.schemas import ExperimentConfig, ExecutionResult
from services.trainer import TrainerService

class ExecutorService:
    def __init__(self):
        self.trainer = TrainerService()

    def run(self, config: ExperimentConfig) -> ExecutionResult:
        """
        Executes an experiment.
        Currently a thin wrapper around TrainerService.
        In V2, this will execute generated PyTorch scripts.
        """
        return self.trainer.run(config)
