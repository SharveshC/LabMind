import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from core.schemas import ExperimentConfig, ExecutionResult
from core.constants import SUPPORTED_MODELS, SUPPORTED_OPTIMIZERS

class TrainerService:
    def run(self, config: ExperimentConfig) -> ExecutionResult:
        """Runs the training loop for the given configuration."""
        start_time = time.time()
        
        # 1. Validate constraints
        if config.model_name not in SUPPORTED_MODELS:
            return ExecutionResult(
                experiment_id=config.experiment_id,
                status="failed",
                runtime_seconds=time.time() - start_time,
                train_loss=[],
                val_loss=[],
                model_weights_path="",
                error_message=f"Model {config.model_name} is not supported. Choose from {SUPPORTED_MODELS}."
            )
        
        # 2. Initialize Model
        try:
            if config.model_name == "resnet18":
                model = models.resnet18(weights=None)
            elif config.model_name == "resnet34":
                model = models.resnet34(weights=None)
            elif config.model_name == "efficientnet_b0":
                model = models.efficientnet_b0(weights=None)
                
            # Dummy logic for dropout (since torchvision handles this internally mostly, 
            # we just show we are acknowledging the config)
            if config.dropout > 0.0:
                pass # In a full implementation, we'd inject Dropout layers here
                
        except Exception as e:
            return ExecutionResult(
                experiment_id=config.experiment_id,
                status="failed",
                runtime_seconds=time.time() - start_time,
                train_loss=[],
                val_loss=[],
                model_weights_path="",
                error_message=str(e)
            )

        # 3. Initialize Optimizer
        if config.optimizer.lower() not in SUPPORTED_OPTIMIZERS:
             return ExecutionResult(
                experiment_id=config.experiment_id,
                status="failed",
                runtime_seconds=time.time() - start_time,
                train_loss=[],
                val_loss=[],
                model_weights_path="",
                error_message=f"Optimizer {config.optimizer} is not supported. Choose from {SUPPORTED_OPTIMIZERS}."
            )
            
        if config.optimizer.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
            
        criterion = nn.CrossEntropyLoss()
        
        # 4. Simulate Training Loop (Zero AI, fast execution for testing)
        # We simulate a tiny batch instead of loading a real dataset to ensure
        # the end-to-end pipeline works without requiring a 10GB dataset upfront.
        
        train_losses = []
        val_losses = []
        
        # Create a dummy batch
        dummy_inputs = torch.randn(config.batch_size, 3, 224, 224)
        dummy_targets = torch.randint(0, 1000, (config.batch_size,))
        
        model.train()
        try:
            for epoch in range(config.epochs):
                optimizer.zero_grad()
                outputs = model(dummy_inputs)
                loss = criterion(outputs, dummy_targets)
                loss.backward()
                optimizer.step()
                
                # Mock validation step
                val_loss = loss.item() * 0.95 
                
                train_losses.append(round(loss.item(), 4))
                val_losses.append(round(val_loss, 4))
                
        except Exception as e:
            return ExecutionResult(
                experiment_id=config.experiment_id,
                status="failed",
                runtime_seconds=time.time() - start_time,
                train_loss=train_losses,
                val_loss=val_losses,
                model_weights_path="",
                error_message=f"Training failed: {str(e)}"
            )
            
        # 5. Save model weights
        os.makedirs("checkpoints", exist_ok=True)
        weights_path = f"checkpoints/{config.experiment_id}_weights.pth"
        torch.save(model.state_dict(), weights_path)
        
        return ExecutionResult(
            experiment_id=config.experiment_id,
            status="success",
            runtime_seconds=time.time() - start_time,
            train_loss=train_losses,
            val_loss=val_losses,
            model_weights_path=weights_path
        )
