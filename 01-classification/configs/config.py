import os
from dataclasses import dataclass


@dataclass
class PetDatasetConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = min(8, os.cpu_count())
    train_split: float = 0.8
    seed: int = 42
    pin_memory: bool = True
    persistent_workers: bool = True
    num_classes = 37  # oxford pet dataset



@dataclass
class ModelConfig:
    num_epochs = 30
    learning_rate = 0.001
    max_lr = 0.0015
    weight_decay: float = 1.0e-4
    dropout = 0.4
    seed = 42
    output_path = "outputs"


class ResNet50Config(ModelConfig):
    model_name: str = "resnet50"


CONFIGS = {
    "resnet50": ResNet50Config,
}


def get_config(model_name: str):
    if model_name not in CONFIGS:
        raise ValueError(f"Unknown model: {model_name}.")
    return CONFIGS[model_name]()
