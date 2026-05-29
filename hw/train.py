from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """"""Run one optimization step and return scalar loss."""
    model.train()
    
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    outputs = model(batch)
    loss = outputs.loss

    if not torch.isfinite(loss):
        raise ValueError(f"Loss is {loss.item()}, stopping training")

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    proc_config = ProcessorConfig(
        image_size=config["model"]["image_size"],
        num_image_tokens=config["model"]["num_image_tokens"],
    )
    processor = MathVLMProcessor(tokenizer, config=proc_config)

    train_dataset = MathVQADataset(
        manifest_path=config["data"]["train_manifest"],
        split="train",
        max_samples=16 if fast_train else config["data"].get("max_samples"),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=processor.collate,
    )

    vision_encoder = AutoModel.from_pretrained(config["model"]["vision_model_id"])
    language_model = AutoModelForCausalLM.from_pretrained(config["model"]["text_model_id"])
    
    model_config = ModelConfig(
        vision_hidden_size=config["model"]["vision_hidden_size"],
        text_hidden_size=config["model"]["text_hidden_size"],
        num_image_tokens=config["model"]["num_image_tokens"],
        image_token_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
    )
    
    model = MathVLM(vision_encoder, language_model, model_config)
    model.freeze_backbones()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=float(config["train"]["lr"]))

    max_steps = 2 if fast_train else config["train"]["max_steps"]
    current_step = 0
    
    
    while current_step < max_steps:
        for batch in train_loader:
            if current_step >= max_steps:
                break
            
            loss = train_one_step(model, batch, optimizer)
            
            current_step += 1

    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.adapter.state_dict(), output_dir / "adapter.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
