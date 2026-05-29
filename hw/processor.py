from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()
        self.transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        image = image.convert("RGB")
        pixel_values = self.transform(image)
        return pixel_values.unsqueeze(0)

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        options_str = " ".join(sample.options)
        prompt = f"{IMAGE_START_TOKEN}{IMAGE_TOKEN * self.config.num_image_tokens}{IMAGE_END_TOKEN} "
        prompt += f"Question: {sample.question} Options: {options_str} Answer:"
        
        if include_answer:
            prompt += f" {sample.answer}"

        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        full_prompt = self.build_prompt(sample, include_answer=True)
        instruction_prompt = self.build_prompt(sample, include_answer=False)

        full_enc = self.tokenizer(full_prompt, max_length=self.config.max_length, truncation=True, return_tensors="pt")
        inst_enc = self.tokenizer(instruction_prompt, max_length=self.config.max_length, truncation=True, return_tensors="pt")

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)
        
        labels = input_ids.clone()
        labels[:inst_enc["input_ids"].shape[1]] = self.config.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.config.ignore_index)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }
