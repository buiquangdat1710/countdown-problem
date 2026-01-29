#!/usr/bin/env python3
"""
Script to use a trained GRPO model for arithmetic countdown problems.

This script loads a model trained with train_grpo_hydra.py and provides
both interactive and batch evaluation modes for solving arithmetic problems.
"""

import logging
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset.grpo import map_problem_description_to_conversation_grpo
from src.utils.rewards import _is_valid_arithmetic_expression
from src.utils.string_helper import extract_answer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_inference")


class GRPOModelInference:
    """Class for loading and running inference with a trained GRPO model."""

    def __init__(
        self,
        sft_model_path: str | None = None,
        grpo_model_path: str | None = None,
        base_model_id: str = "Qwen/Qwen2.5-Math-1.5B",
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the model inference class.

        Args:
            model_path: Path to the trained LoRA model directory
            base_model_id: Base model identifier from Hugging Face
            device: Device to load the model on
            dtype: Torch data type for the model
        """
        self.sft_model_path = sft_model_path
        self.grpo_model_path = grpo_model_path
        self.base_model_id = base_model_id
        self.device = device
        self.dtype = dtype

        self.tokenizer = None
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the base model, LoRA adapters, and tokenizer."""
        logger.info(f"Loading base model: {self.base_model_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            dtype=self.dtype,
            device_map=self.device,
        )

        # Load SFT model
        if self.sft_model_path:
            logger.info(f"Loading SFT LoRA adapters from: {self.sft_model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.sft_model_path)
            self.model = self.model.merge_and_unload()

        # Check if LoRA adapters exist
        if self.grpo_model_path:
            logger.info(f"Loading GRPO LoRA adapters from: {self.grpo_model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.grpo_model_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        logger.info("Model loaded successfully")

    def _format_conversation(self, problem_description: str) -> list[dict[str, str]]:
        """
        Format the problem description into the expected conversation format.

        Args:
            problem_description: The arithmetic problem description

        Returns:
            List of conversation messages
        """
        result = map_problem_description_to_conversation_grpo(
            {
                "problem_description": problem_description,
            }
        )
        return result["prompt"]

    def _generate_response(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate a response from the model given conversation messages.

        Args:
            messages: List of conversation messages
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter

        Returns:
            Generated response text
        """
        # Format messages using the tokenizer's chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize the input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )

        # Move to device
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return response.strip()

    def solve_problem(
        self,
        problem_description: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> tuple[str, str, bool]:
        """
        Solve a single arithmetic countdown problem.

        Args:
            problem_description: The problem description
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            verbose: Whether to print detailed output

        Returns:
            Tuple of (full_response, extracted_answer, is_valid_format)
        """
        # Format conversation
        messages = self._format_conversation(problem_description)

        # Generate response
        response = self._generate_response(
            messages, max_new_tokens=max_new_tokens, temperature=temperature
        )

        # Extract answer
        extracted_answer = extract_answer(response)
        is_valid = _is_valid_arithmetic_expression(extracted_answer)

        return response, extracted_answer, is_valid
