import sys
import unittest
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is in sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from level0_naive.naive_generator import generate_padded_batch, generate_sequential
except ModuleNotFoundError:
    from naive_generator import generate_padded_batch, generate_sequential


class TestNaiveGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use a tiny model for fast, reproducible testing
        cls.model_id = "hf-internal-testing/tiny-random-gpt2"
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id).to(cls.device)
        cls.model.eval()

        cls.prompts = [
            "Hello world",
            "The quick brown fox jumps over",
            "System architecture is"
        ]

    def test_generate_sequential_structure(self):
        """Verify sequential generator returns list of lists of generated token ids."""
        max_new_tokens = 8
        results = generate_sequential(
            self.model,
            self.tokenizer,
            self.prompts,
            max_new_tokens=max_new_tokens
        )

        self.assertIsInstance(results, list, "Output must be a list")
        self.assertEqual(len(results), len(self.prompts), "Must return results for each prompt")

        for i, seq in enumerate(results):
            self.assertIsInstance(seq, list, f"Sequence {i} must be a list of token IDs")
            self.assertGreater(len(seq), 0, f"Sequence {i} should have generated tokens")
            self.assertLessEqual(
                len(seq),
                max_new_tokens,
                f"Sequence {i} generated more than max_new_tokens"
            )
            for token in seq:
                self.assertIsInstance(token, int, f"Token {token} in sequence {i} must be an integer")

    def test_generate_padded_batch_structure(self):
        """Verify padded batch generator returns list of lists of generated token ids."""
        max_new_tokens = 8
        results = generate_padded_batch(
            self.model,
            self.tokenizer,
            self.prompts,
            max_new_tokens=max_new_tokens
        )

        self.assertIsInstance(results, list, "Output must be a list")
        self.assertEqual(len(results), len(self.prompts), "Must return results for each prompt")

        for i, seq in enumerate(results):
            self.assertIsInstance(seq, list, f"Sequence {i} must be a list of token IDs")
            self.assertGreater(len(seq), 0, f"Sequence {i} should have generated tokens")
            self.assertLessEqual(
                len(seq),
                max_new_tokens,
                f"Sequence {i} generated more than max_new_tokens"
            )

    def test_sequential_vs_batched_parity(self):
        """
        Under greedy argmax decoding, sequential and batch generation
        must produce identical token output sequences.
        """
        max_new_tokens = 6
        seq_results = generate_sequential(
            self.model,
            self.tokenizer,
            self.prompts,
            max_new_tokens=max_new_tokens
        )
        batch_results = generate_padded_batch(
            self.model,
            self.tokenizer,
            self.prompts,
            max_new_tokens=max_new_tokens
        )

        for i in range(len(self.prompts)):
            self.assertEqual(
                seq_results[i],
                batch_results[i],
                f"Prompt {i} output divergence between sequential and batched decoding"
            )


if __name__ == "__main__":
    unittest.main()
