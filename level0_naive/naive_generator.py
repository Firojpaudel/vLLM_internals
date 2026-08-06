import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_sequential(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 32
) -> List[List[int]]:
    """
    Process prompts ONE BY ONE sequentially.
    
    Task for user:
    1. For each prompt string, encode it into input_ids using the tokenizer.
    2. Pass input_ids into model(input_ids, use_cache=True) to complete the prefill phase.
    3. Extract past_key_values (the initial KV cache) and the initial output logits.
    4. Run a decode loop for max_new_tokens steps:
       a. Pick the next token (e.g. via torch.argmax on the last logit).
       b. Append the new token to output sequence.
       c. Pass single next_token_id and past_key_values into model(next_token_id, past_key_values=past_key_values, use_cache=True).
       d. Observe the shape of past_key_values[0][0] at each step.
       e. Stop if EOS token is produced.
    
    Returns:
        List of generated token ID lists for each prompt.
    """
    results = []
    # TODO: Implement sequential autoregressive loop
    return results


def generate_padded_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 32
) -> List[List[int]]:
    """
    Process all prompts together in a SINGLE PADDED BATCH.
    
    Task for user:
    1. Tokenize all prompts together using tokenizer(prompts, padding=True, return_tensors="pt").
    2. Pass batch input_ids and attention_mask into model(..., use_cache=True) for batch prefill.
    3. Run decode loop for max_new_tokens steps for the entire batch simultaneously:
       a. Pick next tokens for each sequence in batch via torch.argmax.
       b. Pass next_tokens and updated past_key_values back into model.
       c. Observe batch KV cache memory overhead.
    
    Returns:
        List of generated token ID lists for each prompt.
    """
    results = []
    # TODO: Implement padded batch generation loop
    return results


if __name__ == "__main__":
    # Test script entry point
    print("Level 0 Naive Generator Skeleton ready for implementation.")
