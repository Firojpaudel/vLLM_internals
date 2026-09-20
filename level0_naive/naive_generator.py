import torch
from typing import List, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_kv_cache_shape(past_key_values: Any, layer_idx: int = 0):
    """
    Safely extract key tensor shape across different transformers versions
    (legacy tuple-of-tuples or modern DynamicCache object).
    """
    if past_key_values is None:
        return "None"
    # Modern transformers DynamicCache (transformers 5.x)
    if hasattr(past_key_values, "layers") and len(past_key_values.layers) > layer_idx:
        layer = past_key_values.layers[layer_idx]
        if hasattr(layer, "keys") and layer.keys is not None:
            return layer.keys.shape
        elif hasattr(layer, "key") and layer.key is not None:
            return layer.key.shape
        elif isinstance(layer, (tuple, list)):
            return layer[0].shape
    # DynamicCache (transformers 4.36-4.4x)
    if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > layer_idx:
        return past_key_values.key_cache[layer_idx].shape
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            return past_key_values.to_legacy_cache()[layer_idx][0].shape
        except Exception:
            pass
    # Legacy tuple of tuples: ((key, value), ...)
    try:
        return past_key_values[layer_idx][0].shape
    except Exception:
        return "N/A"

def generate_sequential(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 32,
    verbose: bool = True
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
    log = print if verbose else (lambda *args, **kwargs: None)

    # Set model to evaluation mode
    model.eval()

    log("\n" + "=" * 60)
    log(f"=== SEQUENTIAL GENERATION ({len(prompts)} prompts) ===")
    log("=" * 60)

    with torch.no_grad():
        # processing one prompt at a time
        for idx, prompt in enumerate(prompts):
            log(f"\n--- [Sequential] Prompt {idx + 1}/{len(prompts)}: {prompt!r} ---")
            
            # tokenize
            inputs = tokenizer(
                prompt, 
                return_tensors="pt" 
            )

            input_ids = inputs['input_ids'].to(model.device)
            prompt_len = input_ids.shape[1]
            log(f"  Input IDs shape: {input_ids.shape} (batch_size=1, seq_len={prompt_len})")

            # prefill phase (GEMM compute-bound)
            outputs = model(
                input_ids=input_ids, 
                use_cache=True
            )

            ## initial KV cache 
            past_key_values = outputs.past_key_values

            ## initial logits 
            logits = outputs.logits 

            log(f"  [Prefill] Logits shape: {logits.shape} (batch_size, seq_len, vocab_size)")
            log(f"  [Prefill] Initial KV cache shape (layer 0, key): {get_kv_cache_shape(past_key_values)} (batch, heads, seq_len, head_dim)")

            # first gen token (we do argmax on the last token logits)
            next_token = torch.argmax(
                logits[:, -1, :], 
                dim=-1, 
                keepdim=True
            )
            
            generated = []

            # decode loop (GEMV memory-bandwidth bound)
            log(f"  --- Decode Phase (max_new_tokens={max_new_tokens}) ---")
            for step in range(max_new_tokens):
                token_id = next_token.item()
                token_str = tokenizer.decode([token_id])

                # EOS check
                if (
                    tokenizer.eos_token_id is not None and 
                    token_id == tokenizer.eos_token_id
                ): 
                    log(f"    Step {step + 1:02d}: token_id={token_id} ({token_str!r}) -> [EOS encountered, stopping early]")
                    break

                generated.append(token_id)
                log(f"    Step {step + 1:02d}: token_id={token_id} ({token_str!r}) | KV shape: {get_kv_cache_shape(past_key_values)}")

                # Feed ONLY the single new token
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                ## new KV cache
                past_key_values = outputs.past_key_values

                ## new logits
                logits = outputs.logits

                # next token (argmax)
                next_token = torch.argmax(
                    logits[:, -1, :], 
                    dim=-1, 
                    keepdim=True
                )

            results.append(generated)
            log(f"  [Prompt {idx + 1} Done] Generated text: {tokenizer.decode(generated)!r} ({len(generated)} tokens)")
                
    return results


def generate_padded_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 32,
    verbose: bool = True
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
    model.eval()
    log = print if verbose else (lambda *args, **kwargs: None)

    # Left-padding is required for causal LM autoregressive batched decoding
    # so that the last token of each prompt aligns at the rightmost index (-1)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("\n" + "=" * 60)
    log(f"=== PADDED BATCH GENERATION ({len(prompts)} prompts) ===")
    log("=" * 60)

    try:
        with torch.no_grad():
            # Batch tokenization with padding
            inputs = tokenizer(
                prompts, 
                padding=True, 
                return_tensors="pt"
            )

            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            batch_size = input_ids.shape[0]
            padded_seq_len = input_ids.shape[1]

            log(f"\n--- [Batch Tokenization] ---")
            log(f"  Batch size: {batch_size}")
            log(f"  Padded input_ids shape: {input_ids.shape} (batch_size={batch_size}, padded_seq_len={padded_seq_len})")
            log(f"  Attention mask shape: {attention_mask.shape}")
            for i, p in enumerate(prompts):
                pad_count = (input_ids[i] == tokenizer.pad_token_id).sum().item()
                log(f"    Prompt {i}: {p!r} (length={padded_seq_len - pad_count}, padding_tokens={pad_count})")

            # Explicit position IDs ensure padding tokens on the left do not shift prompt token position embeddings
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            # Batch Prefill Phase (GEMM: processes all sequences in parallel)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits

            log(f"\n--- [Batch Prefill Phase] ---")
            log(f"  Batch logits shape: {logits.shape} (batch_size, padded_seq_len, vocab_size)")
            log(f"  Initial KV Cache shape (layer 0 key): {get_kv_cache_shape(past_key_values)} (batch, heads, seq_len, head_dim)")

            # Next token for each sequence in batch via argmax on last position
            next_token = torch.argmax(
                logits[:, -1, :], 
                dim=-1, 
                keepdim=True
            )

            # Tracking output and completion status per sequence
            results: List[List[int]] = [[] for _ in range(batch_size)]
            finished = [False] * batch_size

            # Batch Decode Loop (GEMV: 1 new token per sequence per iteration)
            log(f"\n--- [Batch Decode Phase] (max_new_tokens={max_new_tokens}) ---")
            for step in range(max_new_tokens):
                step_tokens = next_token.squeeze(-1).tolist()
                if isinstance(step_tokens, int):
                    step_tokens = [step_tokens]

                # Record tokens and check EOS status per sequence
                for i in range(batch_size):
                    if not finished[i]:
                        tok = step_tokens[i]
                        if tokenizer.eos_token_id is not None and tok == tokenizer.eos_token_id:
                            finished[i] = True
                        else:
                            results[i].append(tok)

                # Diagnostic print for the step
                log(f"  Step {step + 1:02d}/{max_new_tokens} | KV cache shape: {get_kv_cache_shape(past_key_values)}")
                for i in range(batch_size):
                    tok = step_tokens[i]
                    status = " [EOS - stopped]" if finished[i] and (len(results[i]) == 0 or results[i][-1] != tok) else ""
                    log(f"    Seq {i}: token_id={tok} ({tokenizer.decode([tok])!r}){status}")

                # If all sequences in the batch have hit EOS, we can exit early
                if all(finished):
                    log(f"  --> All sequences reached EOS at step {step + 1}. Halting batch generation.")
                    break

                # Position for the new token is the total number of valid previous tokens
                current_position = attention_mask.sum(dim=-1, keepdim=True)

                # Update attention mask for the newly appended token position
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=model.device)],
                    dim=1
                )

                # Feed ONLY the single new token per sequence into model with KV cache and explicit position
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=current_position,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                past_key_values = outputs.past_key_values
                logits = outputs.logits

                # Pick next token for each sequence
                next_token = torch.argmax(
                    logits[:, -1, :], 
                    dim=-1, 
                    keepdim=True
                )

            log(f"\n--- [Batch Generation Summary] ---")
            for i in range(batch_size):
                log(f"  Prompt {i}: {prompts[i]!r}")
                log(f"  Generated text ({len(results[i])} tokens): {tokenizer.decode(results[i])!r}")

    finally:
        # Restore tokenizer padding side to avoid side effects
        tokenizer.padding_side = original_padding_side

    return results


def main() -> None:
    """Demonstrates sequential and padded batch autoregressive generation with a parity check."""
    print("Running Level 0 Naive Generator demonstration...")

    model_id = "hf-internal-testing/tiny-random-gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Model: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        test_prompts = [
            "Hello world",
            "The quick brown fox jumps over",
            "System architecture is"
        ]

        print("\n" + "#" * 60)
        print("# 1. Running Sequential Generation")
        print("#" * 60)
        seq_out = generate_sequential(model, tokenizer, test_prompts, max_new_tokens=8)

        print("\n" + "#" * 60)
        print("# 2. Running Batched Generation")
        print("#" * 60)
        batch_out = generate_padded_batch(model, tokenizer, test_prompts, max_new_tokens=8)

        print("\n" + "#" * 60)
        print("# 3. Parity Check (Sequential vs Batched)")
        print("#" * 60)
        parity = seq_out == batch_out
        print(f"Sequential and Batched outputs match: {parity}")
        if not parity:
            for i in range(len(test_prompts)):
                print(f"  Seq {i}:   {seq_out[i]}")
                print(f"  Batch {i}: {batch_out[i]}")
    except Exception as e:
        print(f"Could not run live model demonstration ({e}).")
        print("Ensure model weights and dependencies are accessible in your environment.")


if __name__ == "__main__":
    main()

