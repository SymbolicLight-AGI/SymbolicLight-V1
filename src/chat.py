#!/usr/bin/env python3
"""Interactive chat loop for SymbolicLight checkpoints."""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from eval_08 import (
    DEFAULT_CHECKPOINT,
    _SL_TOKENIZER_PATH,
    TokenizerWrapper,
    _resolve_load_dtype,
    build_config_from_checkpoint,
    build_model_from_checkpoint,
    build_model_from_checkpoint_zip,
    load_checkpoint,
    load_checkpoint_metadata,
)


class _ArgsShim:
    seq_len = None


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat for SymbolicLight checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to .pt checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=_SL_TOKENIZER_PATH,
                        help="Path to tokenizer model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Execution device")
    parser.add_argument("--allow_windows_cuda", action="store_true",
                        help="Allow CUDA transfer on Windows")
    parser.add_argument("--load_dtype", type=str, default="auto", choices=["auto", "fp32", "fp16"],
                        help="Weight dtype during loading")
    parser.add_argument("--max_new_tokens", type=int, default=96,
                        help="Maximum new tokens per reply")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Top-k sampling cutoff")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling cutoff")
    parser.add_argument("--repetition_penalty", type=float, default=1.15,
                        help="Penalty for tokens that already appeared in the reply")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                        help="Block repeated n-grams in generated text")
    parser.add_argument("--history_turns", type=int, default=0,
                        help="Number of recent turns to keep in the prompt")
    parser.add_argument("--prompt_format", type=str, default="answer",
                        choices=["raw", "qa", "chat", "answer"],
                        help="Prompt template style")
    parser.add_argument("--no_adaptive_temperature", action="store_true",
                        help="Disable entropy-based adaptive temperature")
    return parser.parse_args()


def resolve_device(args):
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_runtime(args):
    device = resolve_device(args)
    load_dtype = _resolve_load_dtype(args.load_dtype)

    if os.name == "nt" and device.type == "cuda" and not args.allow_windows_cuda:
        raise RuntimeError(
            "Windows CUDA loading is disabled by default for this checkpoint. "
            "Re-run with --allow_windows_cuda to enable the GPU path."
        )

    if os.name == "nt":
        ckpt_meta = load_checkpoint_metadata(args.checkpoint_path)
        config, global_step = build_config_from_checkpoint(ckpt_meta, _ArgsShim())
        model, load_info = build_model_from_checkpoint_zip(
            config,
            args.checkpoint_path,
            ckpt_meta,
            target_float_dtype=load_dtype,
        )
    else:
        ckpt = load_checkpoint(args.checkpoint_path, device="cpu")
        config, global_step = build_config_from_checkpoint(ckpt, _ArgsShim())
        model, load_info = build_model_from_checkpoint(config, ckpt)

    model = model.to(device)
    model.eval()
    tokenizer = TokenizerWrapper(args.tokenizer_path)
    return device, config, global_step, model, tokenizer, load_info


def trim_history(history, keep_turns):
    if keep_turns <= 0:
        return []
    return history[-keep_turns:]


def build_prompt(history, user_text, prompt_format):
    if prompt_format == "answer":
        lines = [
            "Answer the question below in short, natural, coherent English.",
            "Do not repeat yourself, do not produce outlines, and do not drift into unrelated textbook content.",
        ]
        for old_user, old_assistant in history:
            lines.append(f"Question: {old_user}")
            lines.append(f"Short answer: {old_assistant}")
        lines.append(f"Question: {user_text}")
        lines.append("Short answer:")
        return "\n".join(lines)

    if prompt_format == "raw":
        if not history:
            return user_text
        parts = []
        for old_user, old_assistant in history:
            parts.append(old_user)
            parts.append(old_assistant)
        parts.append(user_text)
        return "\n".join(parts)

    if prompt_format == "qa":
        lines = []
        for old_user, old_assistant in history:
            lines.append(f"Question: {old_user}")
            lines.append(f"Answer: {old_assistant}")
        lines.append(f"Question: {user_text}")
        lines.append("Answer:")
        return "\n".join(lines)

    lines = []
    for old_user, old_assistant in history:
        lines.append(f"User: {old_user}")
        lines.append(f"Assistant: {old_assistant}")
    lines.append(f"User: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def clean_reply(text, prompt_format):
    reply = text.strip()
    if prompt_format == "answer":
        for marker in ["Question:", "Short answer:", "User:", "Assistant:", "###", "Answer:"]:
            reply = reply.split(marker)[0].strip()
    elif prompt_format == "qa":
        reply = reply.split("Question:")[0].strip()
    elif prompt_format == "chat":
        reply = reply.split("User:")[0].strip()
    reply = dedupe_lines(reply)
    reply = strip_leaked_instructions(reply)
    reply = trim_to_sentences(reply, max_sentences=2)
    return reply.strip()


def dedupe_lines(text):
    seen = set()
    cleaned = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        cleaned.append(line)
    return "\n".join(cleaned)


def trim_to_sentences(text, max_sentences=2):
    if not text:
        return text
    out = []
    sentence_count = 0
    for ch in text:
        out.append(ch)
        if ch in "。！？!?":
            sentence_count += 1
            if sentence_count >= max_sentences:
                break
    trimmed = "".join(out).strip()
    if trimmed:
        return trimmed
    return text.strip()


def strip_leaked_instructions(text):
    cleaned = text
    leaked_phrases = [
        "Answer the question below in short, natural, coherent English.",
        "Do not repeat yourself, do not produce outlines, and do not drift into unrelated textbook content.",
        "Coherent answer:",
        "Short answer:",
    ]
    for phrase in leaked_phrases:
        cleaned = cleaned.replace(phrase, "").strip()
    return cleaned


def canned_reply(user_text):
    normalized = user_text.strip().lower()
    if normalized in {"你好", "您好", "hi", "hello", "hey"}:
        return "Hello. You can ask a specific question and I will try to answer briefly."
    if normalized in {"你是谁", "你是谁？", "请介绍一下你自己", "你好，请介绍一下你自己"}:
        return "I am the local SymbolicLight demo script. The loaded weights are a pre-training checkpoint, so replies can run end to end but may not always be stable."
    if normalized in {"你在说什么", "你在说什么？", "are you ok?", "are you ok"}:
        return "The previous reply was unstable. This checkpoint is still a pre-training artifact, not a polished dialogue model. Try a shorter and more specific question."
    if normalized in {"说中文", "请说中文"}:
        return "I can try, but this public demo is configured primarily for brief English replies."
    return None


def apply_repetition_penalty(logits, token_ids, penalty):
    if penalty <= 1.0 or not token_ids:
        return logits
    unique_ids = set(token_ids)
    for token_id in unique_ids:
        value = logits[0, token_id]
        logits[0, token_id] = value / penalty if value > 0 else value * penalty
    return logits


def calc_banned_tokens(generated_ids, ngram_size):
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return set()
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned = set()
    for i in range(len(generated_ids) - ngram_size + 1):
        if tuple(generated_ids[i:i + ngram_size - 1]) == prefix:
            banned.add(generated_ids[i + ngram_size - 1])
    return banned


def sample_next_token(logits, top_k, top_p):
    if top_k > 0:
        top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = top_k_vals[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        removal_mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
        removal_mask.scatter_(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(removal_mask, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_reply(model, tokenizer, device, prompt, args):
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = input_ids.clone()
    generated_reply_ids = []
    past_key_values = [{} for _ in range(len(model.blocks) + 1)]
    eos_id = tokenizer.eos_id()

    logits = model.forward(input_ids, use_cache=True, past_key_values=past_key_values)

    for _ in range(args.max_new_tokens):
        next_logits = logits[:, -1, :] / max(args.temperature, 1e-5)
        next_logits = apply_repetition_penalty(next_logits, generated_reply_ids, args.repetition_penalty)

        banned_tokens = calc_banned_tokens(generated_reply_ids, args.no_repeat_ngram_size)
        if banned_tokens:
            banned_tensor = torch.tensor(sorted(banned_tokens), device=next_logits.device, dtype=torch.long)
            next_logits.index_fill_(1, banned_tensor, float("-inf"))

        next_token = sample_next_token(next_logits, args.top_k, args.top_p)
        token_id = next_token.item()
        if token_id == eos_id:
            break

        generated = torch.cat([generated, next_token], dim=1)
        generated_reply_ids.append(token_id)

        partial_text = tokenizer.decode(generated_reply_ids)
        if any(stop in partial_text for stop in ["Question:", "Short answer:", "User:", "Assistant:", "###", "Answer:"]):
            break

        logits = model.forward(next_token, use_cache=True, past_key_values=past_key_values)

    return clean_reply(tokenizer.decode(generated_reply_ids), args.prompt_format)


def main():
    args = parse_args()
    device, config, global_step, model, tokenizer, load_info = load_runtime(args)

    print("=" * 60)
    print(" SymbolicLight Interactive Chat")
    print("=" * 60)
    print(f"Checkpoint: {Path(args.checkpoint_path).resolve()}")
    print(f"Tokenizer:  {Path(args.tokenizer_path).resolve()}")
    print(f"Device:     {device}")
    print(f"Step:       {global_step}")
    print(f"Seq len:    {config.max_seq_len}")
    print(f"Format:     {args.prompt_format}")
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        print(f"Missing keys:    {len(load_info['missing_keys'])}")
        print(f"Unexpected keys: {len(load_info['unexpected_keys'])}")
    print("Type 'exit' or 'quit' to stop.")
    print()

    history = []
    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exit]")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("[Exit]")
            break

        fallback = canned_reply(user_text)
        if fallback is not None:
            print(f"Model> {fallback}\n")
            history.append((user_text, fallback))
            continue

        prompt = build_prompt(trim_history(history, args.history_turns), user_text, args.prompt_format)
        with torch.no_grad():
            reply = generate_reply(model, tokenizer, device, prompt, args)
        print(f"Model> {reply}\n")
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
