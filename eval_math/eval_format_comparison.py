# -*- coding: utf-8 -*-
"""
Zero-shot evaluation comparing two prompt formats on math benchmarks.

Format 1 (dapo): Verbose system prompt forcing <think>/<answer> tags
Format 2 (qwen): Minimal system prompt with \boxed{} (Qwen2.5-Instruct native format)

Usage:
    python eval_math/eval_format_comparison.py --model_path Qwen/Qwen2.5-1.5B-Instruct
    python eval_math/eval_format_comparison.py --model_path Qwen/Qwen2.5-7B-Instruct
    python eval_math/eval_format_comparison.py --model_path Qwen/Qwen3-4B
    python eval_math/eval_format_comparison.py --model_path meta-llama/Llama-3.2-3B-Instruct
    python eval_math/eval_format_comparison.py --model_path Qwen/Qwen2.5-1.5B-Instruct --datasets test_gsm8k test_math500
"""

import argparse
import json
import os
import re
from datetime import datetime

from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Suppress math_verify timeout warnings
parse.__globals__["TIMEOUT_WARNING_SHOWN"] = True
verify.__globals__["TIMEOUT_WARNING_SHOWN"] = True

# ============================================================
# System prompts
# ============================================================

DAPO_SYSTEM_PROMPT = """You are a mathematical problem solver. Your task is to solve mathematical problems step by step.

## Response Format:
You MUST use this exact format for every response. All tags are REQUIRED in sequential order:

<think>your step-by-step reasoning and solution process</think>
<answer>your final answer</answer>

## Instructions:
1. Carefully read and understand the problem
2. Show your reasoning step by step in the <think> tags
3. Provide your final answer in the <answer> tags
4. For numerical answers, provide the exact value
5. If the problem asks for a specific format (e.g., \\boxed{}), use that format in your answer

## Example:
Problem: "What is the sum of all positive integers less than 100 that are divisible by 3?"

<think>
I need to find all positive integers less than 100 that are divisible by 3, then sum them.

The integers divisible by 3 less than 100 are: 3, 6, 9, ..., 99
This is an arithmetic sequence with:
- First term a₁ = 3
- Common difference d = 3
- Last term aₙ = 99

To find how many terms: aₙ = a₁ + (n-1)d
99 = 3 + (n-1)×3
96 = (n-1)×3
n-1 = 32
n = 33

Sum of arithmetic sequence: S = n(a₁ + aₙ)/2
S = 33(3 + 99)/2
S = 33 × 102/2
S = 33 × 51
S = 1683
</think>
<answer>\\boxed{1683}</answer>

## Notes:
- Be thorough in your reasoning
- Show all important steps
- Double-check your calculations
- Provide the final answer clearly in the <answer> tags"""

QWEN_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# ============================================================
# Dataset configurations
# ============================================================

ALL_DATASETS = {
    "dapo_test": {
        "name": "DAPO-17k",
        "path": "weijiezz/DAPO-Math-17k-split",
        "split": "test",
        "prompt_key": "prompt",  # list of chat messages, extract content
        "answer_key": "reward_model",  # dict with "ground_truth"
    },
    "test_gsm8k": {
        "name": "GSM8K",
        "path": "weijiezz/math-datasets-100k",
        "split": "test_gsm8k",
        "prompt_key": "question",
        "answer_key": "answer",
    },
    "test_math500": {
        "name": "MATH500",
        "path": "weijiezz/math-datasets-100k",
        "split": "test_math500",
        "prompt_key": "question",
        "answer_key": "answer",
    },
    "test_aime24": {
        "name": "AIME2024",
        "path": "weijiezz/math-datasets-100k",
        "split": "test_aime24",
        "prompt_key": "question",
        "answer_key": "answer",
    },
    "test_aime25": {
        "name": "AIME2025",
        "path": "weijiezz/math-datasets-100k",
        "split": "test_aime25",
        "prompt_key": "question",
        "answer_key": "answer",
    },
    "test_amc23": {
        "name": "AMC2023",
        "path": "weijiezz/math-datasets-100k",
        "split": "test_amc23",
        "prompt_key": "question",
        "answer_key": "answer",
    },
    "test_minervamath": {
        "name": "MinervaMath",
        "path": "weijiezz/math-datasets-100k",
        "split": "test_minervamath",
        "prompt_key": "question",
        "answer_key": "answer",
    },
    "test_olympiadbench": {
        "name": "OlympiadBench",
        "path": "weijiezz/math-datasets-100k",
        "split": "test_olympiadbench",
        "prompt_key": "question",
        "answer_key": "answer",
    },
}

# ============================================================
# Answer parsing
# ============================================================


def parse_ground_truth(answer_str: str):
    """Parse ground truth answer string into math_verify format."""
    if not answer_str:
        return None
    # GSM8K format: "... #### 42"
    if "#" in answer_str:
        answer_str = answer_str.split("#")[-1].strip()
    try:
        return parse(answer_str)
    except Exception:
        return None


def parse_dapo_response(response: str):
    """Parse answer from dapo format response: extract <answer> tags first, then math_verify."""
    # Try <answer> tag extraction
    search_ans = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if search_ans:
        response = search_ans.group(1)
    try:
        return parse(response)
    except Exception:
        return None


def parse_qwen_response(response: str):
    """Parse answer from qwen format response: directly use math_verify (handles \\boxed{})."""
    try:
        return parse(response)
    except Exception:
        return None


def verify_answer(gold, pred) -> bool:
    """Verify if predicted answer matches ground truth."""
    if gold is None or pred is None:
        return False
    try:
        return verify(gold, pred)
    except Exception:
        return False


# ============================================================
# Main
# ============================================================


def build_prompts(questions: list[str], system_prompt: str, tokenizer) -> list[str]:
    """Build tokenized prompts with chat template applied."""
    prompts = []
    for q in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def evaluate_format(
    llm: LLM,
    tokenizer,
    questions: list[str],
    answers: list[str],
    system_prompt: str,
    parse_fn,
    sampling_params: SamplingParams,
) -> dict:
    """Evaluate a single format on a dataset. Returns accuracy and details."""
    prompts = build_prompts(questions, system_prompt, tokenizer)

    # Batch generate
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    total = len(questions)
    details = []

    for i, output in enumerate(outputs):
        response_text = output.outputs[0].text
        pred = parse_fn(response_text)
        gold = parse_ground_truth(answers[i])
        is_correct = verify_answer(gold, pred)

        if is_correct:
            correct += 1

        details.append(
            {
                "question": questions[i][:100],
                "ground_truth": answers[i][:100],
                "response": response_text[:200],
                "correct": is_correct,
            }
        )

    accuracy = correct / total * 100 if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Compare prompt formats on math benchmarks")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Dataset splits to evaluate. Default: all. Options: {list(ALL_DATASETS.keys())}",
    )
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max generation tokens")
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=None, help="TP size (default: auto)"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization"
    )
    parser.add_argument("--output_dir", type=str, default="eval_math/results", help="Output dir")
    args = parser.parse_args()

    # Determine datasets
    dataset_splits = args.datasets or list(ALL_DATASETS.keys())
    for ds in dataset_splits:
        if ds not in ALL_DATASETS:
            raise ValueError(f"Unknown dataset: {ds}. Options: {list(ALL_DATASETS.keys())}")

    # Determine TP size
    tp_size = args.tensor_parallel_size or 1

    print(f"Model: {args.model_path}")
    print(f"Datasets: {dataset_splits}")
    print(f"Tensor parallel size: {tp_size}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load vLLM model
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0,  # greedy decoding for reproducibility
        max_tokens=args.max_tokens,
    )

    # Results table
    results = {}

    for ds_key in dataset_splits:
        ds_cfg = ALL_DATASETS[ds_key]
        dataset_name = ds_cfg["name"]
        print(f"{'=' * 60}")
        print(f"Evaluating: {dataset_name} ({ds_key})")
        print(f"{'=' * 60}")

        # Load dataset
        ds = load_dataset(ds_cfg["path"], split=ds_cfg["split"])

        # Extract questions and answers based on dataset format
        questions = []
        answers = []
        for item in ds:
            # Extract question
            raw_prompt = item[ds_cfg["prompt_key"]]
            if isinstance(raw_prompt, list):
                # DAPO format: [{"role": "user", "content": "..."}]
                if len(raw_prompt) > 0 and isinstance(raw_prompt[0], dict):
                    questions.append(raw_prompt[0].get("content", ""))
                else:
                    questions.append(str(raw_prompt))
            else:
                questions.append(str(raw_prompt))

            # Extract answer
            raw_answer = item[ds_cfg["answer_key"]]
            if isinstance(raw_answer, dict):
                # DAPO format: {"ground_truth": "..."}
                answers.append(raw_answer.get("ground_truth", ""))
            else:
                answers.append(str(raw_answer))

        print(f"Loaded {len(questions)} examples")

        # Evaluate dapo format
        print(f"\n[dapo format] Generating {len(questions)} responses...")
        dapo_result = evaluate_format(
            llm, tokenizer, questions, answers, DAPO_SYSTEM_PROMPT, parse_dapo_response, sampling_params
        )
        print(f"[dapo format] Accuracy: {dapo_result['accuracy']:.1f}% ({dapo_result['correct']}/{dapo_result['total']})")

        # Evaluate qwen format
        print(f"\n[qwen format] Generating {len(questions)} responses...")
        qwen_result = evaluate_format(
            llm, tokenizer, questions, answers, QWEN_SYSTEM_PROMPT, parse_qwen_response, sampling_params
        )
        print(f"[qwen format] Accuracy: {qwen_result['accuracy']:.1f}% ({qwen_result['correct']}/{qwen_result['total']})")

        results[dataset_name] = {
            "split": ds_key,
            "n": len(questions),
            "dapo": dapo_result["accuracy"],
            "qwen": qwen_result["accuracy"],
            "dapo_detail": dapo_result,
            "qwen_detail": qwen_result,
        }

        diff = qwen_result["accuracy"] - dapo_result["accuracy"]
        print(f"\nDifference (qwen - dapo): {diff:+.1f}%")
        print()

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_path}")
    print(f"Decoding: greedy (temperature=0)")
    print()
    print(f"| {'Dataset':<20} | {'N':>5} | {'dapo format':>12} | {'qwen format':>12} | {'diff':>8} |")
    print(f"|{'-' * 22}|{'-' * 7}|{'-' * 14}|{'-' * 14}|{'-' * 10}|")
    for name, r in results.items():
        diff = r["qwen"] - r["dapo"]
        print(f"| {name:<20} | {r['n']:>5} | {r['dapo']:>11.1f}% | {r['qwen']:>11.1f}% | {diff:>+7.1f}% |")
    print()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model_path.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"{model_name}_{timestamp}.json")

    save_data = {
        "model": args.model_path,
        "timestamp": timestamp,
        "settings": {"temperature": 0, "max_tokens": args.max_tokens},
        "results": {
            name: {"split": r["split"], "n": r["n"], "dapo_accuracy": r["dapo"], "qwen_accuracy": r["qwen"]}
            for name, r in results.items()
        },
    }
    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
