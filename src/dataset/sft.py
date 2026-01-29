from collections.abc import Callable

import pandas as pd
from datasets import Dataset


def map_problem_description_to_conversation_sft(
    row: dict[str, any],
) -> list[dict[str, any]]:
    """
    Map a problem description to a conversation.

    Args:
        row: The row

    Returns:
        list[dict[str, any]]: The conversation
    """
    system_prompt = """
You are an expert mathematician specializing in arithmetic countdown problems. Your task is to find arithmetic expressions using exactly four given numbers and basic operators (+, -, *, /) to reach a target result.

**Your approach must be:**
1. Use **a single <think> block** to show your systematic reasoning process
2. Consider different combinations of numbers and operators
3. Apply proper order of operations (multiplication and division before addition and subtraction)
4. Verify your calculations step by step
5. Provide your final arithmetic expression in the <answer> block
6. There should ONLY be ONE <answer> block containing only the arithmetic expression.

**Rules:**
- Use each of the four given numbers exactly once
- Only use operators: +, -, *, / (use '*' for multiplication)
- The expression must equal the target result exactly
- Show clear mathematical reasoning in your thinking
- Your final answer must be a valid arithmetic expression

**Format:**
<think>
Analyze the numbers and target result, try different combinations and operations, calculate and verify results step by step.
</think>
<answer>
(Your arithmetic expression, e.g., "3 + 7 * 2 - 1")
</answer>

Example:
<think>
(Analyze the numbers and target result, try different combinations and operations, calculate and verify results step by step.)
</think>
<answer>
3 + 7 * 2 - 1
</answer>

There should ONLY be ONE <answer> block containing only the arithmetic expression.
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["problem_description"]},
        {"role": "assistant", "content": row["reasoning"]},
    ]


def load_csv_dataset_sft(file_path: str, mapping_function: Callable) -> Dataset:
    """
    Load a CSV dataset.
    """
    dataset = pd.read_csv(file_path)
    dataset["messages"] = dataset.apply(mapping_function, axis=1)
    return Dataset.from_pandas(dataset)
