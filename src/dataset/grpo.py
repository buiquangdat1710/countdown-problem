from collections.abc import Callable

from datasets import Dataset, load_dataset
from openai import OpenAI


def map_problem_description_to_conversation_grpo(
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
Analyze the numbers and target result, try different combinations and operations, calculate and verify results step by step.
</think>
<answer>
3 + 7 * 2 - 1
</answer>

There should ONLY be ONE <answer> block containing only the arithmetic expression.
"""
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["problem_description"]},
        ]
    }





def get_reasoning_for_answer(problem_description: str, correct_expression: str) -> str:
    """
    Generate mathematically accurate reasoning for a countdown puzzle,
    based strictly on the known correct arithmetic expression.
    """

    client = OpenAI()

    system_prompt = f"""You are an expert math tutor generating dataset-quality reasoning.

Your job is to produce a clean, precise explanation for a countdown arithmetic problem.
You have already been given the **correct expression**, so you must NOT try alternative paths, test other combinations, or guess.
Just clearly explain WHY this expression works and HOW it reaches the target.

### REQUIRED BEHAVIOR
- Use EXACTLY ONE <think> block.
- Explain the expression logically, step-by-step.
- Explain how each operation transforms the intermediate total.
- Ensure all numbers in the expression are referenced.
- No incorrect attempts, no searching, no "trying" alternative expressions.
- No contradictions or speculative reasoning.
- No extra expressions outside the final one.
- Length: 80–140 words (optimal for SFT).

### REQUIRED FORMAT
<think>
Explain the correct_expression step-by-step and show that it equals the target.
</think>
<answer>
{correct_expression}
</answer>

### DO NOT:
- invent other expressions  
- explore failed attempts  
- discuss permutations  
- contradict the final expression  
- add ANYTHING outside the two blocks
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_description},
        ],
        max_output_tokens=512,
    )

    return response.output_text.strip()




def load_csv_dataset_grpo(
    file_path: str, split: str, mapping_function: Callable
) -> Dataset:
    """
    Load a CSV dataset.

    Args:
        file_path: Path to the CSV file
        mapping_function: Function to map the dataset
        split: Split of the dataset

    Returns:
        Dataset: The loaded dataset
    """
    dataset = load_dataset("csv", data_files=file_path, split=split)
    dataset = dataset.map(mapping_function)
    return dataset
