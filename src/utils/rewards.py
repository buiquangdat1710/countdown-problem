"""
Collection of various reward signal for the arithmetic problem.
"""

import logging
import re

from src.utils.string_helper import (
    extract_answers_from_completions,
    extract_response_from_completions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rewards")


def _is_valid_arithmetic_expression(expression: str) -> bool:
    """
    Check if a string is a valid arithmetic expression containing only:
    - Numbers (integers only)
    - Arithmetic operators: +, -, x, /
    - Whitespace

    Args:
        expression: The expression to validate

    Returns:
        bool: True if valid arithmetic expression, False otherwise
    """
    if not expression or not expression.strip():
        return False

    # Pattern that matches valid arithmetic expressions
    # Allows: integers, operators (+, -, x, /), and whitespace (no decimal points)
    pattern = r"^[\d\s\+\-x\/]+$"

    # First check: contains only allowed characters
    if not re.match(pattern, expression):
        return False

    # Second check: must contain at least one number and one operator
    has_number = re.search(r"\d", expression)
    has_operator = re.search(r"[\+\-x\/]", expression)

    if not (has_number and has_operator):
        return False

    # Third check: basic structure validation (no consecutive operators)
    try:
        # Replace x with * for evaluation
        normalized = expression.replace("x", "*")

        # Remove whitespace for easier validation
        normalized = "".join(normalized.split())

        # Check for consecutive operators
        if re.search(r"[\+\-\*\/]{2,}", normalized):
            return False

        # Try to evaluate the expression to ensure it's syntactically correct
        # This will catch issues like "1 + + 2" or "3 * / 4"
        eval(normalized)
        return True

    except (SyntaxError, ValueError):
        # If eval fails due to syntax errors, it's not a valid expression
        return False
    except ZeroDivisionError:
        # ZeroDivisionError is okay (the expression is valid, just results in division by zero)
        return True
    except:
        # Any other error means invalid expression
        return False


def _calculate_distance_based_reward(answer: str, correct_answer: int) -> float:
    """
    Calculate reward based on distance from the correct answer.

    Uses linear scaling: reward = max(0, max_reward - (distance * penalty_per_unit))

    Args:
        answer: The arithmetic expression to evaluate
        correct_answer: The expected result

    Returns:
        float: Reward between 0.0 and 2.0 based on distance from correct answer
    """
    if not answer or not answer.strip():
        return 0.0

    # First check if it's a valid arithmetic expression
    if not _is_valid_arithmetic_expression(answer):
        return 0.0

    try:
        # Replace x with * for evaluation
        normalized = answer.replace("x", "*")

        # Evaluate the expression
        result = eval(normalized)

        # Calculate distance from correct answer
        if isinstance(result, (int, float)):
            distance = abs(result - correct_answer)

            # Perfect match gets full reward
            if distance < 0.0001:
                return 2.0

            # Linear scaling reward function
            # Parameters can be tuned:
            # - max_reward: maximum possible reward (2.0)
            # - penalty_per_unit: how much reward decreases per unit of distance
            max_reward = 2.0
            penalty_per_unit = 0.2  # Lose 0.2 points per unit of distance

            reward = max_reward - (distance * penalty_per_unit)

            # Ensure minimum reward is 0.0
            return max(0.0, reward)

        return 0.0

    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        # If evaluation fails, no reward
        return 0.0
    except:
        # Any other error means no reward
        return 0.0


def format_reward_functiondef(
    completions: list[list[dict[str, str]]], **kwargs: dict[str, any]
) -> list[float]:
    """
    Reward function that checks if a completion contains <think>...</think> and
    <answer>...</answer> sections.

    Args:
        completions: List of completions of the format:
        [
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        ]

    Returns:
        List of rewards.
    """
    pattern = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", re.DOTALL)
    responses = extract_response_from_completions(completions)
    matches = [bool(pattern.search(response)) for response in responses]
    return [1.0 if match else 0.0 for match in matches]


def arithmetic_format_reward_function(
    completions: list[list[dict[str, str]]],
    **kwargs: dict[str, any],
) -> list[float]:
    """
    Reward function that checks if the content of the answer tag is a valid arithmetic expression.

    The answer should contain only numbers, arithmetic operators (+, -, x, /),
    and spaces. Examples of valid formats:
    - "1 + 2 x 6 / 3"
    - "2 x 1 + 3 - 1"
    - "4 + 5 x 2 - 1"

    Args:
        completions: List of completions of the format:
        [
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        ]

    Returns:
        List of rewards (1.0 for valid arithmetic expressions, 0.0 otherwise).
    """
    # Extract responses from the completions
    answers = extract_answers_from_completions(completions)

    return [
        1.0 if _is_valid_arithmetic_expression(answer) else 0.0 for answer in answers
    ]


def correctness_reward_function(
    completions: list[list[dict[str, str]]], **kwargs: dict[str, any]
) -> list[float]:
    """
    Reward function that provides rewards based on how close the arithmetic answer is to the correct result.

    The reward is calculated using linear scaling:
    - Perfect match (distance = 0): reward = 2.0
    - Each unit of distance reduces reward by 0.2 points
    - Minimum reward is 0.0
    - Invalid expressions get 0.0

    Args:
        completions: List of completions of the format:
        [
            [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]
        ]
        **kwargs: Must contain 'correct_answer' key with the expected result

    Returns:
        List of rewards (0.0 to 2.0 based on distance from correct answer).

    Raises:
        ValueError: If the correct answer is not provided in the kwargs.
    """
    # Get the correct answer from the kwargs
    correct_answer = kwargs["correct_answer"]

    # Get the answer from the completions
    answers = extract_answers_from_completions(completions)
    completions = [completion[-1]["content"] for completion in completions]

    # Display first question and answer
    logger.info("First question: %s", completions[0])
    logger.info("First answer: %s", answers[0])

    return [
        _calculate_distance_based_reward(answer, correct_answer) for answer in answers
    ]


def mathematical_correctness_reward_function(
    completions: list[str], **kwargs
) -> list[float]:
    """
    Evaluates completions based on Mathematical correctness of the answer

    Args:
        completions: Generated outputs
        target: Expected answers
        **kwargs: Additional keyword arguments

    Returns:
        list[float]: Reward scores (1.0 for correct, 0.0 for incorrect)
    """
    completions = [completion[-1]["content"] for completion in completions]
    target = kwargs["correct_answer"]
    first_nums = kwargs["num1"]
    second_nums = kwargs["num2"]
    third_nums = kwargs["num3"]
    fourth_nums = kwargs["num4"]
    rewards = []

    # Display completions
    logger.info("Completion:\n%s", completions[0])

    for completion, gt, first_num, second_num, third_num, fourth_num in zip(
        completions,
        target,
        first_nums,
        second_nums,
        third_nums,
        fourth_nums,
        strict=False,
    ):
        reward = 0.0
        try:
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
            if match is None:
                logger.info(
                    "┌─────────────────────────────────────────────────────────────────────┐"
                )
                logger.info(
                    "│ ❌ FORMAT ERROR: No <answer> tags found in completion              │"
                )
                logger.info(
                    "├─────────────────────────────────────────────────────────────────────┤"
                )
                logger.info(
                    "│ Completion snippet: %-47s │",
                    completion[:47] + "..." if len(completion) > 47 else completion,
                )
                logger.info(
                    "└─────────────────────────────────────────────────────────────────────┘"
                )
                rewards.append(reward)
                continue

            # Add reward
            reward += 1.0

            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            if "=" in equation:
                equation = equation.split("=")[0]

            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # Check if all numbers are used exactly once
            correct_numbers = [first_num, second_num, third_num, fourth_num]
            if sorted(used_numbers) != sorted(correct_numbers):
                logger.info(
                    "┌─────────────────────────────────────────────────────────────────────┐"
                )
                logger.info(
                    "│ ❌ NUMBER USAGE ERROR: Incorrect numbers used                      │"
                )
                logger.info(
                    "├─────────────────────────────────────────────────────────────────────┤"
                )
                logger.info("│ Equation: %-57s │", equation[:57])
                logger.info("│ Expected numbers: %-51s │", str(correct_numbers))
                logger.info("│ Used numbers: %-55s │", str(used_numbers))
                logger.info(
                    "└─────────────────────────────────────────────────────────────────────┘"
                )
                rewards.append(reward)
                continue

            # Add reward
            reward += 1.0

            # Define a regex pattern that only allows numbers, operators, and whitespace
            allowed_pattern = r"^[\d+\-*/.\s]+$"
            if not re.match(allowed_pattern, equation):
                logger.info(
                    "┌─────────────────────────────────────────────────────────────────────┐"
                )
                logger.info(
                    "│ ❌ INVALID CHARACTERS: Equation contains disallowed characters     │"
                )
                logger.info(
                    "├─────────────────────────────────────────────────────────────────────┤"
                )
                logger.info("│ Equation: %-57s │", equation[:57])
                logger.info(
                    "└─────────────────────────────────────────────────────────────────────┘"
                )
                rewards.append(reward)
                continue

            # Add reward
            reward += 1.0

            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builtins__": None}, {})

            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(gt)) < 1e-5:
                logger.info(
                    "┌─────────────────────────────────────────────────────────────────────┐"
                )
                logger.info(
                    "│ ✅ CORRECT ANSWER: Perfect match!                                  │"
                )
                logger.info(
                    "├─────────────────────────────────────────────────────────────────────┤"
                )
                logger.info(
                    "│ Equation: %-35s = %-20s │", equation[:35], str(result)[:20]
                )
                logger.info("│ Target: %-59s │", str(gt))
                logger.info(
                    "└─────────────────────────────────────────────────────────────────────┘"
                )
                reward += 4.0
                rewards.append(reward)
            else:
                logger.info(
                    "┌─────────────────────────────────────────────────────────────────────┐"
                )
                logger.info(
                    "│ ❌ WRONG RESULT: Equation evaluated to incorrect value             │"
                )
                logger.info(
                    "├─────────────────────────────────────────────────────────────────────┤"
                )
                logger.info(
                    "│ Equation: %-35s = %-20s │", equation[:35], str(result)[:20]
                )
                logger.info("│ Expected: %-57s │", str(gt))
                logger.info(
                    "│ Difference: %-55s │", str(abs(float(result) - float(gt)))
                )
                logger.info(
                    "└─────────────────────────────────────────────────────────────────────┘"
                )
                rewards.append(reward)
        except Exception as e:
            # If evaluation fails, reward is 0
            logger.info(
                "┌─────────────────────────────────────────────────────────────────────┐"
            )
            logger.info(
                "│ ❌ EVALUATION ERROR: Exception occurred during processing           │"
            )
            logger.info(
                "├─────────────────────────────────────────────────────────────────────┤"
            )
            logger.info("│ Error: %-61s │", str(e)[:61])
            logger.info(
                "│ Equation: %-57s │",
                (equation if "equation" in locals() else "N/A")[:57],
            )
            logger.info(
                "└─────────────────────────────────────────────────────────────────────┘"
            )
            rewards.append(reward)
    return rewards
