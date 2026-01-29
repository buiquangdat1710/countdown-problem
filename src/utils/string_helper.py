import re


def extract_response_from_completions(
    completions: list[list[dict[str, str]]],
) -> list[str]:
    """
    Extract the response from the completions.

    Args:
        completions: The completions to extract the response from

    Returns:
        The response from the completions
    """
    return [completion[0]["content"] for completion in completions]


def extract_answer(completion: str) -> str:
    """
    Extract the answer from the completion.

    Args:
        completion: The completion to extract the answer from

    Returns:
        The answer from the completion
    """
    result = re.search(
        r"<answer>\s*(.*?)\s*</answer>", completion, re.DOTALL | re.IGNORECASE
    )
    if result is None:
        return completion.strip()
    return result.group(1).strip()


def extract_answers_from_completions(
    completions: list[list[dict[str, str]]],
) -> list[str]:
    """
    Extract the answers from the completions.

    Args:
        completions: The completions to extract the answers from

    Returns:
        The answers from the completions
    """
    return [extract_answer(completion[-1]["content"]) for completion in completions]
