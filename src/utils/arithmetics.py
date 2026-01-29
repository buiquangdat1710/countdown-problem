import random
import re
from dataclasses import dataclass
from enum import Enum


def check_valid_arithmetic_expression(expression: str, result: int) -> bool:
    """
    Check if a string is a valid arithmetic expression.

    With format

    num1 op1 num2 op2 num3 op3 num4

    with operators +, -, *, /

    Args:
        expression: The expression to check

    Returns:
        bool: True if valid arithmetic expression, False otherwise
    """
    # Regex pattern for: number operator number operator number operator number
    # Number can only be positive integer (no negative numbers)
    # Operators are +, -, *, /
    # Spacing between numbers and operators is optional
    pattern = (
        r"^\s*(\d+)\s*([+\-*/])\s*(\d+)\s*([+\-*/])\s*(\d+)\s*([+\-*/])\s*(\d+)\s*$"
    )

    if bool(re.match(pattern, expression)):
        return eval(expression) == result
    return False


@dataclass
class ArithmeticProblem:
    num_1: int
    num_2: int
    num_3: int
    num_4: int
    op1: str
    op2: str
    op3: str
    expression: str
    result: int


class Mode(Enum):
    ALL = "all"  # All operators are allowed
    MUL_DIV = "mul_div"  # Only multiplication and division are allowed


class ArithmeticProblemGenerator:
    def __init__(
        self,
        min_num: int = 1,
        max_num: int = 100,
        result_min: int = 1,
        result_max: int = 1000,
        max_attempts: int = 100,
        operators: tuple[str] = ("+", "-", "*", "/"),
        mode: Mode = Mode.ALL,
    ):
        """
        Initialize the arithmetic problem generator.

        Args:
            min_num: The minimum number to use in the arithmetic problem
            max_num: The maximum number to use in the arithmetic problem
            operators: The operators to use in the arithmetic problem
            mode: The mode of the arithmetic problem
        """
        self.min_num = min_num
        self.max_num = max_num
        self.result_min = result_min
        self.result_max = result_max
        self.operators = operators
        self.max_attempts = max_attempts
        self.mode = mode

    def _generate_random_number(self) -> int:
        return random.randint(self.min_num, self.max_num)

    def _generate_random_operator(self) -> str:
        return random.choice(self.operators)

    def generate_problem(self) -> ArithmeticProblem:
        """
        Generate an countdown arithmetic problem.

        Generate four numbers, num_1, num_2, num_3, num_4,
        and operators between them, and apply the operators to the numbers to get the result.

        Make sure that the result must exactly be an integer, and
        match the result of the arithmetic problem.

        Returns:
            ArithmeticProblem: The generated arithmetic problem
        """
        max_attempts = 1_000

        for _ in range(max_attempts):
            # Generate four random numbers
            num_1 = self._generate_random_number()
            num_2 = self._generate_random_number()
            num_3 = self._generate_random_number()
            num_4 = self._generate_random_number()

            # Generate three random operators for the expression: num_1 op1 num_2 op2 num_3 op3 num_4
            op1 = self._generate_random_operator()
            op2 = self._generate_random_operator()
            op3 = self._generate_random_operator()

            if (
                self.mode == Mode.MUL_DIV
                and op1 not in ("*", "/")
                and op2 not in ("*", "/")
                and op3 not in ("*", "/")
            ):
                continue

            # Try to evaluate the expression and ensure it results in an integer
            result = self._evaluate_expression(
                num_1, op1, num_2, op2, num_3, op3, num_4
            )

            # Check if result is an integer (no floating point remainder)
            if (
                isinstance(result, (int, float))
                and result == int(result)
                and self.result_min <= result <= self.result_max
            ):
                result = int(result)
                return ArithmeticProblem(
                    num_1=num_1,
                    num_2=num_2,
                    num_3=num_3,
                    num_4=num_4,
                    op1=op1,
                    op2=op2,
                    op3=op3,
                    expression=f"{num_1} {op1} {num_2} {op2} {num_3} {op3} {num_4}",
                    result=result,
                )

        return None

    def _evaluate_expression(
        self,
        num_1: int,
        op1: str,
        num_2: int,
        op2: str,
        num_3: int,
        op3: str,
        num_4: int,
    ) -> float:
        """
        Evaluate the arithmetic expression following standard order of operations.

        Args:
            num_1: First number
            op1: First operator
            num_2: Second number
            op2: Second operator
            num_3: Third number
            op3: Third operator
            num_4: Fourth number

        Returns:
            float: The result of the arithmetic expression
        """
        # Build expression string: num_1 op1 num_2 op2 num_3 op3 num_4
        expression = f"{num_1} {op1} {num_2} {op2} {num_3} {op3} {num_4}"

        # Use eval to calculate the result (following Python's order of operations)
        # This handles operator precedence correctly (* and / before + and -)
        return eval(expression)


class ArithmeticProblemDescriptionGenerator:
    def __init__(self):
        """Initialize the description generator with various problem templates."""
        self.problem_templates = [
            # Direct challenge templates
            "Using the numbers {num_1}, {num_2}, {num_3}, and {num_4}, create an expression that equals {result}. You can only use +, -, x, and / operators.",
            "Can you make {result} using {num_1}, {num_2}, {num_3}, and {num_4}? Use only +, -, x, and / operators.",
            "Find a way to combine {num_1}, {num_2}, {num_3}, and {num_4} to get {result} using only +, -, x, and / operators.",
            "Use all four numbers ({num_1}, {num_2}, {num_3}, {num_4}) to make {result}. Only +, -, x, and / operators are allowed.",
            # Instructional templates
            "Given the numbers {num_1}, {num_2}, {num_3}, and {num_4}, arrange them with +, -, x, and / operators to achieve {result}.",
            "Your task: Use {num_1}, {num_2}, {num_3}, and {num_4} exactly once each with only +, -, x, and / operators to create an expression equal to {result}.",
            "Problem: How can you use the four numbers {num_1}, {num_2}, {num_3}, {num_4} with +, -, x, and / operators to get {result}?",
            # Additional templates with operator emphasis
            "Create a mathematical expression using {num_1}, {num_2}, {num_3}, and {num_4} that equals {result}. Only basic arithmetic operators (+, -, x, /) are permitted.",
            "Arrange {num_1}, {num_2}, {num_3}, and {num_4} with +, -, x, and / to make {result}.",
            "Using only addition (+), subtraction (-), multiplication (x), and division (/), combine {num_1}, {num_2}, {num_3}, and {num_4} to equal {result}.",
        ]

    def generate_description(self, problem: ArithmeticProblem) -> tuple[str, int]:
        """
        Generate a problem description for the given arithmetic problem.

        Args:
            problem: The ArithmeticProblem to generate description for

        Returns:
            tuple[str, int]: A tuple containing (problem_description, result)
        """
        # Select random template
        problem_template = random.choice(self.problem_templates)

        # Generate the problem description
        problem_description = problem_template.format(
            num_1=problem.num_1,
            num_2=problem.num_2,
            num_3=problem.num_3,
            num_4=problem.num_4,
            result=problem.result,
        )

        return problem_description, problem.result
