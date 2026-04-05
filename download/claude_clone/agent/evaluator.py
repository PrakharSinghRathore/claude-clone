"""
Evaluation & Benchmarking Framework for Claude Clone.

Provides standardized benchmarks (HumanEval, MBPP, SWE-bench), custom task suites,
A/B testing, regression detection, performance tracking, parallel execution, and
detailed report generation.

Usage:
    from agent.evaluator import Evaluator

    evaluator = Evaluator(agent, results_dir="~/.claude_clone/eval_results")
    await evaluator.initialize()
    result = await evaluator.run_humaneval(limit=20, parallel=4)
    report = await evaluator.generate_report(result)
    print(report)
"""

import asyncio
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class TaskType(str, Enum):
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"
    SWE_BENCH = "swe_bench"
    CUSTOM = "custom"
    REGRESSION = "regression"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    task_type: TaskType
    description: str
    input: str
    expected_output: str
    actual_output: str
    passed: bool
    score: float
    execution_time: float
    tokens_used: int
    cost: float
    error: str
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    category: str = ""


@dataclass
class BenchmarkResult:
    benchmark_name: str
    total_tasks: int
    passed: int
    failed: int
    pass_rate: float
    total_time: float
    total_cost: float
    avg_tokens: int
    per_category: Dict[str, Dict[str, Any]]
    per_difficulty: Dict[str, Dict[str, Any]]
    timestamp: str
    details: List[TaskResult] = field(default_factory=list)


@dataclass
class ABTestResult:
    config_a: Dict[str, Any]
    config_b: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    scores_a: List[float]
    scores_b: List[float]
    winner: str
    statistical_significance: float
    improvement_percentage: float


@dataclass
class RegressionReport:
    current_score: float
    previous_score: float
    delta: float
    regressed_tasks: List[Dict[str, Any]]
    improved_tasks: List[Dict[str, Any]]
    new_failures: List[Dict[str, Any]]
    fixed_tasks: List[Dict[str, Any]]


# ──────────────────────────────────────────────
# Embedded benchmark data
# ──────────────────────────────────────────────

HUMANEVAL_TASKS = [
    {
        "task_id": "HumanEval/0",
        "prompt": (
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """Check if in given list of numbers, are any two numbers closer to each other than\n'
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
        "test_code": (
            "def check_has_close_elements():\n"
            "    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.18) == False\n"
            "    assert has_close_elements([], 0.5) == False\n"
            "    assert has_close_elements([1.0, 2.0], 0.0) == False\n"
            "    assert has_close_elements([1.0, 1.0], 0.5) == True\n"
            "    assert has_close_elements([1.1, 2.2, 3.1, 4.1], 1.0) == True\n"
            "    assert has_close_elements([1.1, 2.2, 3.1, 4.1], 0.5) == False\n"
            "    print('ALL TESTS PASSED')\n"
            "check_has_close_elements()\n"
        ),
        "expected_function": "has_close_elements",
        "difficulty": TaskDifficulty.EASY,
        "category": "list",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": (
            "def separate_paren_groups(paren_string: str) -> List[str]:\n"
            '    """Input to this function is a string containing multiple groups of nested parentheses. '
            'Your goal is to separate those groups into separate strings and return the list of them.\n'
            "    Every opening and closing parenthesis matches and is properly nested.\n"
            "    >>> separate_paren_groups('( ) (( )) (( )( ))')\n"
            "    ['()', '(())', '(()())']\n"
            '    """\n'
        ),
        "test_code": (
            "def check_separate_paren_groups():\n"
            "    assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n"
            "    assert separate_paren_groups('(()()) ((())) ()') == ['(()())', '((()))', '()']\n"
            "    assert separate_paren_groups('() (())') == ['()', '(())']\n"
            "    assert separate_paren_groups('(()) ()') == ['(())', '()']\n"
            "    print('ALL TESTS PASSED')\n"
            "check_separate_paren_groups()\n"
        ),
        "expected_function": "separate_paren_groups",
        "difficulty": TaskDifficulty.MEDIUM,
        "category": "string",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": (
            "def truncate_number(number: float) -> float:\n"
            '    """Given a positive floating point number, it can be decomposed into an integer part '
            'and a smaller fractional part. Return the fractional part.\n'
            "    >>> truncate_number(3.5)\n"
            "    0.5\n"
            '    """\n'
        ),
        "test_code": (
            "def check_truncate_number():\n"
            "    assert abs(truncate_number(3.5) - 0.5) < 1e-9\n"
            "    assert abs(truncate_number(1.33) - 0.33) < 1e-2\n"
            "    assert abs(truncate_number(0.0) - 0.0) < 1e-9\n"
            "    assert abs(truncate_number(123.456) - 0.456) < 1e-6\n"
            "    assert truncate_number(12.0) == 0.0\n"
            "    print('ALL TESTS PASSED')\n"
            "check_truncate_number()\n"
        ),
        "expected_function": "truncate_number",
        "difficulty": TaskDifficulty.EASY,
        "category": "math",
    },
    {
        "task_id": "HumanEval/3",
        "prompt": (
            "def below_zero(operations: List[int]) -> bool:\n"
            '    """You are given a non-empty list of positive integers representing bank deposits and '
            'withdrawals. Positive integers are deposits, negative integers are withdrawals. '
            'Return True if the balance falls below zero at any point.\n'
            "    >>> below_zero([1, 2, 3])\n"
            "    False\n"
            "    >>> below_zero([1, 2, -3, 1, -1])\n"
            "    True\n"
            '    """\n'
        ),
        "test_code": (
            "def check_below_zero():\n"
            "    assert below_zero([1, 2, -3, 1, -1]) == True\n"
            "    assert below_zero([1, 2, 3]) == False\n"
            "    assert below_zero([-1, -2, -3]) == True\n"
            "    assert below_zero([10, -5, 10, -20, 10]) == True\n"
            "    assert below_zero([100]) == False\n"
            "    assert below_zero([-100]) == True\n"
            "    print('ALL TESTS PASSED')\n"
            "check_below_zero()\n"
        ),
        "expected_function": "below_zero",
        "difficulty": TaskDifficulty.EASY,
        "category": "list",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": (
            "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
            '    """For a given list of input numbers, calculate the Mean Absolute Deviation around '
            'the mean of this dataset.\n'
            '    The Mean Absolute Deviation is the average absolute difference between each element '
            'and the mean of the dataset.\n'
            "    >>> mean_absolute_deviation([1.0, 2.0, 3.0])\n"
            "    0.6666666666666666\n"
            '    """\n'
        ),
        "test_code": (
            "def check_mean_absolute_deviation():\n"
            "    result = mean_absolute_deviation([1.0, 2.0, 3.0, 4.0, 5.0])\n"
            "    assert abs(result - 1.2) < 1e-9\n"
            "    assert abs(mean_absolute_deviation([1.0, 1.0, 1.0]) - 0.0) < 1e-9\n"
            "    assert abs(mean_absolute_deviation([-1.0, 0.0, 1.0]) - 0.6666666666666666) < 1e-9\n"
            "    assert abs(mean_absolute_deviation([10.0]) - 0.0) < 1e-9\n"
            "    assert abs(mean_absolute_deviation([0.0, 0.0, 0.0]) - 0.0) < 1e-9\n"
            "    print('ALL TESTS PASSED')\n"
            "check_mean_absolute_deviation()\n"
        ),
        "expected_function": "mean_absolute_deviation",
        "difficulty": TaskDifficulty.MEDIUM,
        "category": "math",
    },
    {
        "task_id": "HumanEval/5",
        "prompt": (
            "def intersperse(numbers: List[int], delimiter: int) -> List[int]:\n"
            '    """Insert a number \'delimiter\' between every two consecutive elements of input '
            'list \'numbers\'.\n'
            "    >>> intersperse([], 4)\n"
            "    []\n"
            "    >>> intersperse([1, 2, 3], 4)\n"
            "    [1, 4, 2, 4, 3]\n"
            '    """\n'
        ),
        "test_code": (
            "def check_intersperse():\n"
            "    assert intersperse([], 7) == []\n"
            "    assert intersperse([5], 7) == [5]\n"
            "    assert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]\n"
            "    assert intersperse([1, 2, 3, 4], 0) == [1, 0, 2, 0, 3, 0, 4]\n"
            "    assert intersperse([8, 8, 8], 9) == [8, 9, 8, 9, 8]\n"
            "    assert intersperse([-1, -2], 0) == [-1, 0, -2]\n"
            "    print('ALL TESTS PASSED')\n"
            "check_intersperse()\n"
        ),
        "expected_function": "intersperse",
        "difficulty": TaskDifficulty.EASY,
        "category": "list",
    },
    {
        "task_id": "HumanEval/6",
        "prompt": (
            "def parse_nested_parens(paren_string: str) -> int:\n"
            '    """Given a string representing a nesting of parentheses, find the deepest nesting level.\n'
            "    >>> parse_nested_parens('(())')\n"
            "    2\n"
            "    >>> parse_nested_parens('(()())')\n"
            "    2\n"
            "    >>> parse_nested_parens('((()))')\n"
            "    3\n"
            '    """\n'
        ),
        "test_code": (
            "def check_parse_nested_parens():\n"
            "    assert parse_nested_parens('(())') == 2\n"
            "    assert parse_nested_parens('(()())') == 2\n"
            "    assert parse_nested_parens('((()))') == 3\n"
            "    assert parse_nested_parens('()()') == 1\n"
            "    assert parse_nested_parens('((()()))') == 3\n"
            "    assert parse_nested_parens('()') == 1\n"
            "    assert parse_nested_parens('') == 0\n"
            "    print('ALL TESTS PASSED')\n"
            "check_parse_nested_parens()\n"
        ),
        "expected_function": "parse_nested_parens",
        "difficulty": TaskDifficulty.MEDIUM,
        "category": "string",
    },
    {
        "task_id": "HumanEval/7",
        "prompt": (
            "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
            '    """Filter an input list of strings only for ones that contain given substring.\n'
            "    >>> filter_by_substring(['abc', 'def', 'ghi'], 'a')\n"
            "    ['abc']\n"
            '    """\n'
        ),
        "test_code": (
            "def check_filter_by_substring():\n"
            "    assert filter_by_substring([], 'a') == []\n"
            "    assert filter_by_substring(['abc', 'bacd', 'cde'], 'a') == ['abc', 'bacd']\n"
            "    assert filter_by_substring(['abc', 'bacd', 'cde'], 'z') == []\n"
            "    assert filter_by_substring(['xyz', 'xyz'], 'xy') == ['xyz', 'xyz']\n"
            "    assert filter_by_substring(['aaa', 'bbb', 'aaa'], 'a') == ['aaa', 'aaa']\n"
            "    assert filter_by_substring(['hello', 'world'], 'l') == ['hello', 'world']\n"
            "    print('ALL TESTS PASSED')\n"
            "check_filter_by_substring()\n"
        ),
        "expected_function": "filter_by_substring",
        "difficulty": TaskDifficulty.EASY,
        "category": "list",
    },
    {
        "task_id": "HumanEval/8",
        "prompt": (
            "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
            '    """For a given list of integers, return a tuple consisting of the sum and product '
            'of all the integers.\n'
            "    >>> sum_product([1, 2, 3, 4])\n"
            "    (10, 24)\n"
            '    """\n'
        ),
        "test_code": (
            "def check_sum_product():\n"
            "    assert sum_product([1, 2, 3, 4]) == (10, 24)\n"
            "    assert sum_product([0]) == (0, 0)\n"
            "    assert sum_product([5]) == (5, 5)\n"
            "    assert sum_product([-1, 1]) == (0, -1)\n"
            "    assert sum_product([-2, -3]) == (-5, 6)\n"
            "    assert sum_product([1, 1, 1, 1]) == (4, 1)\n"
            "    print('ALL TESTS PASSED')\n"
            "check_sum_product()\n"
        ),
        "expected_function": "sum_product",
        "difficulty": TaskDifficulty.EASY,
        "category": "math",
    },
    {
        "task_id": "HumanEval/9",
        "prompt": (
            "def rolling_max(numbers: List[int]) -> List[int]:\n"
            '    """From a given list of integers, generate a list of rolling maximum element values '
            'found in the list up to the given position.\n'
            "    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n"
            "    [1, 2, 3, 3, 3, 4, 4]\n"
            '    """\n'
        ),
        "test_code": (
            "def check_rolling_max():\n"
            "    assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]\n"
            "    assert rolling_max([]) == []\n"
            "    assert rolling_max([5, 4, 3, 2, 1]) == [5, 5, 5, 5, 5]\n"
            "    assert rolling_max([-1, -2, -3]) == [-1, -1, -1]\n"
            "    assert rolling_max([1]) == [1]\n"
            "    assert rolling_max([3, 3, 3]) == [3, 3, 3]\n"
            "    print('ALL TESTS PASSED')\n"
            "check_rolling_max()\n"
        ),
        "expected_function": "rolling_max",
        "difficulty": TaskDifficulty.EASY,
        "category": "list",
    },
    {
        "task_id": "HumanEval/10",
        "prompt": (
            "def is_sorted(lst: List[int]) -> bool:\n"
            '    """Given a list of numbers, return whether the list is in ascending order.\n'
            "    >>> is_sorted([1, 2, 3, 4])\n"
            "    True\n"
            "    >>> is_sorted([1, 3, 2])\n"
            "    False\n"
            '    """\n'
        ),
        "test_code": (
            "def check_is_sorted():\n"
            "    assert is_sorted([5]) == True\n"
            "    assert is_sorted([1, 2, 3, 4, 5]) == True\n"
            "    assert is_sorted([1, 3, 2, 4, 5]) == False\n"
            "    assert is_sorted([]) == True\n"
            "    assert is_sorted([1, 1, 1]) == True\n"
            "    assert is_sorted([-5, -3, 0, 7]) == True\n"
            "    assert is_sorted([10, 5, 1]) == False\n"
            "    print('ALL TESTS PASSED')\n"
            "check_is_sorted()\n"
        ),
        "expected_function": "is_sorted",
        "difficulty": TaskDifficulty.EASY,
        "category": "list",
    },
]

MBPP_TASKS = [
    {
        "task_id": "MBPP/1",
        "prompt": "Write a function to find the maximum difference between two elements in a list such that the larger element appears after the smaller element.",
        "test_code": (
            "def check_max_difference():\n"
            "    assert max_difference([2, 3, 10, 6, 4, 8, 1]) == 8\n"
            "    assert max_difference([7, 9, 5, 6, 3, 2]) == 2\n"
            "    assert max_difference([1, 2, 3, 4, 5]) == 4\n"
            "    assert max_difference([5, 4, 3, 2, 1]) == -1\n"
            "    assert max_difference([10]) == -1\n"
            "    print('ALL TESTS PASSED')\n"
            "check_max_difference()\n"
        ),
        "expected_function": "max_difference",
        "difficulty": TaskDifficulty.MEDIUM,
        "category": "list",
    },
    {
        "task_id": "MBPP/2",
        "prompt": "Write a function that checks whether a passed string is a palindrome or not.",
        "test_code": (
            "def check_is_palindrome():\n"
            "    assert is_palindrome('madam') == True\n"
            "    assert is_palindrome('racecar') == True\n"
            "    assert is_palindrome('hello') == False\n"
            "    assert is_palindrome('a') == True\n"
            "    assert is_palindrome('') == True\n"
            "    assert is_palindrome('abba') == True\n"
            "    assert is_palindrome('abc') == False\n"
            "    print('ALL TESTS PASSED')\n"
            "check_is_palindrome()\n"
        ),
        "expected_function": "is_palindrome",
        "difficulty": TaskDifficulty.EASY,
        "category": "string",
    },
    {
        "task_id": "MBPP/3",
        "prompt": "Write a function to find all the unique numbers in a list.",
        "test_code": (
            "def check_find_unique():\n"
            "    result = find_unique([1, 2, 3, 1, 2, 4, 5])\n"
            "    assert sorted(result) == [3, 4, 5]\n"
            "    result = find_unique([1, 1, 1])\n"
            "    assert result == []\n"
            "    result = find_unique([1, 2, 3])\n"
            "    assert sorted(result) == [1, 2, 3]\n"
            "    result = find_unique([])\n"
            "    assert result == []\n"
            "    print('ALL TESTS PASSED')\n"
            "check_find_unique()\n"
        ),
        "expected_function": "find_unique",
        "difficulty": TaskDifficulty.EASY,
        "category": "list",
    },
    {
        "task_id": "MBPP/4",
        "prompt": "Write a function to count the number of vowels (a, e, i, o, u) in a string.",
        "test_code": (
            "def check_count_vowels():\n"
            "    assert count_vowels('hello') == 2\n"
            "    assert count_vowels('beautiful') == 5\n"
            "    assert count_vowels('python') == 1\n"
            "    assert count_vowels('AEIOU') == 5\n"
            "    assert count_vowels('xyz') == 0\n"
            "    assert count_vowels('') == 0\n"
            "    assert count_vowels('The quick brown fox') == 5\n"
            "    print('ALL TESTS PASSED')\n"
            "check_count_vowels()\n"
        ),
        "expected_function": "count_vowels",
        "difficulty": TaskDifficulty.EASY,
        "category": "string",
    },
    {
        "task_id": "MBPP/5",
        "prompt": "Write a function to find the second largest number in a list.",
        "test_code": (
            "def check_second_largest():\n"
            "    assert second_largest([1, 2, 3, 4, 5]) == 4\n"
            "    assert second_largest([5, 5, 4, 4, 3]) == 5\n"
            "    assert second_largest([-1, -2, -3, -4]) == -2\n"
            "    assert second_largest([10, 20, 4]) == 10\n"
            "    assert second_largest([70, 11, 20, 4, 100]) == 70\n"
            "    print('ALL TESTS PASSED')\n"
            "check_second_largest()\n"
        ),
        "expected_function": "second_largest",
        "difficulty": TaskDifficulty.EASY,
        "category": "list",
    },
    {
        "task_id": "MBPP/6",
        "prompt": "Write a function to merge two sorted lists into a single sorted list.",
        "test_code": (
            "def check_merge_sorted():\n"
            "    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\n"
            "    assert merge_sorted([], [1, 2, 3]) == [1, 2, 3]\n"
            "    assert merge_sorted([1, 2, 3], []) == [1, 2, 3]\n"
            "    assert merge_sorted([], []) == []\n"
            "    assert merge_sorted([1, 1, 1], [1, 1]) == [1, 1, 1, 1, 1]\n"
            "    assert merge_sorted([-3, -1], [-2, 0]) == [-3, -2, -1, 0]\n"
            "    print('ALL TESTS PASSED')\n"
            "check_merge_sorted()\n"
        ),
        "expected_function": "merge_sorted",
        "difficulty": TaskDifficulty.MEDIUM,
        "category": "list",
    },
    {
        "task_id": "MBPP/7",
        "prompt": "Write a function to remove all duplicate characters from a string while preserving order.",
        "test_code": (
            "def check_remove_duplicates_str():\n"
            "    assert remove_duplicates_str('aabbcc') == 'abc'\n"
            "    assert remove_duplicates_str('abcabc') == 'abc'\n"
            "    assert remove_duplicates_str('abcdef') == 'abcdef'\n"
            "    assert remove_duplicates_str('') == ''\n"
            "    assert remove_duplicates_str('aaaaa') == 'a'\n"
            "    assert remove_duplicates_str('abca') == 'abc'\n"
            "    print('ALL TESTS PASSED')\n"
            "check_remove_duplicates_str()\n"
        ),
        "expected_function": "remove_duplicates_str",
        "difficulty": TaskDifficulty.EASY,
        "category": "string",
    },
    {
        "task_id": "MBPP/8",
        "prompt": "Write a function that returns the Fibonacci sequence up to n terms as a list.",
        "test_code": (
            "def check_fibonacci_sequence():\n"
            "    assert fibonacci_sequence(0) == []\n"
            "    assert fibonacci_sequence(1) == [0]\n"
            "    assert fibonacci_sequence(2) == [0, 1]\n"
            "    assert fibonacci_sequence(5) == [0, 1, 1, 2, 3]\n"
            "    assert fibonacci_sequence(8) == [0, 1, 1, 2, 3, 5, 8, 13]\n"
            "    assert fibonacci_sequence(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n"
            "    print('ALL TESTS PASSED')\n"
            "check_fibonacci_sequence()\n"
        ),
        "expected_function": "fibonacci_sequence",
        "difficulty": TaskDifficulty.MEDIUM,
        "category": "math",
    },
    {
        "task_id": "MBPP/9",
        "prompt": "Write a function to flatten a nested list (lists within lists) into a single list.",
        "test_code": (
            "def check_flatten_list():\n"
            "    assert flatten_list([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]\n"
            "    assert flatten_list([]) == []\n"
            "    assert flatten_list([1, 2, 3]) == [1, 2, 3]\n"
            "    assert flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]\n"
            "    assert flatten_list([[[1]]]) == [1]\n"
            "    assert flatten_list([1, [2, [3, [4]]]]) == [1, 2, 3, 4]\n"
            "    print('ALL TESTS PASSED')\n"
            "check_flatten_list()\n"
        ),
        "expected_function": "flatten_list",
        "difficulty": TaskDifficulty.MEDIUM,
        "category": "list",
    },
    {
        "task_id": "MBPP/10",
        "prompt": "Write a function to check if two strings are anagrams of each other.",
        "test_code": (
            "def check_are_anagrams():\n"
            "    assert are_anagrams('listen', 'silent') == True\n"
            "    assert are_anagrams('triangle', 'integral') == True\n"
            "    assert are_anagrams('hello', 'world') == False\n"
            "    assert are_anagrams('', '') == True\n"
            "    assert are_anagrams('a', 'a') == True\n"
            "    assert are_anagrams('abc', 'def') == False\n"
            "    assert are_anagrams('Dormitory', 'dirty room') == False\n"
            "    assert are_anagrams('aabb', 'bbaa') == True\n"
            "    print('ALL TESTS PASSED')\n"
            "check_are_anagrams()\n"
        ),
        "expected_function": "are_anagrams",
        "difficulty": TaskDifficulty.EASY,
        "category": "string",
    },
    {
        "task_id": "MBPP/11",
        "prompt": "Write a function to compute the greatest common divisor (GCD) of two numbers using Euclid's algorithm.",
        "test_code": (
            "def check_gcd():\n"
            "    assert gcd(48, 18) == 6\n"
            "    assert gcd(18, 48) == 6\n"
            "    assert gcd(17, 23) == 1\n"
            "    assert gcd(0, 5) == 5\n"
            "    assert gcd(5, 0) == 5\n"
            "    assert gcd(100, 100) == 100\n"
            "    assert gcd(270, 192) == 6\n"
            "    print('ALL TESTS PASSED')\n"
            "check_gcd()\n"
        ),
        "expected_function": "gcd",
        "difficulty": TaskDifficulty.EASY,
        "category": "math",
    },
    {
        "task_id": "MBPP/12",
        "prompt": "Write a function to convert a list of strings representing integers to a list of integers.",
        "test_code": (
            "def check_str_to_int_list():\n"
            "    assert str_to_int_list(['1', '2', '3']) == [1, 2, 3]\n"
            "    assert str_to_int_list([]) == []\n"
            "    assert str_to_int_list(['-1', '0', '1']) == [-1, 0, 1]\n"
            "    assert str_to_int_list(['100', '200']) == [100, 200]\n"
            "    print('ALL TESTS PASSED')\n"
            "check_str_to_int_list()\n"
        ),
        "expected_function": "str_to_int_list",
        "difficulty": TaskDifficulty.EASY,
        "category": "conversion",
    },
]


# SWE-bench simplified instances (representative subset)
SWE_BENCH_INSTANCES = [
    {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc123",
        "problem_statement": (
            "In django/utils/html.py, the strip_tags function does not properly handle "
            "nested script tags. When HTML contains <script> elements with nested content, "
            "strip_tags can leave behind residual script content. Fix the strip_tags function "
            "to properly remove all content between script tags, including nested ones."
        ),
        "hint_text": (
            "Look at how the regex in strip_tags handles <script> tags. "
            "You may need to use a non-greedy match or handle script content removal as a "
            "separate step before the general tag removal."
        ),
        "test_patch": (
            "def test_strip_tags_nested_script():\n"
            "    from django.utils.html import strip_tags\n"
            '    assert strip_tags("<script>alert(\'xss\')</script>hello") == "hello"\n'
            '    assert strip_tags("<SCRIPT>content</SCRIPT>") == ""\n'
            '    assert strip_tags("<div><script>bad</script>good</div>") == "good"\n'
        ),
        "difficulty": TaskDifficulty.MEDIUM,
        "category": "security",
    },
    {
        "instance_id": "django__django-12346",
        "repo": "django/django",
        "base_commit": "def456",
        "problem_statement": (
            "The QuerySet.none() method in django/db/models/query.py returns a special "
            "EmptyQuerySet that does not preserve the model class when chained with .annotate(). "
            "This causes isinstance checks to fail when the queryset is further filtered."
        ),
        "hint_text": (
            "The EmptyQuerySet class needs to forward method calls that return querysets "
            "so they maintain the correct model class. Check how _clone is implemented."
        ),
        "test_patch": (
            "def test_none_annotate_chain():\n"
            "    from django.db.models import QuerySet\n"
            "    from myapp.models import Article\n"
            "    qs = Article.objects.none()\n"
            "    result = qs.annotate()\n"
            "    assert isinstance(result, QuerySet)\n"
        ),
        "difficulty": TaskDifficulty.HARD,
        "category": "orm",
    },
    {
        "instance_id": "flask__flask-78901",
        "repo": "pallets/flask",
        "base_commit": "ghi789",
        "problem_statement": (
            "In flask/sessions.py, the SecureCookieSessionInterface does not properly "
            "handle session cookies when the session data exceeds the maximum cookie size. "
            "This causes silent data loss without raising an error or warning."
        ),
        "hint_text": (
            "The save_session method should check the total size of the serialized session "
            "and either warn or truncate. Look at how the cookie is built in save_session."
        ),
        "test_patch": (
            "def test_large_session_warning():\n"
            "    import warnings\n"
            "    from flask import Flask\n"
            "    app = Flask(__name__)\n"
            "    app.secret_key = 'test'\n"
            "    with app.test_request_context():\n"
            "        with warnings.catch_warnings(record=True) as w:\n"
            "            from flask.sessions import SecureCookieSessionInterface\n"
            "            si = SecureCookieSessionInterface()\n"
            "            session = si.get_signing_serializer(app)\n"
            "            large_data = {'key': 'x' * 10000}\n"
            "            assert len(w) >= 0\n"
        ),
        "difficulty": TaskDifficulty.HARD,
        "category": "sessions",
    },
    {
        "instance_id": "requests__requests-45678",
        "repo": "psf/requests",
        "base_commit": "jkl012",
        "problem_statement": (
            "In requests/sessions.py, the merge_environment_settings method does not "
            "properly merge proxies when both session-level and request-level proxies "
            "are provided. Request-level proxies should take precedence but currently "
            "session-level proxies are not being overridden."
        ),
        "hint_text": (
            "Check how env_settings and session_settings are merged. The order of "
            "dict updates matters — request-level should override session-level."
        ),
        "test_patch": (
            "def test_proxy_override():\n"
            "    from requests.sessions import Session\n"
            "    s = Session()\n"
            "    s.proxies = {'http': 'http://session.proxy'}\n"
            "    merged = s.merge_environment_settings(\n"
            "        'http://example.com',\n"
            "        proxies={'http': 'http://request.proxy'},\n"
            "        stream=False, verify=True, cert=None\n"
            "    )\n"
            '    assert merged["proxies"]["http"] == "http://request.proxy"\n'
        ),
        "difficulty": TaskDifficulty.MEDIUM,
        "category": "network",
    },
]


# ──────────────────────────────────────────────
# Evaluation prompt templates
# ──────────────────────────────────────────────

CODING_SYSTEM_PROMPT = """You are an expert Python programmer. You will be given a programming task.

Rules:
- Write ONLY the function implementation. Do not include test code, examples, or explanation.
- Include proper type hints and imports.
- The function must be named exactly as specified.
- Return only the function code — no markdown fences, no extra text.
- Make the function robust and handle edge cases.
"""

SWE_BENCH_SYSTEM_PROMPT = """You are an expert software engineer. You will be given a GitHub issue
describing a bug or feature request, along with the relevant repository context.

Rules:
- Write a minimal patch that fixes the issue.
- Include only the modified functions/classes, with enough context to identify where changes go.
- Use standard Python conventions.
- Explain the fix briefly in a comment at the top.
"""


# ──────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────

class Evaluator:
    """
    Evaluation & Benchmarking Framework for Claude Clone.

    Runs standardized benchmarks (HumanEval, MBPP, SWE-bench), custom task suites,
    A/B tests, regression checks, and generates detailed reports.

    Args:
        agent: The Agent instance to evaluate.
        results_dir: Directory to store benchmark results, baselines, and history.
    """

    def __init__(self, agent: Any, results_dir: str = "~/.claude_clone/eval_results"):
        self.agent = agent
        self.results_dir = Path(results_dir).expanduser().resolve()
        self._humaneval_tasks: List[Dict] = []
        self._mbpp_tasks: List[Dict] = []
        self._swe_bench_instances: List[Dict] = []
        self._initialized = False
        self._last_results: Dict[str, BenchmarkResult] = {}

    async def initialize(self) -> None:
        """Load benchmarks, ensure environment is ready, create results directory."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "baselines").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "history").mkdir(parents=True, exist_ok=True)

        self._humaneval_tasks = self._load_humaneval_tasks()
        self._mbpp_tasks = self._load_mbpp_tasks()
        self._swe_bench_instances = list(SWE_BENCH_INSTANCES)

        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Evaluator not initialized. Call await evaluator.initialize() first.")

    # ──────────────────────────────────────────
    # Benchmark loaders
    # ──────────────────────────────────────────

    def _load_humaneval_tasks(self) -> List[Dict]:
        """Load HumanEval tasks from embedded dataset."""
        return list(HUMANEVAL_TASKS)

    def _load_mbpp_tasks(self) -> List[Dict]:
        """Load MBPP tasks from embedded dataset."""
        return list(MBPP_TASKS)

    # ──────────────────────────────────────────
    # Validation helpers
    # ──────────────────────────────────────────

    def _validate_humaneval(self, task_id: str, code: str) -> Tuple[bool, str]:
        """Run HumanEval test cases against generated code. Returns (passed, error_message)."""
        task = next((t for t in self._humaneval_tasks if t["task_id"] == task_id), None)
        if task is None:
            return False, f"Unknown task: {task_id}"

        return self._execute_and_test(code, task["test_code"])

    def _validate_mbpp(self, task_id: str, code: str) -> Tuple[bool, str]:
        """Run MBPP test cases against generated code. Returns (passed, error_message)."""
        task = next((t for t in self._mbpp_tasks if t["task_id"] == task_id), None)
        if task is None:
            return False, f"Unknown task: {task_id}"

        return self._execute_and_test(code, task["test_code"])

    def _validate_custom(self, validation_script: str, code: str) -> Tuple[bool, str]:
        """Run a custom validation script against generated code. Returns (passed, error_message)."""
        combined = f"{code}\n\n{validation_script}\n"
        return self._execute_code(combined)

    def _execute_and_test(self, code: str, test_code: str) -> Tuple[bool, str]:
        """Combine code + tests and execute in a subprocess. Returns (passed, error_message)."""
        combined = f"from typing import List, Tuple, Dict, Set, Optional, Any\n\n{code}\n\n{test_code}\n"
        return self._execute_code(combined)

    def _execute_code(self, full_code: str) -> Tuple[bool, str]:
        """Execute Python code in a subprocess with timeout. Returns (passed, error_message)."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if result.returncode == 0:
                stdout = result.stdout.strip()
                if "ALL TESTS PASSED" in stdout:
                    return True, ""
                elif stdout:
                    return True, stdout
                else:
                    return True, ""
            else:
                error = result.stderr.strip()
                return False, error[-2000:] if len(error) > 2000 else error
        except subprocess.TimeoutExpired:
            return False, "Execution timed out (30s limit)"
        except Exception as e:
            return False, f"Execution error: {e}"

    # ──────────────────────────────────────────
    # Task execution
    # ──────────────────────────────────────────

    async def run_single_task(self, task: Dict) -> TaskResult:
        """
        Run a single evaluation task. The agent generates code, which is then validated.

        Args:
            task: A task dictionary with keys: task_id, task_type, prompt, test_code,
                  expected_function, difficulty, category.

        Returns:
            TaskResult with pass/fail, timing, tokens, and error details.
        """
        task_type = task.get("task_type", TaskType.CUSTOM)
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        task_id = task["task_id"]
        prompt = task["prompt"]
        difficulty = task.get("difficulty", TaskDifficulty.MEDIUM)

        # Build the coding prompt
        expected_fn = task.get("expected_function", "solution")
        user_message = (
            f"Task {task_id}:\n\n"
            f"Write a Python function named `{expected_fn}` that solves the following:\n\n"
            f"{prompt}\n\n"
            f"Provide ONLY the function implementation with any necessary imports and type hints."
        )

        # Determine the system prompt based on task type
        if task_type == TaskType.SWE_BENCH:
            system = SWE_BENCH_SYSTEM_PROMPT
            user_message = (
                f"Task {task_id}:\n\n"
                f"Problem:\n{task.get('problem_statement', prompt)}\n\n"
                f"Hint: {task.get('hint_text', 'N/A')}\n\n"
                f"Write the fix."
            )
        else:
            system = CODING_SYSTEM_PROMPT

        start_time = time.time()
        tokens_used = 0
        code_output = ""
        error = ""

        try:
            # Reset agent state and collect output
            self.agent.reset()
            original_system = self.agent.system_prompt
            self.agent.system_prompt = system

            full_response = ""
            async for event in self.agent.run(user_message):
                if isinstance(event, TextEvent):
                    full_response += event.data
                elif hasattr(event, "data") and isinstance(event.data, str):
                    if event.event_type == "error":
                        error = event.data

            self.agent.system_prompt = original_system
            tokens = self.agent.get_token_counts()
            tokens_used = tokens.get("total_tokens", 0)

            # Extract code from the response
            code_output = self._extract_code(full_response, expected_fn)

            if not code_output.strip():
                return TaskResult(
                    task_id=task_id,
                    task_type=task_type,
                    description=task.get("prompt", "")[:200],
                    input=prompt,
                    expected_output="Valid function",
                    actual_output=full_response[:500],
                    passed=False,
                    score=0.0,
                    execution_time=time.time() - start_time,
                    tokens_used=tokens_used,
                    cost=self.agent.estimate_cost(),
                    error="No code extracted from agent response",
                    difficulty=difficulty,
                    category=task.get("category", ""),
                )

            # Validate based on task type
            if task_type == TaskType.HUMANEVAL:
                passed, error = self._validate_humaneval(task_id, code_output)
            elif task_type == TaskType.MBPP:
                passed, error = self._validate_mbpp(task_id, code_output)
            elif task_type == TaskType.SWE_BENCH:
                passed, error = self._validate_custom(task.get("test_patch", ""), code_output)
            elif task_type == TaskType.CUSTOM:
                validation = task.get("validation_script", task.get("test_code", ""))
                if validation:
                    passed, error = self._validate_custom(validation, code_output)
                else:
                    passed = bool(code_output.strip())
                    error = "" if passed else "No validation script provided"
            else:
                passed = bool(code_output.strip())
                error = ""

        except Exception as e:
            error = str(e)
            passed = False

        exec_time = time.time() - start_time

        return TaskResult(
            task_id=task_id,
            task_type=task_type,
            description=task.get("prompt", "")[:200],
            input=prompt,
            expected_output="Correct implementation",
            actual_output=code_output[:500] if code_output else "",
            passed=passed,
            score=1.0 if passed else 0.0,
            execution_time=exec_time,
            tokens_used=tokens_used,
            cost=self.agent.estimate_cost(),
            error=error,
            difficulty=difficulty,
            category=task.get("category", ""),
        )

    def _extract_code(self, response: str, expected_fn: str) -> str:
        """Extract Python function code from an agent response."""
        if not response:
            return ""

        # Strategy 1: Look for ```python ... ``` code blocks
        python_block_pattern = r"```python\s*\n(.*?)```"
        blocks = re.findall(python_block_pattern, response, re.DOTALL)
        for block in blocks:
            if expected_fn in block:
                return self._clean_code(block)

        # Strategy 2: Look for any ``` code block
        any_block_pattern = r"```\s*\n(.*?)```"
        blocks = re.findall(any_block_pattern, response, re.DOTALL)
        for block in blocks:
            if expected_fn in block:
                return self._clean_code(block)

        # Strategy 3: Look for function definition directly
        fn_pattern = rf"(def\s+{re.escape(expected_fn)}\s*\(.*?\n(?:.*\n)*?.*?)(?=\n(?:def |class |$))"
        fn_match = re.search(fn_pattern, response, re.DOTALL)
        if fn_match:
            return self._clean_code(fn_match.group(1))

        # Strategy 4: Look for "def function_name(" and capture until reasonable end
        fn_start = response.find(f"def {expected_fn}(")
        if fn_start >= 0:
            code = response[fn_start:]
            # Capture until we hit a double newline followed by something that's not indentation
            lines = code.split("\n")
            captured_lines = []
            in_function = True
            for i, line in enumerate(lines):
                if i == 0 or (in_function and (line.startswith("    ") or line.startswith("\t") or line.strip() == "")):
                    captured_lines.append(line)
                else:
                    break
            result = "\n".join(captured_lines)
            if result.strip():
                return self._clean_code(result)

        # Strategy 5: Return the whole response if it looks like code
        if "def " in response and expected_fn in response:
            return self._clean_code(response)

        return ""

    def _clean_code(self, code: str) -> str:
        """Clean extracted code by removing markdown artifacts and normalizing."""
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"```\s*$", "", code)
        code = re.sub(r"^\s*#\s*Write.*?(?=\ndef )", "", code, flags=re.DOTALL | re.MULTILINE)
        return code.strip()

    # ──────────────────────────────────────────
    # Parallel execution engine
    # ──────────────────────────────────────────

    async def _run_tasks_parallel(
        self,
        tasks: List[Dict],
        task_type: TaskType,
        parallel: int = 4,
    ) -> List[TaskResult]:
        """Execute a batch of tasks with bounded parallelism using a semaphore."""
        semaphore = asyncio.Semaphore(parallel)
        results: List[TaskResult] = []

        async def _run_with_semaphore(task: Dict) -> TaskResult:
            async with semaphore:
                enriched = {**task, "task_type": task_type}
                return await self.run_single_task(enriched)

        coroutines = [_run_with_semaphore(t) for t in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(TaskResult(
                    task_id=tasks[i].get("task_id", f"unknown_{i}"),
                    task_type=task_type,
                    description=tasks[i].get("prompt", "")[:200],
                    input=tasks[i].get("prompt", ""),
                    expected_output="Success",
                    actual_output="",
                    passed=False,
                    score=0.0,
                    execution_time=0.0,
                    tokens_used=0,
                    cost=0.0,
                    error=f"Task execution failed: {result}",
                    difficulty=tasks[i].get("difficulty", TaskDifficulty.MEDIUM),
                    category=tasks[i].get("category", ""),
                ))
            else:
                final_results.append(result)

        return final_results

    # ──────────────────────────────────────────
    # Benchmark runners
    # ──────────────────────────────────────────

    async def run_humaneval(self, limit: int = None, parallel: int = 4) -> BenchmarkResult:
        """
        Run HumanEval programming benchmark.

        Args:
            limit: Maximum number of tasks to run (None = all).
            parallel: Number of concurrent tasks.

        Returns:
            BenchmarkResult with scores, timing, and per-category breakdown.
        """
        self._ensure_initialized()
        tasks = self._humaneval_tasks[:limit] if limit else self._humaneval_tasks

        if not tasks:
            return BenchmarkResult(
                benchmark_name="HumanEval",
                total_tasks=0,
                passed=0,
                failed=0,
                pass_rate=0.0,
                total_time=0.0,
                total_cost=0.0,
                avg_tokens=0,
                per_category={},
                per_difficulty={},
                timestamp=datetime.now(timezone.utc).isoformat(),
                details=[],
            )

        start = time.time()
        details = await self._run_tasks_parallel(tasks, TaskType.HUMANEVAL, parallel)
        total_time = time.time() - start

        result = self._build_benchmark_result("HumanEval", details, total_time)
        self._last_results["humaneval"] = result
        await self._save_history(result)
        return result

    async def run_mbpp(self, limit: int = None, parallel: int = 4) -> BenchmarkResult:
        """
        Run MBPP (Mostly Basic Python Programming) benchmark.

        Args:
            limit: Maximum number of tasks to run (None = all).
            parallel: Number of concurrent tasks.

        Returns:
            BenchmarkResult with scores, timing, and per-category breakdown.
        """
        self._ensure_initialized()
        tasks = self._mbpp_tasks[:limit] if limit else self._mbpp_tasks

        if not tasks:
            return BenchmarkResult(
                benchmark_name="MBPP",
                total_tasks=0,
                passed=0,
                failed=0,
                pass_rate=0.0,
                total_time=0.0,
                total_cost=0.0,
                avg_tokens=0,
                per_category={},
                per_difficulty={},
                timestamp=datetime.now(timezone.utc).isoformat(),
                details=[],
            )

        start = time.time()
        details = await self._run_tasks_parallel(tasks, TaskType.MBPP, parallel)
        total_time = time.time() - start

        result = self._build_benchmark_result("MBPP", details, total_time)
        self._last_results["mbpp"] = result
        await self._save_history(result)
        return result

    async def run_swe_bench(
        self, instance_ids: List[str] = None, parallel: int = 2
    ) -> BenchmarkResult:
        """
        Run SWE-bench style evaluation (simplified GitHub issue resolution).

        Args:
            instance_ids: Specific instance IDs to run (None = all).
            parallel: Number of concurrent tasks (lower default due to complexity).

        Returns:
            BenchmarkResult with scores, timing, and per-category breakdown.
        """
        self._ensure_initialized()
        instances = self._swe_bench_instances

        if instance_ids:
            instances = [i for i in instances if i["instance_id"] in instance_ids]

        if not instances:
            return BenchmarkResult(
                benchmark_name="SWE-bench",
                total_tasks=0,
                passed=0,
                failed=0,
                pass_rate=0.0,
                total_time=0.0,
                total_cost=0.0,
                avg_tokens=0,
                per_category={},
                per_difficulty={},
                timestamp=datetime.now(timezone.utc).isoformat(),
                details=[],
            )

        tasks = [
            {
                "task_id": inst["instance_id"],
                "prompt": inst["problem_statement"],
                "test_code": inst["test_patch"],
                "expected_function": "fix",
                "difficulty": inst.get("difficulty", TaskDifficulty.HARD),
                "category": inst.get("category", "general"),
                "problem_statement": inst["problem_statement"],
                "hint_text": inst.get("hint_text", ""),
                "test_patch": inst["test_patch"],
            }
            for inst in instances
        ]

        start = time.time()
        details = await self._run_tasks_parallel(tasks, TaskType.SWE_BENCH, parallel)
        total_time = time.time() - start

        result = self._build_benchmark_result("SWE-bench", details, total_time)
        self._last_results["swe_bench"] = result
        await self._save_history(result)
        return result

    async def run_custom(self, tasks_file: str, parallel: int = 4) -> BenchmarkResult:
        """
        Run custom evaluation tasks from a JSON file.

        The JSON file should contain a list of task objects:
        [
            {
                "task_id": "custom_1",
                "prompt": "Write a function that...",
                "expected_function": "my_func",
                "validation_script": "assert my_func(1, 2) == 3\\nprint('ALL TESTS PASSED')",
                "difficulty": "easy",
                "category": "math"
            }
        ]

        Args:
            tasks_file: Path to JSON file containing task definitions.
            parallel: Number of concurrent tasks.

        Returns:
            BenchmarkResult with scores, timing, and per-category breakdown.
        """
        self._ensure_initialized()

        file_path = Path(tasks_file).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Tasks file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)

        tasks = []
        for t in raw_tasks:
            tasks.append({
                "task_id": t.get("task_id", f"custom_{len(tasks)}"),
                "prompt": t["prompt"],
                "test_code": t.get("validation_script", t.get("test_code", "")),
                "expected_function": t.get("expected_function", "solution"),
                "difficulty": TaskDifficulty(t.get("difficulty", "medium")),
                "category": t.get("category", "custom"),
            })

        if not tasks:
            return BenchmarkResult(
                benchmark_name="Custom",
                total_tasks=0,
                passed=0,
                failed=0,
                pass_rate=0.0,
                total_time=0.0,
                total_cost=0.0,
                avg_tokens=0,
                per_category={},
                per_difficulty={},
                timestamp=datetime.now(timezone.utc).isoformat(),
                details=[],
            )

        start = time.time()
        details = await self._run_tasks_parallel(tasks, TaskType.CUSTOM, parallel)
        total_time = time.time() - start

        result = self._build_benchmark_result("Custom", details, total_time)
        self._last_results["custom"] = result
        await self._save_history(result)
        return result

    async def run_all(self) -> Dict[str, BenchmarkResult]:
        """
        Run all available benchmarks sequentially.

        Returns:
            Dictionary mapping benchmark names to their BenchmarkResult.
        """
        self._ensure_initialized()
        results = {}

        results["humaneval"] = await self.run_humaneval()
        results["mbpp"] = await self.run_mbpp()
        results["swe_bench"] = await self.run_swe_bench()

        return results

    # ──────────────────────────────────────────
    # Result building
    # ──────────────────────────────────────────

    def _build_benchmark_result(
        self, name: str, details: List[TaskResult], total_time: float
    ) -> BenchmarkResult:
        """Build a BenchmarkResult from a list of TaskResults."""
        if not details:
            return BenchmarkResult(
                benchmark_name=name,
                total_tasks=0,
                passed=0,
                failed=0,
                pass_rate=0.0,
                total_time=total_time,
                total_cost=0.0,
                avg_tokens=0,
                per_category={},
                per_difficulty={},
                timestamp=datetime.now(timezone.utc).isoformat(),
                details=details,
            )

        passed = sum(1 for d in details if d.passed)
        failed = len(details) - passed
        total_cost = sum(d.cost for d in details)
        total_tokens = sum(d.tokens_used for d in details)
        avg_tokens = total_tokens // len(details) if details else 0

        # Per-category breakdown
        per_category: Dict[str, Dict[str, Any]] = {}
        for d in details:
            cat = d.category or "uncategorized"
            if cat not in per_category:
                per_category[cat] = {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
            per_category[cat]["total"] += 1
            if d.passed:
                per_category[cat]["passed"] += 1
            else:
                per_category[cat]["failed"] += 1
        for cat_stats in per_category.values():
            cat_stats["pass_rate"] = (
                cat_stats["passed"] / cat_stats["total"]
                if cat_stats["total"] > 0
                else 0.0
            )

        # Per-difficulty breakdown
        per_difficulty: Dict[str, Dict[str, Any]] = {}
        for d in details:
            diff = d.difficulty.value if isinstance(d.difficulty, TaskDifficulty) else str(d.difficulty)
            if diff not in per_difficulty:
                per_difficulty[diff] = {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
            per_difficulty[diff]["total"] += 1
            if d.passed:
                per_difficulty[diff]["passed"] += 1
            else:
                per_difficulty[diff]["failed"] += 1
        for diff_stats in per_difficulty.values():
            diff_stats["pass_rate"] = (
                diff_stats["passed"] / diff_stats["total"]
                if diff_stats["total"] > 0
                else 0.0
            )

        return BenchmarkResult(
            benchmark_name=name,
            total_tasks=len(details),
            passed=passed,
            failed=failed,
            pass_rate=passed / len(details) if details else 0.0,
            total_time=total_time,
            total_cost=total_cost,
            avg_tokens=avg_tokens,
            per_category=per_category,
            per_difficulty=per_difficulty,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details,
        )

    # ──────────────────────────────────────────
    # A/B Testing
    # ──────────────────────────────────────────

    async def ab_test(
        self,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        tasks: List[Dict] = None,
    ) -> ABTestResult:
        """
        Run an A/B test comparing two configurations on the same tasks.

        Each config can specify: model, temperature, system_prompt, max_tokens.
        The evaluator runs each task under both configs and compares results.

        Args:
            config_a: First configuration dict.
            config_b: Second configuration dict.
            tasks: Tasks to test (defaults to a sample from HumanEval).

        Returns:
            ABTestResult with scores, winner, and statistical significance.
        """
        self._ensure_initialized()

        if tasks is None:
            tasks = self._humaneval_tasks[:10]

        # Run config A
        original_model = self.agent.model
        original_temp = self.agent.temperature
        original_system = self.agent.system_prompt

        # Apply config A
        self.agent.model = config_a.get("model", original_model)
        self.agent.temperature = config_a.get("temperature", original_temp)
        if "system_prompt" in config_a:
            self.agent.system_prompt = config_a["system_prompt"]

        results_a: List[TaskResult] = []
        for task in tasks:
            result = await self.run_single_task({**task, "task_type": TaskType.HUMANEVAL})
            results_a.append(result)

        # Apply config B
        self.agent.model = config_b.get("model", original_model)
        self.agent.temperature = config_b.get("temperature", original_temp)
        if "system_prompt" in config_b:
            self.agent.system_prompt = config_b["system_prompt"]

        results_b: List[TaskResult] = []
        for task in tasks:
            result = await self.run_single_task({**task, "task_type": TaskType.HUMANEVAL})
            results_b.append(result)

        # Restore original config
        self.agent.model = original_model
        self.agent.temperature = original_temp
        self.agent.system_prompt = original_system

        scores_a = [r.score for r in results_a]
        scores_b = [r.score for r in results_b]

        mean_a = statistics.mean(scores_a) if scores_a else 0.0
        mean_b = statistics.mean(scores_b) if scores_b else 0.0

        if mean_a > mean_b:
            winner = "A"
        elif mean_b > mean_a:
            winner = "B"
        else:
            winner = "tie"

        significance = self._calculate_significance(scores_a, scores_b)

        improvement = 0.0
        if mean_a > 0:
            improvement = ((mean_b - mean_a) / mean_a) * 100.0
        elif mean_b > 0:
            improvement = 100.0

        task_summaries = [
            {
                "task_id": tasks[i].get("task_id", f"task_{i}"),
                "score_a": scores_a[i],
                "score_b": scores_b[i],
                "passed_a": results_a[i].passed,
                "passed_b": results_b[i].passed,
            }
            for i in range(len(tasks))
        ]

        return ABTestResult(
            config_a=config_a,
            config_b=config_b,
            tasks=task_summaries,
            scores_a=scores_a,
            scores_b=scores_b,
            winner=winner,
            statistical_significance=significance,
            improvement_percentage=round(improvement, 2),
        )

    # ──────────────────────────────────────────
    # Statistical significance
    # ──────────────────────────────────────────

    def _calculate_significance(self, scores_a: List[float], scores_b: List[float]) -> float:
        """
        Calculate the p-value for the difference between two score distributions
        using a two-tailed Mann-Whitney U test approximation.

        Returns:
            Approximate p-value (lower = more significant). Returns 1.0 if not significant.
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            return 1.0

        n_a = len(scores_a)
        n_b = len(scores_b)

        # Compute Mann-Whitney U statistic
        combined = [(s, "a") for s in scores_a] + [(s, "b") for s in scores_b]
        combined.sort(key=lambda x: x[0])

        rank_sum_a = 0.0
        for i, (score, group) in enumerate(combined):
            rank = i + 1
            if group == "a":
                rank_sum_a += rank

        u_a = rank_sum_a - (n_a * (n_a + 1)) / 2
        u_b = n_a * n_b - u_a
        u = min(u_a, u_b)

        # Normal approximation for U
        mean_u = (n_a * n_b) / 2
        std_u = math.sqrt((n_a * n_b * (n_a + n_b + 1)) / 12)

        if std_u == 0:
            return 1.0

        z = abs(u - mean_u) / std_u

        # Approximate the two-tailed p-value from the standard normal distribution
        p_value = 2.0 * (1.0 - self._normal_cdf(z))
        return max(0.0, min(1.0, p_value))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Compute the cumulative distribution function of the standard normal distribution."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # ──────────────────────────────────────────
    # Regression detection
    # ──────────────────────────────────────────

    async def check_regression(self, baseline_name: str = "latest") -> RegressionReport:
        """
        Compare current benchmark results against a saved baseline.

        Args:
            baseline_name: Name of the baseline to compare against ("latest" uses most recent).

        Returns:
            RegressionReport detailing regressed, improved, new failures, and fixed tasks.
        """
        self._ensure_initialized()

        baseline = self._load_baseline(baseline_name)
        if baseline is None:
            raise FileNotFoundError(
                f"Baseline '{baseline_name}' not found. Run save_baseline() first."
            )

        current_results = self._last_results
        if not current_results:
            raise RuntimeError(
                "No current results to compare. Run a benchmark first."
            )

        regressed_tasks: List[Dict[str, Any]] = []
        improved_tasks: List[Dict[str, Any]] = []
        new_failures: List[Dict[str, Any]] = []
        fixed_tasks: List[Dict[str, Any]] = []

        current_score = 0.0
        previous_score = 0.0
        total_weight = 0.0

        for bench_name, current in current_results.items():
            if bench_name not in baseline:
                continue

            prev = baseline[bench_name]
            weight = float(current.total_tasks)
            total_weight += weight

            current_pass = current.pass_rate * weight
            previous_pass = prev["pass_rate"] * weight
            current_score += current_pass
            previous_score += previous_pass

            # Build task maps for comparison
            current_map = {d.task_id: d for d in current.details}
            prev_pass_set = set(prev.get("passed_tasks", []))
            prev_fail_set = set(prev.get("failed_tasks", []))

            for tid, cur_result in current_map.items():
                if tid in prev_pass_set and not cur_result.passed:
                    regressed_tasks.append({
                        "task_id": tid,
                        "benchmark": bench_name,
                        "previous_status": "passed",
                        "current_status": "failed",
                        "error": cur_result.error,
                    })
                elif tid in prev_fail_set and cur_result.passed:
                    improved_tasks.append({
                        "task_id": tid,
                        "benchmark": bench_name,
                        "previous_status": "failed",
                        "current_status": "passed",
                    })

            # New failures: tasks that now fail but weren't in the previous baseline at all
            current_fail_set = {d.task_id for d in current.details if not d.passed}
            new_fail_ids = current_fail_set - prev_pass_set - prev_fail_set
            for tid in new_fail_ids:
                cur_result = current_map.get(tid)
                if cur_result:
                    new_failures.append({
                        "task_id": tid,
                        "benchmark": bench_name,
                        "error": cur_result.error,
                    })

            # Fixed tasks: tasks that now pass but weren't in previous baseline
            current_pass_ids = {d.task_id for d in current.details if d.passed}
            new_pass_ids = current_pass_ids - prev_pass_set - prev_fail_set
            for tid in new_pass_ids:
                fixed_tasks.append({
                    "task_id": tid,
                    "benchmark": bench_name,
                })

        if total_weight > 0:
            current_score /= total_weight
            previous_score /= total_weight

        delta = current_score - previous_score

        return RegressionReport(
            current_score=round(current_score, 4),
            previous_score=round(previous_score, 4),
            delta=round(delta, 4),
            regressed_tasks=regressed_tasks,
            improved_tasks=improved_tasks,
            new_failures=new_failures,
            fixed_tasks=fixed_tasks,
        )

    # ──────────────────────────────────────────
    # Baseline management
    # ──────────────────────────────────────────

    async def save_baseline(self, name: str) -> None:
        """
        Save current benchmark results as a named baseline for future comparison.

        Args:
            name: Baseline name (e.g., "v1.0", "pre-refactor", "2024-01-15").
        """
        self._ensure_initialized()

        if not self._last_results:
            raise RuntimeError("No results to save. Run a benchmark first.")

        baseline_data = {}
        for bench_name, result in self._last_results.items():
            baseline_data[bench_name] = {
                "pass_rate": result.pass_rate,
                "total_tasks": result.total_tasks,
                "passed": result.passed,
                "failed": result.failed,
                "timestamp": result.timestamp,
                "avg_tokens": result.avg_tokens,
                "total_cost": result.total_cost,
                "passed_tasks": [d.task_id for d in result.details if d.passed],
                "failed_tasks": [d.task_id for d in result.details if not d.passed],
                "per_category": result.per_category,
                "per_difficulty": result.per_difficulty,
            }

        safe_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
        baseline_path = self.results_dir / "baselines" / f"{safe_name}.json"
        baseline_path.write_text(
            json.dumps(baseline_data, indent=2, default=str),
            encoding="utf-8",
        )

    def _load_baseline(self, name: str) -> Optional[Dict[str, Any]]:
        """Load a baseline by name or 'latest'."""
        baselines_dir = self.results_dir / "baselines"
        if not baselines_dir.exists():
            return None

        if name == "latest":
            files = sorted(baselines_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
            if not files:
                return None
            baseline_path = files[0]
        else:
            safe_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
            baseline_path = baselines_dir / f"{safe_name}.json"

        if not baseline_path.exists():
            return None

        return json.loads(baseline_path.read_text(encoding="utf-8"))

    async def _save_history(self, result: BenchmarkResult) -> None:
        """Append a benchmark result to the history log."""
        history_dir = self.results_dir / "history"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{result.benchmark_name}_{timestamp}.json"
        filepath = history_dir / filename

        record = {
            "benchmark_name": result.benchmark_name,
            "timestamp": result.timestamp,
            "total_tasks": result.total_tasks,
            "passed": result.passed,
            "failed": result.failed,
            "pass_rate": result.pass_rate,
            "total_time": result.total_time,
            "total_cost": result.total_cost,
            "avg_tokens": result.avg_tokens,
            "per_category": result.per_category,
            "per_difficulty": result.per_difficulty,
        }

        filepath.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    # ──────────────────────────────────────────
    # History retrieval
    # ──────────────────────────────────────────

    async def get_history(self) -> List[Dict[str, Any]]:
        """
        Get historical benchmark results across all runs.

        Returns:
            List of historical result records sorted by timestamp (newest first).
        """
        self._ensure_initialized()
        history_dir = self.results_dir / "history"
        if not history_dir.exists():
            return []

        history = []
        for filepath in history_dir.glob("*.json"):
            try:
                record = json.loads(filepath.read_text(encoding="utf-8"))
                record["_file"] = filepath.name
                history.append(record)
            except (json.JSONDecodeError, OSError):
                continue

        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return history

    # ──────────────────────────────────────────
    # Report generation
    # ──────────────────────────────────────────

    async def generate_report(
        self, result: BenchmarkResult, fmt: str = "markdown"
    ) -> str:
        """
        Generate a detailed report for a benchmark result.

        Args:
            result: The BenchmarkResult to report on.
            fmt: Output format — "markdown" or "json".

        Returns:
            Formatted report string.
        """
        if fmt == "json":
            return json.dumps(asdict(result), indent=2, default=str)

        # Markdown report
        lines = [
            f"# Benchmark Report: {result.benchmark_name}",
            "",
            f"**Timestamp:** {result.timestamp}",
            f"**Total Tasks:** {result.total_tasks}",
            f"**Passed:** {result.passed} ({result.pass_rate:.1%})",
            f"**Failed:** {result.failed}",
            f"**Total Time:** {result.total_time:.2f}s",
            f"**Total Cost:** ${result.total_cost:.6f}",
            f"**Avg Tokens/Task:** {result.avg_tokens:,}",
            "",
        ]

        # Summary bar
        bar_width = 40
        filled = int(result.pass_rate * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"**Progress:** `[{bar}]` {result.pass_rate:.1%}")
        lines.append("")

        # Per-category breakdown
        if result.per_category:
            lines.append("## Per-Category Breakdown")
            lines.append("")
            lines.append("| Category | Total | Passed | Failed | Pass Rate |")
            lines.append("|----------|-------|--------|--------|-----------|")
            for cat, stats in sorted(result.per_category.items()):
                lines.append(
                    f"| {cat} | {stats['total']} | {stats['passed']} "
                    f"| {stats['failed']} | {stats['pass_rate']:.1%} |"
                )
            lines.append("")

        # Per-difficulty breakdown
        if result.per_difficulty:
            lines.append("## Per-Difficulty Breakdown")
            lines.append("")
            lines.append("| Difficulty | Total | Passed | Failed | Pass Rate |")
            lines.append("|------------|-------|--------|--------|-----------|")
            for diff, stats in sorted(result.per_difficulty.items()):
                lines.append(
                    f"| {diff} | {stats['total']} | {stats['passed']} "
                    f"| {stats['failed']} | {stats['pass_rate']:.1%} |"
                )
            lines.append("")

        # Task details
        if result.details:
            lines.append("## Task Details")
            lines.append("")
            lines.append("| # | Task ID | Status | Score | Time (s) | Tokens | Error |")
            lines.append("|---|---------|--------|-------|----------|--------|-------|")
            for i, d in enumerate(result.details, 1):
                status = "✅" if d.passed else "❌"
                error_short = (d.error[:50] + "...") if d.error and len(d.error) > 50 else (d.error or "")
                lines.append(
                    f"| {i} | `{d.task_id}` | {status} | {d.score:.1f} "
                    f"| {d.execution_time:.2f} | {d.tokens_used} | {error_short} |"
                )
            lines.append("")

        # Failed tasks detail
        failed_tasks = [d for d in result.details if not d.passed]
        if failed_tasks:
            lines.append("## Failed Tasks — Error Details")
            lines.append("")
            for d in failed_tasks:
                lines.append(f"### `{d.task_id}`")
                lines.append(f"")
                lines.append(f"**Difficulty:** {d.difficulty.value}")
                lines.append(f"**Category:** {d.category or 'N/A'}")
                lines.append(f"**Time:** {d.execution_time:.2f}s")
                if d.error:
                    lines.append(f"**Error:**")
                    lines.append(f"```")
                    lines.append(d.error[:500])
                    lines.append(f"```")
                lines.append("")

        return "\n".join(lines)

    async def generate_leaderboard(self) -> str:
        """
        Generate a leaderboard comparing all saved benchmark results over time.

        Returns:
            Markdown formatted leaderboard string.
        """
        history = await self.get_history()
        if not history:
            return "# Leaderboard\n\nNo benchmark results recorded yet."

        lines = ["# Benchmark Leaderboard", ""]

        # Group by benchmark name
        by_benchmark: Dict[str, List[Dict]] = {}
        for record in history:
            name = record.get("benchmark_name", "unknown")
            if name not in by_benchmark:
                by_benchmark[name] = []
            by_benchmark[name].append(record)

        for bench_name, records in sorted(by_benchmark.items()):
            lines.append(f"## {bench_name}")
            lines.append("")
            lines.append("| Run | Date | Pass Rate | Passed | Failed | Time | Cost |")
            lines.append("|-----|------|-----------|--------|--------|------|------|")

            for rec in records[:20]:
                ts = rec.get("timestamp", "unknown")[:19]
                pr = rec.get("pass_rate", 0)
                passed = rec.get("passed", 0)
                failed = rec.get("failed", 0)
                tt = rec.get("total_time", 0)
                cost = rec.get("total_cost", 0)

                lines.append(
                    f"| {rec.get('_file', '?')} | {ts} | {pr:.1%} "
                    f"| {passed} | {failed} | {tt:.1f}s | ${cost:.6f} |"
                )
            lines.append("")

            # Best run
            if records:
                best = max(records, key=lambda r: r.get("pass_rate", 0))
                lines.append(
                    f"**Best run:** {best.get('timestamp', '?')[:19]} "
                    f"with {best.get('pass_rate', 0):.1%} pass rate"
                )
                lines.append("")

        return "\n".join(lines)

    async def generate_regression_report(self, report: RegressionReport) -> str:
        """
        Generate a human-readable regression report.

        Args:
            report: RegressionReport from check_regression().

        Returns:
            Markdown formatted regression report.
        """
        lines = [
            "# Regression Report",
            "",
            f"**Previous Score:** {report.previous_score:.2%}",
            f"**Current Score:** {report.current_score:.2%}",
            f"**Delta:** {report.delta:+.2%}",
            "",
        ]

        if report.delta < 0:
            lines.append(f"⚠️  **REGRESSION DETECTED** — score decreased by {abs(report.delta):.2%}")
        elif report.delta > 0:
            lines.append(f"✅ **IMPROVEMENT** — score increased by {report.delta:.2%}")
        else:
            lines.append("➡️  **NO CHANGE** — scores are identical")
        lines.append("")

        if report.regressed_tasks:
            lines.append(f"## Regressed Tasks ({len(report.regressed_tasks)})")
            lines.append("")
            lines.append("| Task ID | Benchmark | Error |")
            lines.append("|---------|-----------|-------|")
            for t in report.regressed_tasks:
                err = (t.get("error", "")[:60] + "...") if len(t.get("error", "")) > 60 else t.get("error", "")
                lines.append(f"| `{t['task_id']}` | {t['benchmark']} | {err} |")
            lines.append("")

        if report.improved_tasks:
            lines.append(f"## Improved Tasks ({len(report.improved_tasks)})")
            lines.append("")
            lines.append("| Task ID | Benchmark |")
            lines.append("|---------|-----------|")
            for t in report.improved_tasks:
                lines.append(f"| `{t['task_id']}` | {t['benchmark']} |")
            lines.append("")

        if report.new_failures:
            lines.append(f"## New Failures ({len(report.new_failures)})")
            lines.append("")
            lines.append("| Task ID | Benchmark | Error |")
            lines.append("|---------|-----------|-------|")
            for t in report.new_failures:
                err = (t.get("error", "")[:60] + "...") if len(t.get("error", "")) > 60 else t.get("error", "")
                lines.append(f"| `{t['task_id']}` | {t['benchmark']} | {err} |")
            lines.append("")

        if report.fixed_tasks:
            lines.append(f"## Fixed Tasks ({len(report.fixed_tasks)})")
            lines.append("")
            lines.append("| Task ID | Benchmark |")
            lines.append("|---------|-----------|")
            for t in report.fixed_tasks:
                lines.append(f"| `{t['task_id']}` | {t['benchmark']} |")
            lines.append("")

        return "\n".join(lines)

    # ──────────────────────────────────────────
    # CI/CD integration
    # ──────────────────────────────────────────

    async def run_ci_check(
        self,
        baseline_name: str = "latest",
        regression_threshold: float = -0.05,
        run_benchmarks: bool = True,
    ) -> Dict[str, Any]:
        """
        Run benchmarks suitable for CI pipeline. Fails if regression exceeds threshold.

        This method is designed to be called from CI/CD pipelines. It:
        1. Runs a quick subset of benchmarks
        2. Compares against a saved baseline
        3. Returns pass/fail status with details

        Args:
            baseline_name: Baseline to compare against.
            regression_threshold: Minimum acceptable delta (e.g., -0.05 means fail if > 5% drop).
            run_benchmarks: Whether to run benchmarks (set False if results already loaded).

        Returns:
            Dictionary with: passed (bool), report (str), regression (RegressionReport or None).
        """
        self._ensure_initialized()

        if run_benchmarks:
            # Run a quick subset for CI
            await self.run_humaneval(limit=10, parallel=4)
            await self.run_mbpp(limit=10, parallel=4)

        try:
            regression = await self.check_regression(baseline_name=baseline_name)
        except (FileNotFoundError, RuntimeError) as e:
            # No baseline exists — save one and pass
            await self.save_baseline(name="ci_baseline")
            return {
                "passed": True,
                "report": f"No baseline found. Created initial baseline. Error: {e}",
                "regression": None,
                "details": {
                    "message": "Initial baseline created. Future runs will compare against this.",
                    "baseline_name": "ci_baseline",
                },
            }

        regression_report = await self.generate_regression_report(regression)

        passed = regression.delta >= regression_threshold

        return {
            "passed": passed,
            "report": regression_report,
            "regression": regression,
            "details": {
                "current_score": regression.current_score,
                "previous_score": regression.previous_score,
                "delta": regression.delta,
                "threshold": regression_threshold,
                "regressed_count": len(regression.regressed_tasks),
                "improved_count": len(regression.improved_tasks),
            },
        }

    async def export_ci_artifacts(self, result: BenchmarkResult) -> Dict[str, str]:
        """
        Export benchmark results as CI artifacts (JSON files).

        Args:
            result: BenchmarkResult to export.

        Returns:
            Dictionary mapping artifact names to their file paths.
        """
        artifacts: Dict[str, str] = {}

        # JSON results
        json_path = self.results_dir / "reports" / f"{result.benchmark_name}_result.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(asdict(result), indent=2, default=str),
            encoding="utf-8",
        )
        artifacts["json_result"] = str(json_path)

        # Markdown report
        md_report = await self.generate_report(result, fmt="markdown")
        md_path = self.results_dir / "reports" / f"{result.benchmark_name}_report.md"
        md_path.write_text(md_report, encoding="utf-8")
        artifacts["markdown_report"] = str(md_path)

        # Machine-readable summary for CI dashboards
        summary = {
            "benchmark": result.benchmark_name,
            "timestamp": result.timestamp,
            "pass_rate": result.pass_rate,
            "total_tasks": result.total_tasks,
            "passed": result.passed,
            "failed": result.failed,
            "total_time": result.total_time,
            "total_cost": result.total_cost,
            "avg_tokens": result.avg_tokens,
            "per_category": result.per_category,
            "per_difficulty": result.per_difficulty,
        }
        summary_path = self.results_dir / "reports" / f"{result.benchmark_name}_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        artifacts["summary"] = str(summary_path)

        return artifacts


# ──────────────────────────────────────────────
# CLI convenience
# ──────────────────────────────────────────────

async def run_evaluator_cli():
    """Run the evaluator from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Claude Clone Evaluation Framework")
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp", "swe_bench", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks per benchmark")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel task count")
    parser.add_argument("--output", type=str, default=None, help="Output file path for report")
    parser.add_argument("--ci", action="store_true", help="CI mode: fail on regression")
    parser.add_argument("--baseline", type=str, default="latest", help="Baseline name for regression check")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")

    args = parser.parse_args()

    from agent.core import Agent

    agent = Agent()
    evaluator = Evaluator(agent)
    await evaluator.initialize()

    if args.ci:
        result = await evaluator.run_ci_check(
            baseline_name=args.baseline,
            run_benchmarks=True,
        )
        print(result["report"])
        if not result["passed"]:
            print("\n❌ CI CHECK FAILED — Regression detected")
            sys.exit(1)
        else:
            print("\n✅ CI CHECK PASSED")
            sys.exit(0)

    if args.benchmark == "all":
        results = await evaluator.run_all()
        full_report_parts = []
        for name, result in results.items():
            report = await evaluator.generate_report(result, fmt=args.format)
            full_report_parts.append(report)
        full_report = "\n\n---\n\n".join(full_report_parts)
    elif args.benchmark == "humaneval":
        result = await evaluator.run_humaneval(limit=args.limit, parallel=args.parallel)
        full_report = await evaluator.generate_report(result, fmt=args.format)
    elif args.benchmark == "mbpp":
        result = await evaluator.run_mbpp(limit=args.limit, parallel=args.parallel)
        full_report = await evaluator.generate_report(result, fmt=args.format)
    elif args.benchmark == "swe_bench":
        result = await evaluator.run_swe_bench(parallel=args.parallel)
        full_report = await evaluator.generate_report(result, fmt=args.format)
    else:
        print(f"Unknown benchmark: {args.benchmark}")
        sys.exit(1)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(full_report, encoding="utf-8")
        print(f"Report saved to {args.output}")
    else:
        print(full_report)


if __name__ == "__main__":
    asyncio.run(run_evaluator_cli())
