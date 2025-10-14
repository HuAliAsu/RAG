import re
from typing import Tuple, List, Dict, Any

def validate_regex_pattern(pattern: str) -> Tuple[bool, str]:
    """
    Validates a single regex pattern.

    Args:
        pattern (str): The regex pattern to validate.

    Returns:
        A tuple containing:
        - bool: True if the pattern is valid, False otherwise.
        - str: An empty string if valid, or an error message if invalid.
    """
    if not pattern:
        return False, "الگو نمی‌تواند خالی باشد."
    try:
        re.compile(pattern)
        return True, ""
    except re.error as e:
        return False, f"الگوی Regex نامعتبر است: {e}"

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates the main processing configuration dictionary.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        A tuple containing:
        - bool: True if the configuration is valid, False otherwise.
        - list[str]: A list of error messages if invalid, otherwise an empty list.
    """
    errors = []

    # Chunk size validation
    min_size = config.get('min_chunk_size', 200)
    max_size = config.get('max_chunk_size', 800)
    overlap = config.get('overlap_size', 50)

    if min_size >= max_size:
        errors.append(f"حداقل اندازه چانک ({min_size}) باید کمتر از حداکثر ({max_size}) باشد.")

    if overlap >= min_size:
        errors.append(f"اندازه همپوشانی ({overlap}) باید کمتر از حداقل اندازه چانک ({min_size}) باشد.")

    # Threshold validation
    coherence_threshold = config.get('coherence_threshold', 0.15)
    if not 0.0 <= coherence_threshold <= 1.0:
        errors.append(f"آستانه تغییر موضوع ({coherence_threshold}) باید بین 0.0 و 1.0 باشد.")

    similarity_threshold = config.get('similarity_threshold', 0.75)
    if not 0.0 <= similarity_threshold <= 1.0:
        errors.append(f"آستانه شباهت ({similarity_threshold}) باید بین 0.0 و 1.0 باشد.")

    # Pattern validation (check if at least one is active)
    has_patterns = False
    if config.get('structural_patterns'): has_patterns = True
    if config.get('semantic_patterns'): has_patterns = True
    if config.get('special_keywords'): has_patterns = True

    if not has_patterns:
        errors.append("حداقل یک مجموعه الگو (ساختاری، معنایی یا کلمات کلیدی) باید فعال باشد.")

    is_valid = not errors
    return is_valid, errors