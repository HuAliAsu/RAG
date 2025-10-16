"""
اعتبارسنجی ورودی‌ها و تنظیمات
"""

import re
from typing import Tuple, List, Optional
from pathlib import Path


def validate_regex(pattern: str) -> Tuple[bool, str]:
    """
    بررسی صحت الگوی Regex

    Args:
        pattern: الگوی regex به صورت string

    Returns:
        (valid, error_message): tuple از bool و پیام خطا
    """
    if not pattern or not pattern.strip():
        return False, "الگو نمی‌تواند خالی باشد"

    try:
        re.compile(pattern)
        return True, ""
    except re.error as e:
        return False, f"Regex نامعتبر: {str(e)}"
    except Exception as e:
        return False, f"خطا در بررسی الگو: {str(e)}"


def validate_pattern_line(line: str) -> Tuple[bool, str, Optional[dict]]:
    """
    بررسی صحت یک خط الگو با فرمت: weight,priority,name,regex

    Args:
        line: خط الگو

    Returns:
        (valid, error_message, parsed_data): tuple از bool، پیام خطا و داده پارس شده
    """
    # حذف فضاهای خالی
    line = line.strip()

    # نادیده گرفتن خطوط خالی و کامنت‌ها
    if not line or line.startswith('#'):
        return True, "", None

    # تقسیم بر اساس کاما (حداکثر 4 بخش)
    parts = line.split(',', 3)

    if len(parts) != 4:
        return False, f"فرمت نادرست - باید 4 بخش داشته باشد: {line}", None

    try:
        # پارس weight
        weight = float(parts[0].strip())
        if not 0.1 <= weight <= 5.0:
            return False, f"weight باید بین 0.1 تا 5.0 باشد: {weight}", None

        # پارس priority
        priority = int(parts[1].strip())
        if not 1 <= priority <= 3:
            return False, f"priority باید 1، 2 یا 3 باشد: {priority}", None

        # پارس name
        name = parts[2].strip()
        if not name:
            return False, "name نمی‌تواند خالی باشد", None

        # پارس regex
        regex_str = parts[3].strip()
        is_valid, error = validate_regex(regex_str)
        if not is_valid:
            return False, f"در خط '{line}': {error}", None

        # ساخت دیکشنری
        parsed = {
            'weight': weight,
            'priority': priority,
            'name': name,
            'regex': regex_str,
            'original_line': line
        }

        return True, "", parsed

    except ValueError as e:
        return False, f"خطا در پارس مقادیر عددی: {str(e)}", None
    except Exception as e:
        return False, f"خطا در پارس خط '{line}': {str(e)}", None


def validate_config(config: dict) -> Tuple[bool, List[str]]:
    """
    بررسی صحت تنظیمات chunking

    Args:
        config: دیکشنری تنظیمات

    Returns:
        (valid, errors): tuple از bool و لیست خطاها
    """
    errors = []

    # بررسی وجود کلیدهای ضروری
    required_keys = ['min_chunk_size', 'max_chunk_size', 'overlap_size']
    for key in required_keys:
        if key not in config:
            errors.append(f"کلید ضروری '{key}' موجود نیست")

    if errors:
        return False, errors

    # بررسی اندازه chunk‌ها
    min_size = config.get('min_chunk_size', 0)
    max_size = config.get('max_chunk_size', 0)
    overlap = config.get('overlap_size', 0)

    if min_size <= 0:
        errors.append(f"min_chunk_size باید مثبت باشد: {min_size}")

    if max_size <= 0:
        errors.append(f"max_chunk_size باید مثبت باشد: {max_size}")

    if min_size >= max_size:
        errors.append(f"min_chunk_size ({min_size}) باید کوچکتر از max_chunk_size ({max_size}) باشد")

    if overlap < 0:
        errors.append(f"overlap_size نمی‌تواند منفی باشد: {overlap}")

    if overlap >= min_size:
        errors.append(f"overlap_size ({overlap}) باید کوچکتر از min_chunk_size ({min_size}) باشد")

    # بررسی آستانه‌ها (اختیاری)
    if 'coherence_threshold' in config:
        threshold = config['coherence_threshold']
        if not 0.0 <= threshold <= 1.0:
            errors.append(f"coherence_threshold باید بین 0 و 1 باشد: {threshold}")

    if 'similarity_threshold' in config:
        threshold = config['similarity_threshold']
        if not 0.0 <= threshold <= 1.0:
            errors.append(f"similarity_threshold باید بین 0 و 1 باشد: {threshold}")

    # بررسی الگوها (اگر موجود باشند)
    for pattern_type in ['structural_patterns', 'semantic_patterns', 'special_keywords']:
        if pattern_type in config and config[pattern_type]:
            patterns = config[pattern_type]
            if isinstance(patterns, str):
                patterns = patterns.strip().split('\n')

            for idx, line in enumerate(patterns, 1):
                is_valid, error, _ = validate_pattern_line(line)
                if not is_valid and error:  # خطوط خالی/کامنت valid هستند اما error ندارند
                    errors.append(f"{pattern_type} - خط {idx}: {error}")

    return len(errors) == 0, errors


def validate_file_path(file_path: str, check_exists: bool = True,
                       allowed_extensions: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    بررسی صحت مسیر فایل

    Args:
        file_path: مسیر فایل
        check_exists: بررسی وجود فایل
        allowed_extensions: لیست پسوندهای مجاز (مثل ['.docx', '.txt'])

    Returns:
        (valid, error_message): tuple از bool و پیام خطا
    """
    if not file_path or not file_path.strip():
        return False, "مسیر فایل نمی‌تواند خالی باشد"

    path = Path(file_path)

    # بررسی وجود
    if check_exists and not path.exists():
        return False, f"فایل یافت نشد: {file_path}"

    # بررسی نوع (باید فایل باشد، نه پوشه)
    if check_exists and not path.is_file():
        return False, f"مسیر به یک فایل اشاره نمی‌کند: {file_path}"

    # بررسی پسوند
    if allowed_extensions:
        if path.suffix.lower() not in allowed_extensions:
            return False, f"فرمت فایل پشتیبانی نمی‌شود. فرمت‌های مجاز: {', '.join(allowed_extensions)}"

    return True, ""


# تست‌های واحد
if __name__ == "__main__":
    print("🧪 تست validators.py\n")

    # تست regex
    print("1️⃣ تست validate_regex:")
    valid, error = validate_regex(r"^فصل\s+[\d۰-۹]+")
    print(f"   الگوی صحیح: {valid} - {error}")

    valid, error = validate_regex(r"^فصل[")  # نامعتبر
    print(f"   الگوی نامعتبر: {valid} - {error}\n")

    # تست pattern line
    print("2️⃣ تست validate_pattern_line:")
    valid, error, data = validate_pattern_line("5.0,1,chapter,^فصل\\s+[\\d۰-۹]+")
    print(f"   خط صحیح: {valid}")
    print(f"   داده: {data}\n")

    valid, error, data = validate_pattern_line("10.0,1,bad,^test")  # weight بالا
    print(f"   خط نامعتبر: {valid} - {error}\n")

    # تست config
    print("3️⃣ تست validate_config:")
    config = {
        'min_chunk_size': 200,
        'max_chunk_size': 800,
        'overlap_size': 50
    }
    valid, errors = validate_config(config)
    print(f"   Config صحیح: {valid}\n")

    bad_config = {
        'min_chunk_size': 800,
        'max_chunk_size': 200,
        'overlap_size': 50
    }
    valid, errors = validate_config(bad_config)
    print(f"   Config نامعتبر: {valid}")
    print(f"   خطاها: {errors}")