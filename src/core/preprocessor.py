"""
پیش‌پردازش و پاکسازی متن فارسی
"""

import re
import hazm
from typing import Dict, List
from pathlib import Path


class TextPreprocessor:
    """
    کلاس پیش‌پردازش متن فارسی
    """

    def __init__(self):
        """مقداردهی اولیه"""
        self.normalizer = hazm.Normalizer()
        self.word_tokenizer = hazm.WordTokenizer()
        self.sent_tokenizer = hazm.SentenceTokenizer()

    def clean_text(self, text: str, remove_old_tags: bool = True,
                   remove_metadata: bool = True) -> str:
        """
        نرمالیزه و پاکسازی کامل متن

        Args:
            text: متن ورودی
            remove_old_tags: حذف تگ‌های قبلی (@@-0000-@@)
            remove_metadata: حذف metadata قبلی

        Returns:
            str: متن پاک شده
        """
        if not text:
            return ""

        # مرحله 1: نرمالیزه‌سازی اولیه
        text = self.normalizer.normalize(text)

        # مرحله 2: حذف تگ‌های قبلی
        if remove_old_tags:
            # تگ‌های استاندارد: @@-0000-@@
            text = re.sub(r'@@-\d{4}-@@', '', text)

            # تگ‌های احتمالی دیگر
            text = re.sub(r'@@.*?@@', '', text)

            # خطوط جداکننده
            text = re.sub(r'^[─=\-*_]{20,}$', '', text, flags=re.MULTILINE)

        # مرحله 3: حذف metadata قبلی
        if remove_metadata:
            # metadata در براکت مربع: [کلمات: 450 | جملات: 12]
            text = re.sub(r'\[.*?\]', '', text)

            # metadata با pipe: کلمات: 450 | جملات: 12
            text = re.sub(r'(کلمات|جملات|Stage|اطمینان|کلیدواژه):.*?\|', '', text)

        # مرحله 4: پاکسازی فضاها
        # فضاهای متوالی
        text = re.sub(r' {2,}', ' ', text)

        # تب‌های متوالی
        text = re.sub(r'\t+', ' ', text)

        # خطوط خالی بیش از 2 خط
        text = re.sub(r'\n{3,}', '\n\n', text)

        # فضای خالی در ابتدا/انتهای خطوط
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)

        # مرحله 5: Trim نهایی
        text = text.strip()

        return text

    def extract_doc_info(self, text: str) -> Dict:
        """
        استخراج اطلاعات آماری از متن

        Args:
            text: متن ورودی

        Returns:
            dict: دیکشنری اطلاعات
        """
        if not text:
            return {
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'avg_words_per_sentence': 0.0,
                'avg_words_per_paragraph': 0.0
            }

        # شمارش کاراکترها
        char_count = len(text)

        # شمارش کلمات
        words = self.word_tokenizer.tokenize(text)
        word_count = len(words)

        # شمارش جملات
        try:
            sentences = self.sent_tokenizer.tokenize(text)
            sentence_count = len(sentences)
        except:
            # fallback ساده
            sentence_count = text.count('.') + text.count('!') + text.count('?')

        # شمارش پاراگراف‌ها
        paragraphs = [p for p in text.split('\n') if p.strip()]
        paragraph_count = len(paragraphs)

        # میانگین‌ها
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0.0
        avg_words_per_paragraph = word_count / paragraph_count if paragraph_count > 0 else 0.0

        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_words_per_sentence': round(avg_words_per_sentence, 1),
            'avg_words_per_paragraph': round(avg_words_per_paragraph, 1)
        }

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        تقسیم متن به پاراگراف‌ها

        Args:
            text: متن ورودی

        Returns:
            list: لیست پاراگراف‌ها (بدون خطوط خالی)
        """
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def clean_pattern_text(self, text: str) -> str:
        """
        پاکسازی متن الگوها (از TextArea)

        Args:
            text: متن الگوها

        Returns:
            str: متن پاک شده
        """
        # حذف فضاهای اضافی
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # حذف خطوط خالی و کامنت‌ها
            if line and not line.startswith('#'):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)


# تست
if __name__ == "__main__":
    print("🧪 تست preprocessor.py\n")

    preprocessor = TextPreprocessor()

    # متن نمونه
    sample_text = """
    @@-0001-@@

    این    یک   متن     تست است.


    با فضاهای    اضافی و خطوط    خالی.

    [کلمات: 450 | جملات: 12 | Stage: 1]

    ────────────────────────────────

    این پاراگراف دوم است.
    """

    print("1️⃣ متن اصلی:")
    print(repr(sample_text)[:100] + "...\n")

    # پاکسازی
    cleaned = preprocessor.clean_text(sample_text)
    print("2️⃣ متن پاک شده:")
    print(repr(cleaned) + "\n")

    # استخراج اطلاعات
    info = preprocessor.extract_doc_info(cleaned)
    print("3️⃣ اطلاعات متن:")
    for key, value in info.items():
        print(f"   {key}: {value}")