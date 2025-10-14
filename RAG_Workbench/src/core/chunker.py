import re
from typing import List, Dict, Any

from src.utils.logger import logger
from src.core.embedder import ollama_embedder
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer

# --- Default Persian Patterns ---

STRUCTURAL_PATTERNS_DEFAULT = """
# عناوین فصل و بخش - اولویت بالا
5.0,1,chapter_title,^(فصل|بخش|قسمت)\\s+[\\d۰-۹]+
4.5,1,chapter_named,^(فصل|بخش)\\s+[\\d۰-۹]+\\s*[:-]\\s*.+
3.5,1,numbered_section,^[\\d۰-۹]+[\\.\\)]\\s+[^\\n]{5,50}$
# داستان‌ها و حکایت‌ها
4.0,1,story_title,^(داستان|حکایت|قصه)\\s+[\\d۰-۹]+
3.5,1,story_named,^(داستان|حکایت)\\s*[:-]\\s*.+
# شماره‌گذاری‌های ترتیبی
3.0,1,ordinal_title,^(اول|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم|یازدهم|دوازدهم)\\s*[-:]\\s*
# جداکننده‌های بصری
3.5,1,visual_separator,^[=\\-*]{3,}\\s*$
# الگوی عنوان کوتاه در خط مجزا (احتمالی)
2.0,2,short_isolated_title,^.{5,40}$
"""

SEMANTIC_PATTERNS_DEFAULT = """
# الگوهای شروع داستان - اولویت متوسط
2.5,2,story_opening_classic,روزی\\s+روزگاری
2.5,2,story_opening_folklore,یکی\\s+بود\\s+یکی\\s+نبود
2.0,2,story_opening_context,در\\s+(زمانی|روزی|شهری|دیاری|سرزمینی)\\s+که
2.0,2,story_opening_simple,یک\\s+روز
1.8,2,story_opening_person,(مردی|زنی|پیرمردی|جوانی)\\s+بود\\s+که
# الگوهای پایان بخش
2.5,2,lesson_marker,(درس|نکته|آموخته|عبرت)\\s*[:-]\\s*
2.3,2,lesson_phrase,درس\\s+این\\s+(داستان|حکایت|قصه)
2.0,2,conclusion_phrase,به\\s+این\\s+ترتیب
1.8,2,summary_phrase,خلاصه\\s+(اینکه|آنکه)
# نشانه‌های گفتمان (Discourse Markers)
2.0,2,discourse_transition,^(حال|اکنون|بگذارید|بیایید)\\s+
1.8,2,discourse_contrast,^(اما|ولی|با\\s+این\\s+حال|در\\s+مقابل)\\s+
1.5,2,discourse_addition,^(همچنین|علاوه\\s+بر\\s+این|ضمناً)\\s+
1.5,2,discourse_example,^(برای\\s+مثال|مثلاً|از\\s+جمله)\\s+
# نشانه‌های تغییر زمان/مکان
1.8,2,time_shift,(پس\\s+از|بعد\\s+از)\\s+(مدتی|چندی|سال‌ها|روزها)
1.8,2,location_shift,در\\s+(شهری|روستایی|مکانی)\\s+دیگر
"""

SPECIAL_KEYWORDS_DEFAULT = """
# کلمات کلیدی که نشان‌دهنده مرز قوی هستند
3.0,1,section_end,پایان\\s+(فصل|بخش|داستان)
2.5,1,new_topic,موضوع\\s+جدید
2.5,1,question_marker,پرسش\\s*[:-]\\s*
2.0,2,reflection,تأمل\\s+در\\s+این\\s+(موضوع|مطلب)
2.0,2,exercise,تمرین\\s*[:-]\\s*
# کلمات شخصیت‌ها (برای تشخیص تغییر شخصیت)
1.5,3,character_intro,(مردی|زنی|پیرمردی|جوانی|کودکی)\\s+به\\s+نام
1.5,3,character_role,(استاد|معلم|شاگرد|کشاورز|تاجر|پادشاه)\\s+
"""


class OptimizedHybridChunker:
    """
    Implements a three-stage hybrid chunking strategy for Persian documents.
    """
    def __init__(self, config: dict):
        logger.info("Initializing OptimizedHybridChunker...")
        self.config = config
        self.min_chunk_size = config.get('min_chunk_size', 200)
        self.max_chunk_size = config.get('max_chunk_size', 800)
        self.overlap_size = config.get('overlap_size', 50)
        self.coherence_threshold = config.get('coherence_threshold', 0.15)
        self.similarity_threshold = config.get('similarity_threshold', 0.75)

        # Parse all pattern types from the config
        self.structural_patterns = self._parse_patterns(config.get('structural_patterns', STRUCTURAL_PATTERNS_DEFAULT))
        self.semantic_patterns = self._parse_patterns(config.get('semantic_patterns', SEMANTIC_PATTERNS_DEFAULT))
        self.special_keywords = self._parse_patterns(config.get('special_keywords', SPECIAL_KEYWORDS_DEFAULT))

        self.all_patterns = self.structural_patterns + self.semantic_patterns + self.special_keywords
        logger.info(f"Loaded {len(self.all_patterns)} patterns in total.")

        self.embedder = ollama_embedder
        # self.vectorizer = TfidfVectorizer()

    def _parse_patterns(self, pattern_string: str) -> List[Dict[str, Any]]:
        """Parses the multiline pattern string into a list of structured dicts."""
        parsed = []
        for line in pattern_string.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(',', 3)
            if len(parts) != 4:
                logger.warning(f"Skipping malformed pattern line: {line}")
                continue

            try:
                weight = float(parts[0])
                priority = int(parts[1])
                name = parts[2].strip()
                regex_pattern = parts[3].strip()

                parsed.append({
                    'weight': weight,
                    'priority': priority,
                    'name': name,
                    'regex': re.compile(regex_pattern, re.MULTILINE)
                })
            except (ValueError, re.error) as e:
                logger.error(f"Failed to parse pattern line '{line}': {e}")

        return parsed

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Orchestrates the 3-stage chunking process.
        This is a placeholder for the full, complex logic.
        """
        logger.info("Starting 3-stage chunking process...")

        # Stage 1: Find all potential boundaries using regex patterns
        boundaries = self._find_pattern_based_boundaries(text)
        logger.info(f"Stage 1 found {len(boundaries)} potential boundaries.")

        # Stages 2 & 3 will be developed here to refine boundaries
        # For now, we will use a simplified logic

        # Create initial chunks based on definitive boundaries (Priority 1)
        definitive_boundaries = [b['pos'] for b in boundaries if b['pattern']['priority'] == 1]
        split_points = sorted(list(set([0] + definitive_boundaries + [len(text)])))

        raw_chunks = []
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i+1]
            chunk_text = text[start:end].strip()
            if chunk_text:
                raw_chunks.append({'text': chunk_text, 'start_char': start})

        # Post-processing: Merge, Split, and Overlap
        final_chunks = self._post_process_chunks(raw_chunks)

        logger.info(f"Chunking complete. Produced {len(final_chunks)} final chunks.")
        return final_chunks

    def _find_pattern_based_boundaries(self, text: str) -> List[Dict[str, Any]]:
        """Stage 1: Find boundaries based on all regex patterns."""
        boundaries = []
        for pattern in self.all_patterns:
            for match in pattern['regex'].finditer(text):
                boundaries.append({
                    'pos': match.start(),
                    'pattern': pattern,
                    'match_text': match.group(0)
                })
        return sorted(boundaries, key=lambda x: x['pos'])

    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handles merging small chunks, splitting large ones, and adding overlap.
        This is a simplified implementation.
        """
        # 1. Merge small chunks
        processed_chunks = []
        buffer = ""
        for chunk in chunks:
            text = chunk['text']
            if len(text.split()) < self.min_chunk_size and buffer:
                buffer += "\n\n" + text
            elif len(text.split()) < self.min_chunk_size:
                 buffer = text
            else:
                if buffer:
                    processed_chunks.append({'text': buffer})
                    buffer = ""
                processed_chunks.append({'text': text})
        if buffer:
            processed_chunks.append({'text': buffer})

        # 2. Split large chunks (simple split for now)
        final_chunks = []
        for chunk in processed_chunks:
            words = chunk['text'].split()
            if len(words) > self.max_chunk_size:
                # Simple split by word count
                for i in range(0, len(words), self.max_chunk_size):
                     final_chunks.append({'text': ' '.join(words[i:i+self.max_chunk_size])})
            else:
                final_chunks.append(chunk)

        # 3. Add overlap
        for i in range(len(final_chunks) - 1):
            current_chunk_words = final_chunks[i]['text'].split()
            next_chunk_words = final_chunks[i+1]['text'].split()

            if len(current_chunk_words) > self.overlap_size:
                overlap_text = ' '.join(current_chunk_words[-self.overlap_size:])
                final_chunks[i+1]['text'] = overlap_text + "\n\n" + final_chunks[i+1]['text']
                final_chunks[i+1]['has_overlap_prev'] = True
                final_chunks[i]['has_overlap_next'] = True

        return final_chunks