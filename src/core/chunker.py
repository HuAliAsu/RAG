"""
الگوریتم چانکینگ سه مرحله‌ای با سیستم دو لایه
Parent-Child Chunking Strategy
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hazm


class OptimizedHybridChunker:
    """
    کلاس اصلی چانکینگ سه مرحله‌ای با قابلیت Parent-Child

    Stage 1: Pattern-Based Detection (سریع)
    Stage 2: Coherence Analysis (متوسط)
    Stage 3: Semantic Embedding (دقیق)
    """

    def __init__(self, config: Dict):
        """
        مقداردهی اولیه

        Args:
            config: دیکشنری تنظیمات شامل:
                - structural_patterns: الگوهای ساختاری
                - semantic_patterns: الگوهای معنایی
                - special_keywords: کلمات کلیدی
                - min_chunk_size: حداقل کلمات
                - max_chunk_size: حداکثر کلمات
                - overlap_size: کلمات overlap
                - coherence_threshold: آستانه coherence
                - similarity_threshold: آستانه similarity
                - enable_stage1/2/3: فعال/غیرفعال Stage‌ها
                - enable_parent_child: فعال‌سازی سیستم دو لایه
        """
        self.config = config

        # تنظیمات اندازه
        self.min_chunk_size = config.get('min_chunk_size', 150)
        self.max_chunk_size = config.get('max_chunk_size', 800)
        self.overlap_size = config.get('overlap_size', 50)

        # تنظیمات Stage‌ها
        self.enable_stage1 = config.get('enable_stage1', True)
        self.enable_stage2 = config.get('enable_stage2', True)
        self.enable_stage3 = config.get('enable_stage3', True)

        # آستانه‌ها
        self.coherence_threshold = config.get('coherence_threshold', 0.15)
        self.similarity_threshold = config.get('similarity_threshold', 0.75)

        # سیستم دو لایه
        self.enable_parent_child = config.get('enable_parent_child', True)
        self.child_chunk_size = config.get('child_chunk_size', 100)

        # ابزارهای پردازش متن
        self.normalizer = hazm.Normalizer()
        self.word_tokenizer = hazm.WordTokenizer()
        self.sent_tokenizer = hazm.SentenceTokenizer()

        # TF-IDF vectorizer برای Stage 2
        self.vectorizer = TfidfVectorizer(max_features=100)

        # Embedder (از فاز 1)
        from src.core.embedder import OllamaEmbedder
        self.embedder = OllamaEmbedder(
            model=config.get('embedding_model', 'embeddinggemma:latest')
        )

        # الگوها
        self.patterns = self._parse_patterns(config)

        # آمار
        self.stats = {
            'stage1_boundaries': 0,
            'stage2_boundaries': 0,
            'stage3_boundaries': 0,
            'embedding_calls': 0,
            'start_time': 0,
            'end_time': 0
        }

    def _parse_patterns(self, config: Dict) -> Dict[str, List[Dict]]:
        """پارس الگوها از config"""
        from src.utils.validators import validate_pattern_line
        from src.utils.logger import debug, warning

        patterns = {
            'structural': [],
            'semantic': [],
            'special': []
        }

        # پارس هر نوع الگو
        for pattern_type, config_key in [
            ('structural', 'structural_patterns'),
            ('semantic', 'semantic_patterns'),
            ('special', 'special_keywords')
        ]:
            pattern_text = config.get(config_key, '')

            if isinstance(pattern_text, str):
                lines = pattern_text.strip().split('\n')
            elif isinstance(pattern_text, list):
                lines = pattern_text
            else:
                continue

            for line in lines:
                is_valid, error, parsed = validate_pattern_line(line)

                if is_valid and parsed:
                    # کامپایل regex
                    try:
                        parsed['regex_compiled'] = re.compile(parsed['regex'])
                        patterns[pattern_type].append(parsed)
                        debug(f"   الگوی {pattern_type}: {parsed['name']}")
                    except re.error as e:
                        warning(f"   خطا در کامپایل regex '{parsed['name']}': {e}")

        return patterns

    def chunk_text(self, text: str, headings: Optional[List[Dict]] = None,
                   source_metadata: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
        """
        متد اصلی چانکینگ

        Args:
            text: متن ورودی
            headings: لیست عناوین (از Word)
            source_metadata: metadata فایل اصلی

        Returns:
            tuple: (chunks, processing_stats)
        """
        from src.utils.logger import info, debug
        from src.core.preprocessor import TextPreprocessor

        self.stats['start_time'] = time.time()

        info("\n" + "="*70)
        info("🔄 شروع Chunking سه مرحله‌ای")
        info("="*70)

        # پیش‌پردازش
        preprocessor = TextPreprocessor()
        text = preprocessor.clean_text(text, remove_old_tags=True)

        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        info(f"📝 تعداد پاراگراف‌ها: {len(paragraphs)}")

        # مرحله 1: تجمیع مرزها
        all_boundaries = []

        # افزودن Heading‌ها
        if headings:
            info(f"\n📑 {len(headings)} عنوان Heading شناسایی شد")
            for heading in headings:
                all_boundaries.append({
                    'index': heading['para_index'],
                    'score': 5.0,
                    'confidence': 1.0,
                    'text': heading['text'],
                    'signals': ['heading'],
                    'stage': 0,
                    'stage_name': 'Heading',
                    'heading_level': heading['level'],
                    'priority': 1
                })

        # Stage 1: Pattern Detection
        if self.enable_stage1:
            stage1_boundaries = self._stage1_pattern_detection(paragraphs)
            all_boundaries.extend(stage1_boundaries['confirmed'])
            uncertain_boundaries = stage1_boundaries['uncertain']
        else:
            info("\n⏭️ Stage 1 غیرفعال است")
            uncertain_boundaries = []

        # Stage 2: Coherence Analysis
        if self.enable_stage2 and uncertain_boundaries:
            stage2_boundaries = self._stage2_coherence_analysis(paragraphs, uncertain_boundaries)
            all_boundaries.extend(stage2_boundaries['confirmed'])
            uncertain_boundaries = stage2_boundaries['uncertain']
        elif not self.enable_stage2:
            info("\n⏭️ Stage 2 غیرفعال است")

        # Stage 3: Semantic Embedding
        if self.enable_stage3 and uncertain_boundaries and self.embedder.available:
            stage3_boundaries = self._stage3_semantic_verification(paragraphs, uncertain_boundaries)
            all_boundaries.extend(stage3_boundaries)
        elif not self.enable_stage3:
            info("\n⏭️ Stage 3 غیرفعال است")
        elif not self.embedder.available:
            info("\n⚠️ Stage 3: Ollama در دسترس نیست")

        # مرتب‌سازی
        all_boundaries.sort(key=lambda x: x['index'])

        # ساخت chunk‌ها
        info("\n✂️ ساخت chunk‌ها...")
        chunks = self._create_chunks_from_boundaries(
            paragraphs, all_boundaries, source_metadata
        )

        # سیستم Parent-Child
        if self.enable_parent_child:
            info("\n👨‍👦 ساخت ساختار Parent-Child...")
            chunks = self._create_parent_child_structure(chunks)

        # آمار نهایی
        self.stats['end_time'] = time.time()
        self.stats['total_time'] = self.stats['end_time'] - self.stats['start_time']
        self.stats['total_chunks'] = len(chunks)

        self._print_stats()

        processing_stats = self.stats.copy()

        return chunks, processing_stats

