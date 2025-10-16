"""
سرویس ارتباط با Ollama برای Embedding
"""

import requests
import numpy as np
from typing import Optional, List
import time
from pathlib import Path


class OllamaEmbedder:
    """
    کلاس مدیریت ارتباط با Ollama
    """

    def __init__(self, model: str = "embeddinggemma",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        مقداردهی اولیه

        Args:
            model: نام مدل embedding
            base_url: آدرس سرور Ollama
            timeout: زمان انتظار (ثانیه)
            max_retries: تعداد تلاش مجدد
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

        # URLهای API
        self.embed_url = f"{self.base_url}/api/embeddings"
        self.tags_url = f"{self.base_url}/api/tags"

        # وضعیت اتصال
        self.available = False
        self.last_check_time = 0
        self.check_interval = 60  # بررسی هر 60 ثانیه

        # آمار
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_time': 0.0
        }

        # بررسی اولیه اتصال
        self.check_connection()

    def check_connection(self, force: bool = False) -> bool:
        """
        بررسی اتصال به Ollama

        Args:
            force: بررسی اجباری (بدون توجه به زمان)

        Returns:
            bool: وضعیت اتصال
        """
        current_time = time.time()

        # اگر اخیراً چک شده و force نیست، از cache استفاده کن
        if not force and (current_time - self.last_check_time) < self.check_interval:
            return self.available

        try:
            response = requests.get(self.tags_url, timeout=5)

            if response.status_code == 200:
                # بررسی وجود مدل
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]

                if self.model in models:
                    self.available = True
                    from src.utils.logger import info
                    info(f"✅ اتصال به Ollama برقرار - مدل {self.model} موجود است")
                else:
                    self.available = False
                    from src.utils.logger import warning
                    warning(f"⚠️ مدل {self.model} در Ollama یافت نشد. مدل‌های موجود: {', '.join(models)}")
            else:
                self.available = False
                from src.utils.logger import warning
                warning(f"⚠️ Ollama پاسخ نامعتبر داد: {response.status_code}")

        except requests.exceptions.ConnectionError:
            self.available = False
            from src.utils.logger import warning
            warning("⚠️ Ollama در دسترس نیست - آیا سرویس در حال اجرا است؟")

        except requests.exceptions.Timeout:
            self.available = False
            from src.utils.logger import warning
            warning("⚠️ Timeout در اتصال به Ollama")

        except Exception as e:
            self.available = False
            from src.utils.logger import error
            error(f"❌ خطا در بررسی اتصال Ollama: {str(e)}")

        self.last_check_time = current_time
        return self.available

    def get_embedding(self, text: str, retry_count: int = 0) -> Optional[np.ndarray]:
        """
        دریافت embedding برای یک متن

        Args:
            text: متن ورودی
            retry_count: تعداد دفعات تلاش شده

        Returns:
            np.ndarray یا None: بردار embedding
        """
        if not text or not text.strip():
            from src.utils.logger import warning
            warning("⚠️ متن خالی برای embedding")
            return None

        # بررسی اتصال
        if not self.available:
            self.check_connection(force=True)
            if not self.available:
                return None

        start_time = time.time()
        self.stats['total_calls'] += 1

        try:
            # ارسال درخواست
            response = requests.post(
                self.embed_url,
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=self.timeout
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result['embedding'], dtype=np.float32)

                # بروزرسانی آمار
                self.stats['successful_calls'] += 1
                self.stats['total_time'] += elapsed_time

                from src.utils.logger import debug
                debug(f"✅ Embedding دریافت شد - dimension: {len(embedding)}, time: {elapsed_time:.2f}s")

                return embedding

            else:
                from src.utils.logger import error
                error(f"❌ خطا در دریافت embedding: HTTP {response.status_code}")

                # تلاش مجدد
                if retry_count < self.max_retries:
                    from src.utils.logger import info
                    info(f"🔄 تلاش مجدد {retry_count + 1}/{self.max_retries}...")
                    time.sleep(1 * (retry_count + 1))  # exponential backoff
                    return self.get_embedding(text, retry_count + 1)

                self.stats['failed_calls'] += 1
                return None

        except requests.exceptions.Timeout:
            from src.utils.logger import error
            error(f"❌ Timeout در دریافت embedding (>{self.timeout}s)")

            if retry_count < self.max_retries:
                return self.get_embedding(text, retry_count + 1)

            self.stats['failed_calls'] += 1
            return None

        except Exception as e:
            from src.utils.logger import error
            error(f"❌ خطا در دریافت embedding: {str(e)}")

            if retry_count < self.max_retries:
                return self.get_embedding(text, retry_count + 1)

            self.stats['failed_calls'] += 1
            return None

    def get_embeddings_batch(self, texts: List[str],
                             show_progress: bool = False) -> List[Optional[np.ndarray]]:
        """
        دریافت embedding برای لیستی از متن‌ها

        Args:
            texts: لیست متن‌ها
            show_progress: نمایش پیشرفت

        Returns:
            list: لیست embedding‌ها (ممکن است برخی None باشند)
        """
        embeddings = []
        total = len(texts)

        from src.utils.logger import info
        if show_progress:
            info(f"📊 شروع دریافت {total} embedding...")

        for idx, text in enumerate(texts, 1):
            if show_progress and idx % 10 == 0:
                info(f"   پیشرفت: {idx}/{total} ({idx * 100 // total}%)")

            emb = self.get_embedding(text)
            embeddings.append(emb)

        if show_progress:
            successful = sum(1 for e in embeddings if e is not None)
            info(f"✅ {successful}/{total} embedding با موفقیت دریافت شد")

        return embeddings

    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        محاسبه شباهت کسینوسی بین دو embedding

        Args:
            emb1: embedding اول
            emb2: embedding دوم

        Returns:
            float: شباهت (0 تا 1)
        """
        if emb1 is None or emb2 is None:
            return 0.0

        # نرمالیزه کردن
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # cosine similarity
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)

        # تبدیل به بازه [0, 1]
        similarity = (similarity + 1) / 2

        return float(similarity)

    def get_stats(self) -> dict:
        """
        دریافت آمار استفاده

        Returns:
            dict: دیکشنری آمار
        """
        stats = self.stats.copy()

        if stats['successful_calls'] > 0:
            stats['avg_time'] = stats['total_time'] / stats['successful_calls']
        else:
            stats['avg_time'] = 0.0

        if stats['total_calls'] > 0:
            stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
        else:
            stats['success_rate'] = 0.0

        return stats

    def reset_stats(self):
        """صفر کردن آمار"""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_time': 0.0
        }


# تست
if __name__ == "__main__":
    print("🧪 تست embedder.py\n")

    # ایجاد نمونه
    embedder = OllamaEmbedder()

    # بررسی اتصال
    print(f"1️⃣ وضعیت اتصال: {'✅ متصل' if embedder.available else '❌ قطع'}\n")

    if embedder.available:
        # تست embedding تکی
        print("2️⃣ تست embedding تکی:")
        emb1 = embedder.get_embedding("این یک متن تست است")
        if emb1 is not None:
            print(f"   ✅ Embedding دریافت شد - dimension: {len(emb1)}\n")

        # تست شباهت
        print("3️⃣ تست شباهت:")
        emb2 = embedder.get_embedding("این متن مشابه است")
        emb3 = embedder.get_embedding("موضوع کاملاً متفاوت")

        if emb1 is not None and emb2 is not None and emb3 is not None:
            sim_12 = embedder.calculate_similarity(emb1, emb2)
            sim_13 = embedder.calculate_similarity(emb1, emb3)
            print(f"   شباهت متن 1 و 2: {sim_12:.3f}")
            print(f"   شباهت متن 1 و 3: {sim_13:.3f}\n")

        # آمار
        print("4️⃣ آمار:")
        stats = embedder.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
    else:
        print("⚠️ لطفاً Ollama را راه‌اندازی کنید:")
        print("   $ ollama serve")
        print(f"   $ ollama pull {embedder.model}")