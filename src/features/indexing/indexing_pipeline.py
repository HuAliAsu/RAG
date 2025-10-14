import os
import hashlib
import time
from typing import Dict, Any, Callable

try:
    import docx
except ImportError:
    docx = None

from src.utils.logger import logger
from src.core.preprocessor import TextPreprocessor
from src.core.chunker import OptimizedHybridChunker
from src.core.embedder import ollama_embedder
from src.core.vector_db import ChromaDBManager

# Define callback types for clarity
ProgressCallback = Callable[[float, str], None]
LogCallback = Callable[[str, str], None]
CompletionCallback = Callable[[Dict[str, Any]], None]

class IndexingPipeline:
    """
    Orchestrates the entire indexing process from file reading to vector storage.
    """
    def __init__(self, config: Dict[str, Any], callbacks: Dict[str, Callable]):
        self.config = config
        self.callbacks = callbacks
        self.preprocessor = TextPreprocessor()
        self.chunker = OptimizedHybridChunker(config)
        self.vector_db = ChromaDBManager()

    def _on_progress(self, percentage: float, message: str):
        if 'on_progress' in self.callbacks:
            self.callbacks['on_progress'](percentage, message)
        logger.info(f"Progress {percentage}%: {message}")

    def _on_log(self, message: str, level: str = 'info'):
        if 'on_log' in self.callbacks:
            self.callbacks['on_log'](message, level)
        else: # Fallback to standard logger
            if level == 'error': logger.error(message)
            elif level == 'warning': logger.warning(message)
            else: logger.info(message)

    def run(self, input_file: str):
        """
        Executes the full indexing pipeline.
        """
        try:
            self._on_progress(0, "شروع فرآیند ایندکسینگ...")
            start_time = time.time()

            # 1. Read and Preprocess Text
            self._on_progress(5, f"خواندن فایل: {os.path.basename(input_file)}")
            text = self._read_docx(input_file)
            if not text:
                self._on_log(f"فایل '{input_file}' خالی است یا قابل خواندن نیست.", "error")
                return

            self._on_progress(10, "پیش‌پردازش متن...")
            clean_text = self.preprocessor.clean_text(text)
            doc_hash = hashlib.sha256(clean_text.encode('utf-8')).hexdigest()
            self._on_log(f"✓ متن با موفقیت پیش‌پردازش شد. هش: {doc_hash[:10]}...")

            # 2. Chunking
            self._on_progress(20, "مرحله ۱: اجرای چانکینگ...")
            chunks = self.chunker.chunk_text(clean_text)
            if not chunks:
                self._on_log("چانکینگ هیچ متنی تولید نکرد.", "error")
                return
            self._on_log(f"✓ چانکینگ کامل شد. {len(chunks)} چانک تولید شد.")

            # 3. Generate Embeddings
            self._on_progress(60, "مرحله ۲: تولید Embedding برای چانک‌ها...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = ollama_embedder.get_embeddings(chunk_texts)
            if not embeddings or len(embeddings) != len(chunks):
                self._on_log("تولید Embedding ناموفق بود.", "error")
                return
            self._on_log(f"✓ {len(embeddings)} امبدینگ با موفقیت تولید شد.")

            # 4. Generate Metadata and Prepare for DB
            self._on_progress(85, "مرحله ۳: آماده‌سازی فراداده...")
            documents, metadatas, ids = [], [], []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{os.path.basename(input_file)}_{doc_hash[:8]}_{i:04d}"
                metadata = self._create_chunk_metadata(chunk, input_file, doc_hash, i)

                documents.append(chunk['text'])
                metadatas.append(metadata)
                ids.append(chunk_id)
            self._on_log("✓ فراداده برای تمام چانک‌ها تولید شد.")

            # 5. Index in Vector DB
            self._on_progress(90, "مرحله ۴: ذخیره در پایگاه داده وکتور...")
            collection_name = os.path.splitext(os.path.basename(input_file))[0].replace(" ", "_")
            success = self.vector_db.add_to_collection(
                collection_name, embeddings, documents, metadatas, ids
            )
            if not success:
                self._on_log("ذخیره در پایگاه داده ناموفق بود.", "error")
                return
            self._on_log(f"✓ داده‌ها با موفقیت در کالکشن '{collection_name}' ذخیره شدند.")

            # 6. Final Report
            end_time = time.time()
            final_report = {
                "total_time": round(end_time - start_time, 2),
                "chunk_count": len(chunks),
                "collection_name": collection_name
            }
            if 'on_completion' in self.callbacks:
                self.callbacks['on_completion'](final_report)

            self._on_progress(100, "✅ فرآیند ایندکسینگ با موفقیت کامل شد.")

        except Exception as e:
            self._on_log(f"خطای بحرانی در پایپ‌لاین ایندکسینگ: {e}", "error")
            logger.error("Indexing pipeline failed.", exc_info=True)
            self._on_progress(100, "❌ فرآیند با خطا متوقف شد.")

    def _read_docx(self, file_path: str) -> str:
        """Reads text from a .docx file."""
        if not docx:
            self._on_log("کتابخانه python-docx نصب نشده است.", "error")
            return ""
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            self._on_log(f"خطا در خواندن فایل DOCX: {e}", "error")
            return ""

    def _create_chunk_metadata(self, chunk_data: Dict, source_doc: str, doc_hash: str, index: int) -> Dict:
        """Creates the rich metadata dictionary for a single chunk."""
        text = chunk_data['text']
        return {
            "chunk_id": f"{os.path.basename(source_doc)}_{index:04d}",
            "source_document": os.path.basename(source_doc),
            "document_hash": doc_hash,
            "start_char": chunk_data.get('start_char', -1),
            "word_count": len(text.split()),
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # More fields to be added from chunker results
            "boundary_info": chunk_data.get("boundary_info", "N/A"),
        }

def run_indexing_pipeline(input_file: str, config: Dict[str, Any], callbacks: Dict[str, Callable]):
    """
    Entry point function to run the indexing pipeline.
    This function is designed to be run in a separate thread from the UI.
    """
    pipeline = IndexingPipeline(config, callbacks)
    pipeline.run(input_file)