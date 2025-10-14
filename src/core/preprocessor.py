import re
try:
    import hazm
except ImportError:
    print("Error: The 'hazm' library is not installed. Please install it using 'pip install hazm'")
    hazm = None

from src.utils.logger import logger

class TextPreprocessor:
    """
    Handles intelligent preprocessing of Persian text for the RAG pipeline.
    """
    def __init__(self):
        if hazm:
            self.normalizer = hazm.Normalizer()
            logger.info("TextPreprocessor initialized with hazm normalizer.")
        else:
            self.normalizer = None
            logger.warning("TextPreprocessor initialized without hazm normalizer. Normalization will be skipped.")

    def clean_text(self, text: str, remove_old_tags: bool = True) -> str:
        """
        Performs a series of cleaning steps on the input text.
        1. Normalization (if hazm is available).
        2. Removal of previous processing tags.
        3. Whitespace and newline cleanup.
        """
        logger.info(f"Starting text cleaning. Original length: {len(text)} chars.")

        # 1. Normalize the text using hazm
        if self.normalizer:
            text = self.normalizer.normalize(text)
        else:
            logger.warning("Skipping text normalization because hazm library is not available.")

        # 2. Remove old tags if requested
        if remove_old_tags:
            # A generic tag pattern for chunk boundaries, like @@-0001-@@
            text = re.sub(r'@@-\d{4}-@@', '', text)

            # A pattern for visual separators, like a long line of dashes
            text = re.sub(r'^[─=\-*]{10,}$', '', text, flags=re.MULTILINE)

            # A pattern for simple metadata tags, like [words: 123]
            text = re.sub(r'\[\w+:\s*\d+\]', '', text, flags=re.IGNORECASE)
            logger.debug("Removed old processing tags and separators.")

        # 3. Clean up whitespace
        text = re.sub(r' +', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace 3+ newlines with exactly 2

        # 4. Trim leading/trailing whitespace from the whole text
        text = text.strip()

        logger.info(f"Finished text cleaning. New length: {len(text)} chars.")
        return text

    def extract_document_info(self, text: str) -> dict:
        """
        Extracts basic statistics from the document text.
        """
        if not text:
            return {
                'word_count': 0, 'char_count': 0,
                'sentence_count': 0, 'paragraph_count': 0,
                'avg_words_per_para': 0
            }

        # A simple word split; for more accuracy, a tokenizer could be used.
        words = text.split()
        word_count = len(words)

        # Sentences are approximated by counting terminal punctuation.
        # This is a rough estimate.
        sentences = text.count('.') + text.count('!') + text.count('؟')

        # Paragraphs are non-empty lines.
        paragraphs = [p for p in text.split('\n') if p.strip()]
        paragraph_count = len(paragraphs)

        avg_words_per_para = word_count / paragraph_count if paragraph_count > 0 else 0

        stats = {
            'word_count': word_count,
            'char_count': len(text),
            'sentence_count': sentences,
            'paragraph_count': paragraph_count,
            'avg_words_per_para': round(avg_words_per_para, 2)
        }
        logger.info(f"Extracted document info: {stats}")
        return stats