"""
RAG Workbench - سیستم چانکینگ و جستجوی معنایی برای اسناد فارسی

نسخه: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "RAG Workbench Team"

# Import اصلی برای دسترسی آسان
from src.utils.logger import logger
from src.core.preprocessor import TextPreprocessor
from src.core.embedder import OllamaEmbedder
from src.core.config_manager import ConfigManager

__all__ = [
    'logger',
    'TextPreprocessor',
    'OllamaEmbedder',
    'ConfigManager'
]