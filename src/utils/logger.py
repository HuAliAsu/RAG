"""
سیستم لاگینگ مرکزی با پشتیبانی کامل فارسی
"""

import logging
import logging.handlers
from pathlib import Path
import configparser
from datetime import datetime


class CustomFormatter(logging.Formatter):
    """فرمتر سفارشی با رنگ‌های مختلف برای console"""

    # کدهای رنگ ANSI
    COLORS = {
        'DEBUG': '\033[36m',  # آبی فیروزه‌ای
        'INFO': '\033[32m',  # سبز
        'WARNING': '\033[33m',  # زرد
        'ERROR': '\033[31m',  # قرمز
        'CRITICAL': '\033[35m',  # ارغوانی
        'RESET': '\033[0m'  # بازگشت به حالت عادی
    }

    def format(self, record):
        # افزودن رنگ به سطح لاگ
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logger(name: str = 'RAG_Workbench', config_path: str = './config.ini'):
    """
    راه‌اندازی سیستم لاگینگ

    Args:
        name: نام logger
        config_path: مسیر فایل config.ini

    Returns:
        logging.Logger: نمونه logger پیکربندی شده
    """

    # خواندن تنظیمات
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    log_level = config.get('Logging', 'level', fallback='INFO')
    log_file = config.get('Logging', 'file_path', fallback='./logs/app.log')
    max_bytes = int(config.get('Logging', 'max_size_mb', fallback='10')) * 1024 * 1024
    backup_count = int(config.get('Logging', 'backup_count', fallback='5'))
    log_format = config.get('Logging', 'format',
                            fallback='%(asctime)s | %(levelname)s | %(message)s')

    # ایجاد پوشه logs
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ایجاد logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    # جلوگیری از duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # Handler فایل (با rotation)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level))
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)

    # Handler console (با رنگ)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = CustomFormatter(log_format)
    console_handler.setFormatter(console_formatter)

    # اضافه کردن handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # لاگ شروع
    logger.info("=" * 60)
    logger.info(f"🚀 Logger راه‌اندازی شد - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📝 سطح لاگ: {log_level}")
    logger.info(f"📂 فایل لاگ: {log_file}")
    logger.info("=" * 60)

    return logger


# نمونه سراسری
logger = setup_logger()


# توابع کمکی برای استفاده آسان‌تر
def debug(message: str):
    """لاگ سطح DEBUG"""
    logger.debug(message)


def info(message: str):
    """لاگ سطح INFO"""
    logger.info(message)


def warning(message: str):
    """لاگ سطح WARNING"""
    logger.warning(message)


def error(message: str):
    """لاگ سطح ERROR"""
    logger.error(message)


def critical(message: str):
    """لاگ سطح CRITICAL"""
    logger.critical(message)


# مثال استفاده
if __name__ == "__main__":
    debug("این یک پیام DEBUG است")
    info("✅ این یک پیام INFO است")
    warning("⚠️ این یک هشدار است")
    error("❌ این یک خطا است")
    critical("🚨 این یک خطای حیاتی است")