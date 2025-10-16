"""
RAG Workbench - نقطه ورود برنامه
فاز 1: تست زیرساخت
"""

import sys
from pathlib import Path

# اضافه کردن src به path
sys.path.insert(0, str(Path(__file__).parent))
from src.utils.logger import logger, info, warning, error, debug
from src.core.preprocessor import TextPreprocessor
from src.core.embedder import OllamaEmbedder
from src.core.config_manager import ConfigManager
from src.utils.validators import validate_config, validate_pattern_line


def test_phase1():
    """
    تست کامل فاز 1: زیرساخت و ابزارهای پایه
    """

    info("=" * 70)
    info("🚀 RAG Workbench - فاز 1: تست زیرساخت")
    info("=" * 70)

    passed_tests = 0
    failed_tests = 0

    # ═══════════════════════════════════════════════════════════
    # تست 1: Logger
    # ═══════════════════════════════════════════════════════════
    try:
        info("\n📝 تست 1: سیستم لاگینگ")
        info("   ✓ Logger راه‌اندازی شد")
        debug("   ✓ DEBUG level کار می‌کند")
        warning("   ✓ WARNING level کار می‌کند")

        # بررسی فایل لاگ
        log_file = Path("./logs/app.log")
        if log_file.exists():
            info(f"   ✓ فایل لاگ ایجاد شد: {log_file}")
            passed_tests += 1
        else:
            error("   ✗ فایل لاگ ایجاد نشد")
            failed_tests += 1
    except Exception as e:
        error(f"   ✗ خطا در تست Logger: {e}")
        failed_tests += 1

    # ═══════════════════════════════════════════════════════════
    # تست 2: Validators
    # ═══════════════════════════════════════════════════════════
    try:
        info("\n🔍 تست 2: اعتبارسنجی‌ها")

        # تست الگوی صحیح
        is_valid, err, data = validate_pattern_line("5.0,1,chapter,^فصل\\s+[\\d۰-۹]+")
        if is_valid and data:
            info(f"   ✓ الگوی صحیح: {data['name']}")
            passed_tests += 1
        else:
            error(f"   ✗ خطا در الگوی صحیح: {err}")
            failed_tests += 1

        # تست الگوی نامعتبر
        is_valid, err, data = validate_pattern_line("10.0,1,bad,^test")
        if not is_valid:
            info(f"   ✓ الگوی نامعتبر شناسایی شد")
            passed_tests += 1
        else:
            error("   ✗ الگوی نامعتبر تشخیص داده نشد")
            failed_tests += 1

        # تست config
        test_config = {
            'min_chunk_size': 200,
            'max_chunk_size': 800,
            'overlap_size': 50
        }
        is_valid, errors = validate_config(test_config)
        if is_valid:
            info("   ✓ Config صحیح تأیید شد")
            passed_tests += 1
        else:
            error(f"   ✗ Config نامعتبر: {errors}")
            failed_tests += 1

    except Exception as e:
        error(f"   ✗ خطا در تست Validators: {e}")
        failed_tests += 1

    # ═══════════════════════════════════════════════════════════
    # تست 3: Preprocessor
    # ═══════════════════════════════════════════════════════════
    try:
        info("\n🧹 تست 3: پیش‌پردازش متن")

        preprocessor = TextPreprocessor()

        # متن تست با مشکلات
        dirty_text = """
        @@-0001-@@

        این    یک   متن     تست است.


        با فضاهای    اضافی.

        [کلمات: 450 | Stage: 1]

        ────────────────────────────────
        """

        # پاکسازی
        cleaned = preprocessor.clean_text(dirty_text)

        # بررسی نتیجه
        if "@@-" not in cleaned and "────" not in cleaned:
            info("   ✓ تگ‌ها و جداکننده‌ها حذف شدند")
            passed_tests += 1
        else:
            error("   ✗ تگ‌ها حذف نشدند")
            failed_tests += 1

        # استخراج اطلاعات
        info_dict = preprocessor.extract_doc_info(cleaned)
        if info_dict['word_count'] > 0:
            info(f"   ✓ اطلاعات استخراج شد: {info_dict['word_count']} کلمه")
            passed_tests += 1
        else:
            error("   ✗ استخراج اطلاعات ناموفق")
            failed_tests += 1

    except Exception as e:
        error(f"   ✗ خطا در تست Preprocessor: {e}")
        failed_tests += 1

    # ═══════════════════════════════════════════════════════════
    # تست 4: Embedder (اتصال به Ollama)
    # ═══════════════════════════════════════════════════════════
    try:
        info("\n🔗 تست 4: اتصال به Ollama")

        embedder = OllamaEmbedder()

        if embedder.available:
            info("   ✓ Ollama در دسترس است")

            # تست embedding
            test_emb = embedder.get_embedding("تست")
            if test_emb is not None:
                info(f"   ✓ Embedding دریافت شد (dimension: {len(test_emb)})")
                passed_tests += 1

                # تست شباهت
                emb1 = embedder.get_embedding("این متن اول")
                emb2 = embedder.get_embedding("این متن دوم")
                if emb1 is not None and emb2 is not None:
                    sim = embedder.calculate_similarity(emb1, emb2)
                    info(f"   ✓ محاسبه شباهت: {sim:.3f}")
                    passed_tests += 1
                else:
                    error("   ✗ دریافت embedding ناموفق")
                    failed_tests += 1
            else:
                error("   ✗ دریافت embedding ناموفق")
                failed_tests += 1
        else:
            warning("   ⚠ Ollama در دسترس نیست (نرمال برای تست)")
            warning("   💡 برای فعال‌سازی Stage 3:")
            warning("      1. ollama serve")
            warning("      2. ollama pull embeddinggemma")
            passed_tests += 1  # این نرمال است

    except Exception as e:
        error(f"   ✗ خطا در تست Embedder: {e}")
        failed_tests += 1

    # ═══════════════════════════════════════════════════════════
    # تست 5: Config Manager
    # ═══════════════════════════════════════════════════════════
    try:
        info("\n💾 تست 5: مدیریت پروفایل‌ها")

        config_mgr = ConfigManager()

        # لیست پروفایل‌ها
        profiles = config_mgr.list_profiles()
        if len(profiles) >= 3:
            info(f"   ✓ {len(profiles)} پروفایل پیش‌فرض ایجاد شد:")
            for p in profiles:
                info(f"      • {p}")
            passed_tests += 1
        else:
            error(f"   ✗ تعداد پروفایل‌های پیش‌فرض کم است: {len(profiles)}")
            failed_tests += 1

        # تست ذخیره/بارگذاری
        test_profile = {
            "name": "تست_فاز1",
            "config": {
                "min_chunk_size": 200,
                "max_chunk_size": 800
            }
        }

        if config_mgr.save_profile("تست_فاز1", test_profile):
            info("   ✓ ذخیره پروفایل تست")

            loaded = config_mgr.load_profile("تست_فاز1")
            if loaded and loaded['name'] == "تست_فاز1":
                info("   ✓ بارگذاری پروفایل تست")
                passed_tests += 1

                # حذف پروفایل تست
                config_mgr.delete_profile("تست_فاز1")
                info("   ✓ حذف پروفایل تست")
                passed_tests += 1
            else:
                error("   ✗ بارگذاری پروفایل ناموفق")
                failed_tests += 1
        else:
            error("   ✗ ذخیره پروفایل ناموفق")
            failed_tests += 1

    except Exception as e:
        error(f"   ✗ خطا در تست Config Manager: {e}")
        failed_tests += 1

    # ═══════════════════════════════════════════════════════════
    # تست 6: ساختار پوشه‌ها
    # ═══════════════════════════════════════════════════════════
    try:
        info("\n📁 تست 6: ساختار پوشه‌ها")

        required_dirs = [
            "./data/profiles",
            "./data/chunked_files",
            "./data/vector_stores",
            "./logs"
        ]

        all_exist = True
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                info(f"   ✓ {dir_path}")
            else:
                error(f"   ✗ {dir_path} وجود ندارد")
                all_exist = False

        if all_exist:
            passed_tests += 1
        else:
            failed_tests += 1

    except Exception as e:
        error(f"   ✗ خطا در بررسی ساختار: {e}")
        failed_tests += 1

    # ═══════════════════════════════════════════════════════════
    # نتیجه نهایی
    # ═══════════════════════════════════════════════════════════
    total_tests = passed_tests + failed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    info("\n" + "=" * 70)
    info("📊 نتایج تست فاز 1")
    info("=" * 70)
    info(f"✅ تست‌های موفق: {passed_tests}")
    info(f"❌ تست‌های ناموفق: {failed_tests}")
    info(f"📈 نرخ موفقیت: {success_rate:.1f}%")
    info("=" * 70)

    if failed_tests == 0:
        info("\n🎉 فاز 1 با موفقیت کامل شد!")
        info("✅ تمام زیرساخت‌ها آماده هستند")
        info("🚀 می‌توانید به فاز 2 بروید: Chunking Engine")
        return True
    else:
        error("\n⚠️ برخی تست‌ها ناموفق بودند")
        error("🔧 لطفاً خطاها را بررسی و رفع کنید")
        return False


def show_help():
    """نمایش راهنمای استفاده"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              RAG Workbench - فاز 1: زیرساخت                 ║
╚══════════════════════════════════════════════════════════════╝

استفاده:
  python main.py          تست فاز 1
  python main.py --help   نمایش این راهنما

توضیحات:
  این برنامه زیرساخت اصلی RAG Workbench را تست می‌کند:

  ✓ سیستم لاگینگ
  ✓ اعتبارسنجی‌ها
  ✓ پیش‌پردازش متن
  ✓ اتصال به Ollama
  ✓ مدیریت پروفایل‌ها
  ✓ ساختار پوشه‌ها

پیش‌نیازها:
  1. نصب requirements.txt:
     pip install -r requirements.txt

  2. (اختیاری) راه‌اندازی Ollama برای Stage 3:
     ollama serve
     ollama pull embeddinggemma

نکته:
  بدون Ollama، سیستم با Stage 1 و 2 کار می‌کند.
  Stage 3 فقط برای دقت بیشتر لازم است.
""")


if __name__ == "__main__":
    # بررسی آرگومان‌ها
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_help()
        sys.exit(0)

    # اجرای تست
    try:
        success = test_phase1()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ عملیات توسط کاربر لغو شد")
        sys.exit(130)
    except Exception as e:
        error(f"\n❌ خطای غیرمنتظره: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)