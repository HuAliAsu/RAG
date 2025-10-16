"""
مدیریت پروفایل‌های تنظیمات کاربر
"""

import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime


class ConfigManager:
    """
    کلاس مدیریت پروفایل‌های تنظیمات
    """

    def __init__(self, profiles_dir: str = "./data/profiles"):
        """
        مقداردهی اولیه

        Args:
            profiles_dir: پوشه ذخیره پروفایل‌ها
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        # ایجاد پروفایل‌های پیش‌فرض در اولین اجرا
        self._ensure_default_profiles()

    def _ensure_default_profiles(self):
        """ایجاد پروفایل‌های پیش‌فرض اگر وجود ندارند"""

        # پروفایل 1: داستان‌های کوتاه
        if not (self.profiles_dir / "داستان‌های_کوتاه.json").exists():
            self.save_profile("داستان‌های_کوتاه", self._get_stories_profile())

        # پروفایل 2: کتاب‌های معنوی
        if not (self.profiles_dir / "کتاب‌های_معنوی.json").exists():
            self.save_profile("کتاب‌های_معنوی", self._get_spiritual_profile())

        # پروفایل 3: مقالات علمی
        if not (self.profiles_dir / "مقالات_علمی.json").exists():
            self.save_profile("مقالات_علمی", self._get_academic_profile())

    def _get_stories_profile(self) -> Dict:
        """پروفایل پیش‌فرض برای داستان‌های کوتاه"""
        return {
            "name": "داستان‌های کوتاه",
            "description": "برای کتاب‌هایی مانند داستان‌های رومی، حکایت‌های گلستان",
            "created_at": datetime.now().isoformat(),
            "config": {
                "min_chunk_size": 150,
                "max_chunk_size": 600,
                "overlap_size": 40,
                "coherence_threshold": 0.12,
                "similarity_threshold": 0.72,
                "structural_patterns": """5.0,1,story_title,^(داستان|حکایت|قصه)\\s+[\\d۰-۹]+
4.5,1,chapter,^(فصل|بخش)\\s+[\\d۰-۹]+
4.0,1,numbered,^[\\d۰-۹]+[\\)\\.]\\s+
3.5,1,separator,^[=\\-*]{3,}$""",
                "semantic_patterns": """2.5,2,opening_classic,روزی\\s+روزگاری
2.5,2,opening_folklore,یکی\\s+بود\\s+یکی\\s+نبود
2.3,2,lesson,درس\\s+این\\s+(داستان|حکایت)
2.0,2,conclusion,به\\s+این\\s+ترتیب""",
                "special_keywords": """3.0,1,end_marker,پایان\\s+(داستان|حکایت)
2.5,1,question,پرسش\\s*:"""
            }
        }

    def _get_spiritual_profile(self) -> Dict:
        """پروفایل پیش‌فرض برای کتاب‌های معنوی"""
        return {
            "name": "کتاب‌های معنوی و سخنرانی",
            "description": "برای کتاب‌های هارولد کلمپ، سخنرانی‌های معنوی",
            "created_at": datetime.now().isoformat(),
            "config": {
                "min_chunk_size": 200,
                "max_chunk_size": 900,
                "overlap_size": 70,
                "coherence_threshold": 0.18,
                "similarity_threshold": 0.78,
                "structural_patterns": """5.0,1,chapter,^(فصل|بخش|قسمت)\\s+[\\d۰-۹]+
4.0,1,section,^[\\d۰-۹]+[\\)\\.]\\s+
3.5,1,separator,^[=\\-*]{3,}$""",
                "semantic_patterns": """2.5,2,transition,^(حال|اکنون|بگذارید|بیایید)\\s+
2.3,2,contrast,^(اما|ولی|با\\s+این\\s+حال)\\s+
2.0,2,reflection,تأمل\\s+در\\s+(این|آن)
1.8,2,example,^(برای\\s+مثال|مثلاً)\\s+""",
                "special_keywords": """2.5,1,topic,موضوع\\s+جدید
2.0,2,question,پرسش\\s*:
2.0,2,exercise,تمرین\\s*:"""
            }
        }

    def _get_academic_profile(self) -> Dict:
        """پروفایل پیش‌فرض برای مقالات علمی"""
        return {
            "name": "مقالات و متون علمی",
            "description": "برای مقالات تحقیقاتی، کتاب‌های درسی",
            "created_at": datetime.now().isoformat(),
            "config": {
                "min_chunk_size": 250,
                "max_chunk_size": 1000,
                "overlap_size": 80,
                "coherence_threshold": 0.20,
                "similarity_threshold": 0.80,
                "structural_patterns": """5.0,1,section,^\\d+[\\.\\)]\\s+
4.5,1,subsection,^\\d+\\.\\d+\\s+
4.0,1,heading,^[A-Z\\s]{5,30}$
3.5,1,separator,^[=\\-*]{3,}$""",
                "semantic_patterns": """2.0,2,intro,^(مقدمه|چکیده|خلاصه)\\s*:
2.0,2,method,^(روش|متدولوژی)\\s*:
2.0,2,result,^(نتایج|یافته‌ها)\\s*:
2.0,2,conclusion,^(نتیجه‌گیری|جمع‌بندی)\\s*:""",
                "special_keywords": """2.5,1,reference,^منابع\\s*:
2.0,2,table,^جدول\\s+[\\d۰-۹]+
2.0,2,figure,^شکل\\s+[\\d۰-۹]+"""
            }
        }

    def save_profile(self, name: str, config: Dict) -> bool:
        """
        ذخیره یک پروفایل

        Args:
            name: نام پروفایل
            config: دیکشنری تنظیمات

        Returns:
            bool: موفقیت عملیات
        """
        try:
            # افزودن timestamp
            if 'created_at' not in config:
                config['created_at'] = datetime.now().isoformat()

            config['last_modified'] = datetime.now().isoformat()

            # ذخیره فایل
            file_path = self.profiles_dir / f"{name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            from src.utils.logger import info
            info(f"✅ پروفایل '{name}' ذخیره شد")
            return True

        except Exception as e:
            from src.utils.logger import error
            error(f"❌ خطا در ذخیره پروفایل '{name}': {str(e)}")
            return False

    def load_profile(self, name: str) -> Optional[Dict]:
        """
        بارگذاری یک پروفایل

        Args:
            name: نام پروفایل

        Returns:
            dict یا None: تنظیمات پروفایل
        """
        try:
            file_path = self.profiles_dir / f"{name}.json"

            if not file_path.exists():
                from src.utils.logger import warning
                warning(f"⚠️ پروفایل '{name}' یافت نشد")
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            from src.utils.logger import info
            info(f"✅ پروفایل '{name}' بارگذاری شد")
            return config

        except json.JSONDecodeError as e:
            from src.utils.logger import error
            error(f"❌ خطا در پارس JSON پروفایل '{name}': {str(e)}")
            return None

        except Exception as e:
            from src.utils.logger import error
            error(f"❌ خطا در بارگذاری پروفایل '{name}': {str(e)}")
            return None

    def delete_profile(self, name: str) -> bool:
        """
        حذف یک پروفایل

        Args:
            name: نام پروفایل

        Returns:
            bool: موفقیت عملیات
        """
        try:
            file_path = self.profiles_dir / f"{name}.json"

            if not file_path.exists():
                from src.utils.logger import warning
                warning(f"⚠️ پروفایل '{name}' برای حذف یافت نشد")
                return False

            file_path.unlink()

            from src.utils.logger import info
            info(f"✅ پروفایل '{name}' حذف شد")
            return True

        except Exception as e:
            from src.utils.logger import error
            error(f"❌ خطا در حذف پروفایل '{name}': {str(e)}")
            return False

    def list_profiles(self) -> List[str]:
        """
        لیست تمام پروفایل‌های موجود

        Returns:
            list: لیست نام پروفایل‌ها
        """
        try:
            profiles = [f.stem for f in self.profiles_dir.glob("*.json")]
            return sorted(profiles)
        except Exception as e:
            from src.utils.logger import error
            error(f"❌ خطا در لیست کردن پروفایل‌ها: {str(e)}")
            return []

    def get_profile_info(self, name: str) -> Optional[Dict]:
        """
        دریافت اطلاعات خلاصه یک پروفایل

        Args:
            name: نام پروفایل

        Returns:
            dict یا None: اطلاعات پروفایل
        """
        config = self.load_profile(name)
        if not config:
            return None

        return {
            'name': config.get('name', name),
            'description': config.get('description', ''),
            'created_at': config.get('created_at', ''),
            'last_modified': config.get('last_modified', ''),
            'min_chunk_size': config.get('config', {}).get('min_chunk_size', 0),
            'max_chunk_size': config.get('config', {}).get('max_chunk_size', 0)
        }

    def export_profile(self, name: str, export_path: str) -> bool:
        """
        صادر کردن پروفایل به مسیر دیگر

        Args:
            name: نام پروفایل
            export_path: مسیر مقصد

        Returns:
            bool: موفقیت عملیات
        """
        try:
            config = self.load_profile(name)
            if not config:
                return False

            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            from src.utils.logger import info
            info(f"✅ پروفایل '{name}' به '{export_path}' صادر شد")
            return True

        except Exception as e:
            from src.utils.logger import error
            error(f"❌ خطا در صادر کردن پروفایل: {str(e)}")
            return False

    def import_profile(self, import_path: str, new_name: Optional[str] = None) -> bool:
        """
        وارد کردن پروفایل از فایل

        Args:
            import_path: مسیر فایل پروفایل
            new_name: نام جدید (اختیاری)

        Returns:
            bool: موفقیت عملیات
        """
        try:
            import_file = Path(import_path)

            if not import_file.exists():
                from src.utils.logger import error
                error(f"❌ فایل '{import_path}' یافت نشد")
                return False

            with open(import_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # تعیین نام
            if new_name:
                profile_name = new_name
            else:
                profile_name = config.get('name', import_file.stem)

            return self.save_profile(profile_name, config)

        except Exception as e:
            from src.utils.logger import error
            error(f"❌ خطا در وارد کردن پروفایل: {str(e)}")
            return False


# تست
if __name__ == "__main__":
    print("🧪 تست config_manager.py\n")

    manager = ConfigManager()

    # لیست پروفایل‌ها
    print("1️⃣ پروفایل‌های موجود:")
    profiles = manager.list_profiles()
    for p in profiles:
        print(f"   • {p}")
    print()

    # بارگذاری یک پروفایل
    if profiles:
        print(f"2️⃣ بارگذاری پروفایل '{profiles[0]}':")
        config = manager.load_profile(profiles[0])
        if config:
            print(f"   • توضیحات: {config.get('description', 'N/A')}")
            print(f"   • min_chunk_size: {config['config']['min_chunk_size']}")
            print(f"   • max_chunk_size: {config['config']['max_chunk_size']}")
        print()

    # ذخیره پروفایل تست
    print("3️⃣ ذخیره پروفایل تست:")
    test_config = {
        "name": "تست",
        "description": "پروفایل تستی",
        "config": {
            "min_chunk_size": 200,
            "max_chunk_size": 800,
            "overlap_size": 50
        }
    }
    manager.save_profile("تست", test_config)

    # حذف پروفایل تست
    print("\n4️⃣ حذف پروفایل تست:")
    manager.delete_profile("تست")