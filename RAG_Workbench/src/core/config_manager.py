import configparser
import json
import os
from typing import Dict, Any, List

from src.utils.logger import logger

class ConfigManager:
    """
    Manages application configuration from config.ini and user profiles.
    """
    def __init__(self, config_path='config.ini', profiles_dir='data/profiles'):
        self.config_path = config_path
        self.profiles_dir = profiles_dir
        self.config = configparser.ConfigParser()

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"فایل تنظیمات اصلی در مسیر '{self.config_path}' یافت نشد.")

        self.config.read(self.config_path, encoding='utf-8')
        logger.info(f"فایل تنظیمات از '{self.config_path}' با موفقیت خوانده شد.")

        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)
            logger.info(f"پوشه پروفایل‌ها در '{self.profiles_dir}' ایجاد شد.")

    def get(self, section: str, key: str, fallback: Any = None) -> str:
        """Retrieves a value from the config.ini."""
        return self.config.get(section, key, fallback=fallback)

    def get_int(self, section: str, key: str, fallback: int = 0) -> int:
        """Retrieves an integer value from the config.ini."""
        return self.config.getint(section, key, fallback=fallback)

    def get_float(self, section: str, key: str, fallback: float = 0.0) -> float:
        """Retrieves a float value from the config.ini."""
        return self.config.getfloat(section, key, fallback=fallback)

    def get_boolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Retrieves a boolean value from the config.ini."""
        return self.config.getboolean(section, key, fallback=fallback)

    def list_profiles(self) -> List[str]:
        """Lists all available .json profile files in the profiles directory."""
        try:
            return [f.replace('.json', '') for f in os.listdir(self.profiles_dir) if f.endswith('.json')]
        except FileNotFoundError:
            return []

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Loads a specific profile from a JSON file."""
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        if not os.path.exists(profile_path):
            logger.error(f"پروفایل '{profile_name}' در مسیر '{profile_path}' یافت نشد.")
            return {}
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
                logger.info(f"پروفایل '{profile_name}' با موفقیت بارگذاری شد.")
                return profile_data.get("config", {})
        except json.JSONDecodeError:
            logger.error(f"خطا در پارس کردن فایل JSON پروفایل: '{profile_path}'")
            return {}

    def save_profile(self, profile_name: str, config_data: Dict[str, Any]) -> bool:
        """Saves a configuration dictionary as a profile JSON file."""
        if not profile_name.strip():
            logger.warning("نام پروفایل برای ذخیره نمی‌تواند خالی باشد.")
            return False

        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")

        # Structure the data as specified
        save_data = {
            "profile_name": profile_name,
            "created_at": __import__('datetime').datetime.now().isoformat(),
            "config": config_data
        }

        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=4)
            logger.info(f"پروفایل '{profile_name}' با موفقیت در '{profile_path}' ذخیره شد.")
            return True
        except Exception as e:
            logger.error(f"خطا در ذخیره پروفایل '{profile_name}': {e}")
            return False

    def delete_profile(self, profile_name: str) -> bool:
        """Deletes a profile file."""
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        if os.path.exists(profile_path):
            try:
                os.remove(profile_path)
                logger.info(f"پروفایل '{profile_name}' با موفقیت حذف شد.")
                return True
            except OSError as e:
                logger.error(f"خطا در حذف پروفایل '{profile_name}': {e}")
                return False
        else:
            logger.warning(f"پروفایل '{profile_name}' برای حذف یافت نشد.")
            return False

# Example of a singleton instance for the app to use
# The paths are relative to the RAG_Workbench directory
config_manager = ConfigManager(config_path='config.ini', profiles_dir='data/profiles')