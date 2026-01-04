import os
import pickle
import logging
from pathlib import Path
from typing import Optional
from werkzeug.utils import secure_filename
import config

# ------------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# File validation
# ------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    """
    Check if file extension is allowed.
    """
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


# ------------------------------------------------------------------
# Upload handling
# ------------------------------------------------------------------
def save_uploaded_file(file, upload_folder: Path = config.UPLOAD_FOLDER) -> Optional[str]:
    """
    Save uploaded file to disk.
    Returns RELATIVE path (portable).
    """
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"

            upload_folder.mkdir(parents=True, exist_ok=True)

            filepath = upload_folder / filename
            file.save(str(filepath))

            # 🔑 RETURN RELATIVE PATH
            relative_path = filepath.relative_to(config.BASE_DIR)
            logger.info(f"File saved: {relative_path}")

            return str(relative_path).replace("\\", "/")

        return None
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None


# ------------------------------------------------------------------
# 🔑 THIS IS THE MAIN FIX
# ------------------------------------------------------------------
def get_image_paths(image_folder):
    """
    Recursively collect all image paths from subfolders.
    Paths are stored as RELATIVE paths so project works on any laptop.
    """
    image_folder = Path(image_folder)  # ❌ NO .resolve()

    if not image_folder.exists():
        logger.error(f"Images folder does not exist: {image_folder}")
        return []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    image_paths = []

    for p in image_folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_extensions:
            # 🔑 STORE RELATIVE PATH
            relative_path = p.relative_to(config.BASE_DIR)
            image_paths.append(str(relative_path).replace("\\", "/"))

    logger.info(f"Found {len(image_paths)} images in {image_folder}")
    return image_paths


# ------------------------------------------------------------------
# Pickle helpers
# ------------------------------------------------------------------
def save_pickle(data, filepath: Path):
    """
    Save data to pickle file.
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving pickle file: {str(e)}")
        raise


def load_pickle(filepath: Path):
    """
    Load data from pickle file.
    """
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle file: {str(e)}")
        raise


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def ensure_dir(directory: Path):
    """
    Ensure directory exists.
    """
    directory.mkdir(parents=True, exist_ok=True)


def clean_upload_folder(folder: Path = config.UPLOAD_FOLDER, max_age_hours: int = 24):
    """
    Clean old files from upload folder.
    """
    import time

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    try:
        for filepath in folder.glob("*"):
            if filepath.is_file():
                file_age = current_time - filepath.stat().st_mtime
                if file_age > max_age_seconds:
                    filepath.unlink()
                    logger.info(f"Deleted old file: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning upload folder: {str(e)}")
