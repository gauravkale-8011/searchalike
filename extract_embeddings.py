from pathlib import Path
import config
from src.feature_extractor import FeatureExtractor
from src.similarity_search import SimilaritySearch


# ==========================================================
# 🔑 PORTABLE IMAGE PATH COLLECTION (RELATIVE PATHS ONLY)
# ==========================================================
def get_image_paths(image_folder):
    """
    Collect image paths and store them as RELATIVE paths
    so the project works on any laptop.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_folder = Path(image_folder)

    image_paths = []

    for p in image_folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in valid_exts:
            # 🔑 Convert to RELATIVE path
            rel_path = p.relative_to(config.BASE_DIR)
            image_paths.append(str(rel_path).replace("\\", "/"))

    return sorted(image_paths)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    print("🔍 Collecting images...")
    paths = get_image_paths(config.IMAGES_FOLDER)
    print(f"Total images: {len(paths)}")

    print("🔧 Loading trained Triplet model...")
    fe = FeatureExtractor(
        triplet_model_path=str(config.TRIPLET_MODEL_PATH)
    )

    ss = SimilaritySearch(use_faiss=True)

    print("🚀 Extracting embeddings...")
    features = fe.extract_features_batch(
        paths,
        batch_size=64
    )

    print("📦 Building FAISS index...")
    ss.build_index(features, paths)

    print("💾 Saving FAISS index & image paths...")
    ss.save_index(
        index_path=config.TRIPLET_FAISS_INDEX,
        paths_path=config.TRIPLET_IMAGE_PATHS_FILE
    )

    print("💾 Saving embeddings...")
    fe.save_features(
        features,
        config.TRIPLET_EMBEDDINGS_FILE
    )

    print("✅ DONE — Embeddings are portable & deployment-ready!")
