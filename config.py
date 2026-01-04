import os
from pathlib import Path
import torch
from pathlib import Path

# Project root directory (auto-detected)
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data" / "images"

# Static directories (for web)
STATIC_DIR = BASE_DIR / "static"
STATIC_IMAGES_DIR = STATIC_DIR / "images"
UPLOAD_DIR = STATIC_DIR / "uploads"

# Saved files
EMBEDDINGS_FILE = BASE_DIR / "embeddings.npy"
FAISS_INDEX_FILE = BASE_DIR / "faiss.index"
IMAGE_PATHS_FILE = BASE_DIR / "image_paths.pkl"

# ============================================================================
# BASE CONFIGURATION
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Paths
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
DATA_FOLDER = BASE_DIR / 'data'
IMAGES_FOLDER = DATA_FOLDER / 'images'
EMBEDDINGS_FOLDER = DATA_FOLDER / 'embeddings'
MODEL_FOLDER = DATA_FOLDER / 'model'
CHECKPOINT_FOLDER = BASE_DIR / 'checkpoints'
LOG_DIR = BASE_DIR / 'runs'
VIZ_OUTPUT_DIR = BASE_DIR / 'visualizations'
PCA_COMPONENTS = 128
PCA_MODEL_FILE = EMBEDDINGS_FOLDER / 'pca_model.pkl'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, DATA_FOLDER, IMAGES_FOLDER, EMBEDDINGS_FOLDER,
               MODEL_FOLDER, CHECKPOINT_FOLDER, LOG_DIR, VIZ_OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FLASK CONFIGURATION
# ============================================================================

SECRET_KEY = os.environ.get('SECRET_KEY', 'triplet-network-secret-key-2024')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Flask deployment
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# ============================================================================
# AUTO-DETECTION: Check if trained model exists
# ============================================================================

TRIPLET_MODEL_PATH = CHECKPOINT_FOLDER / 'best_model.pth'


# Auto-detect if we should use triplet model
def should_use_triplet_model():
    """Check if trained triplet model exists and is valid"""
    if TRIPLET_MODEL_PATH.exists():
        try:
            # Try to load checkpoint to verify it's valid
            checkpoint = torch.load(TRIPLET_MODEL_PATH, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                return True
        except Exception:
            return False
    return False


USE_TRIPLET_MODEL = should_use_triplet_model()

# Auto-detect GPU
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

print(f"🚀 Using device: {DEVICE.upper()}")
if USE_GPU:
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Pretrained Model Settings (fallback when no triplet model)
MODEL_NAME = 'resnet50'
FEATURE_DIM = 2048

# Triplet Network Settings
TRIPLET_BACKBONE = 'resnet50'
TRIPLET_EMBEDDING_DIM = 128
TRIPLET_MARGIN = 1.0

# Image Preprocessing
IMAGE_SIZE = (224, 224)

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

TOP_K = 10
SIMILARITY_THRESHOLD = 0.0

# FAISS settings
USE_FAISS = True
FAISS_INDEX_TYPE = 'IndexFlatIP'

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training hyperparameters
TRAIN_BATCH_SIZE = 32 if USE_GPU else 16
TRAIN_EPOCHS = 50
TRAIN_LEARNING_RATE = 0.0001
TRAIN_WEIGHT_DECAY = 1e-4

# Triplet mining settings
TRIPLET_MINING_MODE = 'online'
TRIPLET_LOSS_TYPE = 'hardest'

# Data split
VAL_SPLIT = 0.2

# Training optimization
EARLY_STOPPING_PATIENCE = 15
SAVE_CHECKPOINT_EVERY = 5

# Backbone training
FREEZE_BACKBONE_INITIALLY = False
UNFREEZE_AFTER_EPOCHS = 10

# Projection head
PROJECTION_HIDDEN_DIM = 512
PROJECTION_DROPOUT = 0.5

# Data loader settings
NUM_WORKERS = 4 if USE_GPU else 2
PIN_MEMORY = True if USE_GPU else False

# Triplet dataset
TRIPLETS_PER_ANCHOR = 5

# Balanced batch sampler
BALANCED_BATCH_P = 8
BALANCED_BATCH_K = 4

# Optimization features
ENABLE_GRADIENT_CLIPPING = True
GRADIENT_CLIP_VALUE = 1.0
ENABLE_AUGMENTATION = True
ENABLE_MIXED_PRECISION = False

# ============================================================================
# FEATURE FILES & INDICES
# ============================================================================

# Pretrained model files
EMBEDDINGS_FILE = EMBEDDINGS_FOLDER / 'features.npy'
IMAGE_PATHS_FILE = EMBEDDINGS_FOLDER / 'image_paths.pkl'
FAISS_INDEX_FILE = EMBEDDINGS_FOLDER / 'faiss_index.bin'

# Triplet model files
TRIPLET_EMBEDDINGS_FILE = EMBEDDINGS_FOLDER / 'triplet_features.npy'
TRIPLET_IMAGE_PATHS_FILE = EMBEDDINGS_FOLDER / 'triplet_image_paths.pkl'
TRIPLET_FAISS_INDEX = EMBEDDINGS_FOLDER / 'triplet_faiss_index.bin'

# ============================================================================
# LABEL GENERATION SETTINGS
# ============================================================================

FILENAME_DELIMITER = '_'
FILENAME_LABEL_POSITION = 0
SYNTHETIC_N_CLUSTERS = 20

# ============================================================================
# LOGGING & MONITORING
# ============================================================================

LOG_LEVEL = 'INFO'
ENABLE_TENSORBOARD = True
VERBOSE_LOGGING = False

# Visualization settings
VIZ_MAX_CLASSES = 20

# ============================================================================
# UPLOAD & CLEANUP
# ============================================================================

UPLOAD_CLEANUP_HOURS = 24

# ============================================================================
# AUTO-TRAINING SETTINGS
# ============================================================================

# Minimum number of images required for training
MIN_IMAGES_FOR_TRAINING = 50

# Auto-train on first run if model doesn't exist
AUTO_TRAIN_ON_FIRST_RUN = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config():
    """Get current model configuration."""
    return {
        'use_triplet': USE_TRIPLET_MODEL,
        'model_name': MODEL_NAME if not USE_TRIPLET_MODEL else 'triplet',
        'backbone': TRIPLET_BACKBONE if USE_TRIPLET_MODEL else MODEL_NAME,
        'embedding_dim': TRIPLET_EMBEDDING_DIM if USE_TRIPLET_MODEL else FEATURE_DIM,
        'device': DEVICE,
        'use_gpu': USE_GPU
    }


def check_training_requirements():
    """Check if system is ready for training"""
    issues = []

    # Check if images exist
    if not IMAGES_FOLDER.exists():
        issues.append(f"❌ Images folder not found: {IMAGES_FOLDER}")
        return False, issues

    # Count images
    image_files = []
    for ext in ALLOWED_EXTENSIONS:
        image_files.extend(list(IMAGES_FOLDER.rglob(f'*.{ext}')))
        image_files.extend(list(IMAGES_FOLDER.rglob(f'*.{ext.upper()}')))

    num_images = len(image_files)

    if num_images == 0:
        issues.append(f"❌ No images found in {IMAGES_FOLDER}")
        return False, issues

    if num_images < MIN_IMAGES_FOR_TRAINING:
        issues.append(f"⚠️  Only {num_images} images found. Recommend {MIN_IMAGES_FOR_TRAINING}+ for good results")
        return False, issues

    return True, [f"✅ Found {num_images} images ready for training"]


def print_startup_info():
    """Print startup information"""
    print("\n" + "=" * 70)
    print("🚀 IMAGE SIMILARITY SEARCH - SMART AUTO-TRAIN")
    print("=" * 70)

    print(f"\n📊 Model Status:")
    if USE_TRIPLET_MODEL:
        print(f"   ✅ Using Trained Triplet Model")
        print(f"   📁 Model: {TRIPLET_MODEL_PATH}")
        print(f"   🎯 Embedding Dim: {TRIPLET_EMBEDDING_DIM}")
    else:
        print(f"   ⚠️  No trained model found")
        print(f"   📌 Using pretrained {MODEL_NAME} (fallback)")
        if AUTO_TRAIN_ON_FIRST_RUN:
            ready, messages = check_training_requirements()
            if ready:
                print(f"   🔄 Will auto-train on startup")
            else:
                print(f"   ❌ Cannot auto-train:")
                for msg in messages:
                    print(f"      {msg}")

    print(f"\n💻 Hardware:")
    print(f"   Device: {DEVICE.upper()}")
    if USE_GPU:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n📁 Paths:")
    print(f"   Images: {IMAGES_FOLDER}")
    print(f"   Checkpoints: {CHECKPOINT_FOLDER}")

    print("=" * 70 + "\n")


__all__ = [
    'BASE_DIR', 'UPLOAD_FOLDER', 'DATA_FOLDER', 'IMAGES_FOLDER',
    'EMBEDDINGS_FOLDER', 'MODEL_FOLDER', 'CHECKPOINT_FOLDER', 'LOG_DIR',
    'SECRET_KEY', 'MAX_CONTENT_LENGTH', 'ALLOWED_EXTENSIONS',
    'FLASK_HOST', 'FLASK_PORT', 'FLASK_DEBUG',
    'USE_TRIPLET_MODEL', 'MODEL_NAME', 'FEATURE_DIM',
    'TRIPLET_MODEL_PATH', 'TRIPLET_EMBEDDING_DIM', 'TRIPLET_MARGIN',
    'TRIPLET_BACKBONE', 'IMAGE_SIZE',
    'TOP_K', 'SIMILARITY_THRESHOLD', 'USE_FAISS',
    'EMBEDDINGS_FILE', 'IMAGE_PATHS_FILE', 'FAISS_INDEX_FILE',
    'TRIPLET_EMBEDDINGS_FILE', 'TRIPLET_FAISS_INDEX',
    'USE_GPU', 'DEVICE',
    'TRAIN_BATCH_SIZE', 'TRAIN_EPOCHS', 'TRAIN_LEARNING_RATE',
    'TRAIN_WEIGHT_DECAY', 'TRIPLET_LOSS_TYPE', 'TRIPLET_MINING_MODE',
    'VAL_SPLIT', 'EARLY_STOPPING_PATIENCE', 'NUM_WORKERS',
    'get_model_config', 'check_training_requirements', 'print_startup_info',
    'should_use_triplet_model', 'AUTO_TRAIN_ON_FIRST_RUN', 'MIN_IMAGES_FOR_TRAINING'
]