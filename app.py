from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import logging
import sys

import config
from src.feature_extractor import FeatureExtractor
from src.similarity_search import SimilaritySearch
from src.utils import allowed_file, save_uploaded_file, get_image_paths, clean_upload_folder
from src.preprocessing import load_and_validate_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# Global variables
feature_extractor = None
similarity_search = None
index_loaded = False

def check_and_train_if_needed():
    """
    Check if model needs training and train if necessary.
    Returns True if ready to use, False otherwise.
    """
    global feature_extractor, similarity_search, index_loaded

    logger.info("=" * 70)
    logger.info("🔍 CHECKING MODEL STATUS")
    logger.info("=" * 70)

    # Check if trained model exists
    if config.TRIPLET_MODEL_PATH.exists():
        logger.info("✅ Trained triplet model found!")
        logger.info(f"   Model: {config.TRIPLET_MODEL_PATH}")
        return True

    # Model doesn't exist - check if we should auto-train
    logger.info("⚠️  No trained model found")

    if not config.AUTO_TRAIN_ON_FIRST_RUN:
        logger.warning("❌ Auto-train disabled. Please run: python train_triplet.py")
        return False

    # Check training requirements
    ready, messages = config.check_training_requirements()

    for msg in messages:
        logger.info(msg)

    if not ready:
        logger.error("❌ Cannot train - requirements not met")
        logger.info("📝 To fix:")
        logger.info(f"   1. Add images to: {config.IMAGES_FOLDER}")
        logger.info(f"   2. Organize in subfolders by category")
        logger.info(f"   3. Minimum {config.MIN_IMAGES_FOR_TRAINING} images required")
        return False

    # Requirements met - start training
    logger.info("=" * 70)
    logger.info("🚀 STARTING AUTO-TRAINING")
    logger.info("=" * 70)
    logger.info("⏳ This will take some time... Please wait.")
    logger.info("")

    try:
        # Import training modules
        from src.triplet_trainer import TripletTrainer, get_default_transforms, freeze_backbone
        from src.triplet_network import TripletNetwork, OnlineTripletLoss
        from src.triplet_dataset import create_triplet_dataloader, generate_labels_from_folders
        from sklearn.model_selection import train_test_split
        import torch

        # Load data
        logger.info("📁 Loading images...")
        image_paths, labels = generate_labels_from_folders(config.IMAGES_FOLDER)

        # Convert to numeric labels
        label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        numeric_labels = [label_to_idx[label] for label in labels]

        logger.info(f"   Found {len(image_paths)} images")
        logger.info(f"   Classes: {len(set(labels))}")

        # Train/val split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, numeric_labels,
            test_size=config.VAL_SPLIT,
            stratify=numeric_labels,
            random_state=42
        )

        logger.info(f"   Train: {len(train_paths)}, Val: {len(val_paths)}")

        # Create transforms
        train_transform = get_default_transforms(augment=True)
        val_transform = get_default_transforms(augment=False)

        # Create data loaders
        logger.info("🔄 Creating data loaders...")
        train_loader = create_triplet_dataloader(
            train_paths, train_labels, train_transform,
            batch_size=config.TRAIN_BATCH_SIZE,
            mode=config.TRIPLET_MINING_MODE,
            shuffle=True,
            num_workers=config.NUM_WORKERS
        )

        val_loader = create_triplet_dataloader(
            val_paths, val_labels, val_transform,
            batch_size=config.TRAIN_BATCH_SIZE,
            mode=config.TRIPLET_MINING_MODE,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )

        # Create model
        logger.info("🏗️  Creating model...")
        model = TripletNetwork(
            embedding_dim=config.TRIPLET_EMBEDDING_DIM,
            pretrained=True,
            backbone=config.TRIPLET_BACKBONE
        )

        if config.FREEZE_BACKBONE_INITIALLY:
            freeze_backbone(model, freeze=True)

        # Create loss and optimizer
        loss_fn = OnlineTripletLoss(
            margin=config.TRIPLET_MARGIN,
            triplet_selector=config.TRIPLET_LOSS_TYPE
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.TRAIN_LEARNING_RATE,
            weight_decay=config.TRAIN_WEIGHT_DECAY
        )

        # Create trainer
        logger.info("🎯 Initializing trainer...")
        trainer = TripletTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=config.DEVICE,
            checkpoint_dir=str(config.CHECKPOINT_FOLDER),
            log_dir=str(config.LOG_DIR)
        )

        # Train
        logger.info("=" * 70)
        logger.info(f"🚂 TRAINING FOR {config.TRAIN_EPOCHS} EPOCHS")
        logger.info("=" * 70)

        trainer.train(
            num_epochs=config.TRAIN_EPOCHS,
            save_every=config.SAVE_CHECKPOINT_EVERY,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE
        )

        logger.info("=" * 70)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"📁 Model saved: {config.TRIPLET_MODEL_PATH}")
        logger.info("")

        # Update config to use triplet model
        config.USE_TRIPLET_MODEL = True

        return True

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def initialize_models():
    """Initialize feature extractor and similarity search"""
    global feature_extractor, similarity_search, index_loaded

    # Determine which model to use
    triplet_model_path = None
    embeddings_file = config.EMBEDDINGS_FILE
    index_file = config.FAISS_INDEX_FILE

    if config.USE_TRIPLET_MODEL and config.TRIPLET_MODEL_PATH.exists():
        triplet_model_path = str(config.TRIPLET_MODEL_PATH)
        embeddings_file = config.TRIPLET_EMBEDDINGS_FILE
        index_file = config.TRIPLET_FAISS_INDEX
        logger.info(f"✅ Using trained Triplet Network")
    else:
        logger.info(f"📌 Using pretrained {config.MODEL_NAME}")

    # Initialize feature extractor
    feature_extractor = FeatureExtractor(triplet_model_path=triplet_model_path)
    similarity_search = SimilaritySearch(use_faiss=True)

    # Try to load existing index
    try:
        if (index_file.exists() and
                config.IMAGE_PATHS_FILE.exists() and
                embeddings_file.exists()):
            logger.info("📥 Loading existing search index...")
            similarity_search.load_index(
                index_path=index_file,
                paths_path=config.IMAGE_PATHS_FILE,
                features_path=embeddings_file
            )
            index_loaded = True
            logger.info("✅ Search index loaded successfully")
            return True

    except Exception as e:
        logger.warning(f"⚠️  Could not load existing index: {e}")

    # Build new index
    logger.info("🔨 Building new search index...")

    try:
        image_paths = get_image_paths(config.IMAGES_FOLDER)

        if len(image_paths) == 0:
            logger.warning("⚠️  No images found in dataset folder")
            return False

        logger.info(f"📸 Extracting features from {len(image_paths)} images...")
        features = feature_extractor.extract_features_batch(image_paths)

        logger.info("🔍 Building search index...")
        similarity_search.build_index(features, image_paths)

        logger.info("💾 Saving index...")
        similarity_search.save_index(
            index_path=index_file,
            paths_path=config.IMAGE_PATHS_FILE
        )

        feature_extractor.save_features(features, embeddings_file)

        index_loaded = True
        logger.info(f"✅ Search index built for {len(image_paths)} images")
        return True

    except Exception as e:
        logger.error(f"❌ Error building index: {str(e)}")
        return False


@app.route('/')
def index():
    """Home page with upload form"""
    stats = similarity_search.get_statistics() if index_loaded else None
    model_info = feature_extractor.get_model_info() if feature_extractor else None
    return render_template('index.html', stats=stats, index_loaded=index_loaded, model_info=model_info)


@app.route('/search', methods=['POST'])
def search():
    """Handle image upload and search"""
    logger.info("=" * 50)
    logger.info("NEW SEARCH REQUEST RECEIVED")
    logger.info("=" * 50)

    if not index_loaded:
        logger.error("Search index not initialized")
        flash('Search index not initialized. Please wait or check logs.', 'error')
        return redirect(url_for('index'))

    if 'file' not in request.files:
        logger.error("No file in request")
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    logger.info(f"File received: {file.filename}")

    if file.filename == '':
        logger.error("Empty filename")
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            logger.info("Saving uploaded file...")
            filepath = save_uploaded_file(file)

            if filepath is None:
                logger.error("Failed to save file")
                flash('Error saving file', 'error')
                return redirect(url_for('index'))

            logger.info(f"File saved to: {filepath}")

            logger.info("Validating image...")
            load_and_validate_image(filepath)
            logger.info("Image validated successfully")

            logger.info(f"Extracting features from {filepath}")
            query_features = feature_extractor.extract_features(filepath)
            logger.info(f"Features extracted. Shape: {query_features.shape}")

            top_k = int(request.form.get('top_k', config.TOP_K))
            top_k = min(max(top_k, 1), 50)
            logger.info(f"Searching for top {top_k} similar images")

            results = similarity_search.search(query_features, top_k=top_k)
            logger.info(f"Search completed. Found {len(results)} results")

            if len(results) > 0:
                logger.info("--- Top Similarity Scores ---")
                for idx, (path, score) in enumerate(results[:5]):
                    logger.info(f"  Top {idx + 1}: score={score:.4f} ({score * 100:.2f}%)")

            if len(results) == 0:
                logger.warning("No similar images found")
                return render_template('result.html',
                                       query_image=os.path.basename(filepath),
                                       similar_images=[],
                                       num_results=0,
                                       model_info=feature_extractor.get_model_info())

            query_image = os.path.basename(filepath)
            logger.info(f"Query image filename: {query_image}")

            similar_images = []
            for idx, (img_path, score) in enumerate(results):
                logger.info(f"  Result {idx + 1}: {img_path} (score: {score:.4f})")

                # Convert absolute filesystem path -> URL path
                if os.path.isabs(img_path):
                    rel_path = os.path.relpath(img_path, config.DATA_FOLDER)
                else:
                    rel_path = img_path

                rel_path = rel_path.replace('\\', '/')

                from flask import url_for  # make sure this import exists

                rel_path = os.path.relpath(img_path, config.DATA_FOLDER)
                rel_path = rel_path.replace("\\", "/")

                similar_images.append({
                    "path": url_for("serve_data_file", filename=rel_path),
                    "score": round(score * 100, 2),
                    "filename": os.path.basename(img_path)
                })

            logger.info(f"Rendering result page with {len(similar_images)} images")
            logger.info("=" * 50)

            return render_template('result.html',
                                   query_image=query_image,
                                   similar_images=similar_images,
                                   num_results=len(similar_images),
                                   model_info=feature_extractor.get_model_info())

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('error.html',
                                   error_code='Processing Error',
                                   error_message=str(e))
    else:
        logger.error(f"Invalid file type: {file.filename}")
        flash('Invalid file type. Allowed types: ' + ', '.join(config.ALLOWED_EXTENSIONS), 'error')
        return redirect(url_for('index'))


@app.route('/data/<path:filename>')
def serve_data_file(filename):
    """Serve files from data directory"""
    try:
        logger.info(f"Serving data file: {filename}")
        return send_from_directory(config.DATA_FOLDER, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return "File not found", 404


@app.route('/rebuild-index', methods=['POST'])
def rebuild_index():
    """Rebuild the search index"""
    try:
        logger.info("Rebuilding search index...")
        success = initialize_models()

        if success:
            flash('Search index rebuilt successfully', 'success')
        else:
            flash('Error rebuilding search index', 'error')

    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        flash(f'Error: {str(e)}', 'error')

    return redirect(url_for('index'))


@app.route('/stats')
def stats():
    """Get search index statistics as JSON"""
    if index_loaded:
        stats = similarity_search.get_statistics()
        stats['model_info'] = feature_extractor.get_model_info()
        return jsonify(stats)
    else:
        return jsonify({'error': 'Index not loaded'}), 503


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'index_loaded': index_loaded,
        'model_type': feature_extractor.get_model_info()['model_type'] if feature_extractor else 'none'
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('error.html',
                           error_code=404,
                           error_message='Page not found'), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return render_template('error.html',
                           error_code=500,
                           error_message='Internal server error'), 500


if __name__ == '__main__':
    try:
        # Print startup info
        config.print_startup_info()

        # Clean old uploads
        logger.info("🧹 Cleaning old upload files...")
        clean_upload_folder()

        # Check and train if needed
        if not config.USE_TRIPLET_MODEL:
            logger.info("🔍 No trained model detected")
            if config.AUTO_TRAIN_ON_FIRST_RUN:
                logger.info("🚀 Attempting auto-train...")
                trained = check_and_train_if_needed()
                if not trained:
                    logger.warning("⚠️  Could not auto-train. Using pretrained model as fallback.")
            else:
                logger.info("📝 To train: python train_triplet.py")

        # Initialize models and search index
        logger.info("🔧 Initializing models...")
        success = initialize_models()

        if not success:
            logger.error("❌ Failed to initialize. Check logs above.")
            logger.info("\n💡 Quick fixes:")
            logger.info(f"   1. Add images to: {config.IMAGES_FOLDER}")
            logger.info(f"   2. Organize in subfolders: images/category1/, images/category2/")
            logger.info(f"   3. Run: python train_triplet.py")
            sys.exit(1)

        # Start Flask app
        logger.info("=" * 70)
        logger.info("🌐 STARTING WEB SERVER")
        logger.info("=" * 70)
        logger.info(f"   URL: http://localhost:{config.FLASK_PORT}")
        logger.info(f"   Ready to accept requests!")
        logger.info("=" * 70)
        logger.info("")

        app.run(
            debug=config.FLASK_DEBUG,
            host=config.FLASK_HOST,
            port=config.FLASK_PORT
        )

    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)