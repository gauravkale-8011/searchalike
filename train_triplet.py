"""
Main training script for Triplet Network.

Usage:
    python train_triplet.py --mode online --epochs 50 --batch_size 32
"""

import torch
import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.triplet_network import TripletNetwork, TripletLoss, OnlineTripletLoss
from src.triplet_dataset import (
    create_triplet_dataloader,
    generate_labels_from_folders,
    generate_labels_from_filenames,
    create_synthetic_labels
)
from src.triplet_trainer import (
    TripletTrainer,
    get_default_transforms,
    evaluate_embeddings,
    freeze_backbone
)
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Triplet Network for Image Similarity')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/images',
                        help='Directory containing training images')
    parser.add_argument('--label_mode', type=str, default='folders',
                        choices=['folders', 'filenames', 'synthetic'],
                        help='How to generate labels')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')

    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet18', 'efficientnet_b0'],
                        help='Backbone architecture')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone during initial training')

    # Training parameters
    parser.add_argument('--mode', type=str, default='online',
                        choices=['online', 'offline'],
                        help='Triplet mining mode')
    parser.add_argument('--loss_type', type=str, default='hardest',
                        choices=['hardest', 'semi-hard', 'all', 'basic'],
                        help='Type of triplet loss')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for triplet loss')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')

    # Optimization
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    # Directories
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='Directory for TensorBoard logs')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def load_data(args):
    """Load and prepare data."""
    logger.info("Loading data...")

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Generate labels based on mode
    if args.label_mode == 'folders':
        logger.info("Generating labels from folder structure...")
        image_paths, labels = generate_labels_from_folders(data_dir)

    elif args.label_mode == 'filenames':
        logger.info("Generating labels from filenames...")
        # Get all images
        from src.utils import get_image_paths
        image_paths = get_image_paths(data_dir)
        labels = generate_labels_from_filenames(image_paths)

    elif args.label_mode == 'synthetic':
        logger.info("Generating synthetic labels using clustering...")
        from src.utils import get_image_paths
        image_paths = get_image_paths(data_dir)
        n_clusters = min(50, len(image_paths) // 10)  # Heuristic
        labels = create_synthetic_labels(image_paths, n_clusters=n_clusters)

    else:
        raise ValueError(f"Unknown label mode: {args.label_mode}")

    logger.info(f"Loaded {len(image_paths)} images with {len(set(labels))} unique labels")

    # Convert string labels to numeric
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    numeric_labels = [label_to_idx[label] for label in labels]

    # Train/val split
    from sklearn.model_selection import train_test_split

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, numeric_labels,
        test_size=args.val_split,
        stratify=numeric_labels,
        random_state=42
    )

    logger.info(f"Train: {len(train_paths)} images, Val: {len(val_paths)} images")

    return train_paths, val_paths, train_labels, val_labels


def create_model(args):
    """Create triplet network model."""
    logger.info("Creating model...")

    model = TripletNetwork(
        embedding_dim=args.embedding_dim,
        pretrained=args.pretrained,
        backbone=args.backbone
    )

    # Freeze backbone if requested
    if args.freeze_backbone:
        freeze_backbone(model, freeze=True)

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, using CPU")
        args.device = 'cpu'

    return model


def create_loss_function(args):
    """Create loss function."""
    logger.info(f"Creating {args.loss_type} triplet loss with margin={args.margin}")

    if args.mode == 'online':
        loss_fn = OnlineTripletLoss(
            margin=args.margin,
            triplet_selector=args.loss_type
        )
    else:
        loss_fn = TripletLoss(
            margin=args.margin,
            distance='euclidean'
        )

    return loss_fn


def main():
    """Main training function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("TRIPLET NETWORK TRAINING")
    logger.info("=" * 60)
    logger.info(f"Configuration: {vars(args)}")

    # Load data
    train_paths, val_paths, train_labels, val_labels = load_data(args)

    # Create transforms
    train_transform = get_default_transforms(augment=True)
    val_transform = get_default_transforms(augment=False)

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_triplet_dataloader(
        train_paths, train_labels, train_transform,
        batch_size=args.batch_size,
        mode=args.mode,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = create_triplet_dataloader(
        val_paths, val_labels, val_transform,
        batch_size=args.batch_size,
        mode=args.mode,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Create model
    model = create_model(args)

    # Create loss function
    loss_fn = create_loss_function(args)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create trainer
    trainer = TripletTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping
    )

    # Evaluate final embeddings
    logger.info("Evaluating final embeddings...")
    metrics = evaluate_embeddings(
        model=model,
        dataloader=val_loader,
        device=args.device
    )

    logger.info("Training completed successfully!")
    logger.info(f"Best model saved at: {trainer.checkpoint_dir / 'best_model.pth'}")

    return trainer


if __name__ == '__main__':
    try:
        trainer = main()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)