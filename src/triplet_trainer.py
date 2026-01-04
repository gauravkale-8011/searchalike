import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
from .triplet_network import TripletLoss
from datetime import datetime

logger = logging.getLogger(__name__)


class TripletTrainer:
    """
    Trainer for Triplet Networks.
    """

    def __init__(self, model, train_loader, val_loader=None,
                 loss_fn=None, optimizer=None, device='cuda',
                 checkpoint_dir='checkpoints', log_dir='runs'):
        """
        Initialize trainer.

        Args:
            model: Triplet network model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        if loss_fn is None:
            from triplet_network import TripletLoss
            self.loss_fn = TripletLoss(margin=1.0)
        else:
            self.loss_fn = loss_fn

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=0.0001,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        self.best_val_loss = float('inf')
        self.current_epoch = 0

        logger.info(f"TripletTrainer initialized on {device}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')

        for batch in progress_bar:
            # Handle both triplet and online modes
            if len(batch) == 4:  # Offline triplet mode
                anchor, positive, negative, labels = batch
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # Forward pass
                anchor_emb, pos_emb, neg_emb = self.model(anchor, positive, negative)
                loss = self.loss_fn(anchor_emb, pos_emb, neg_emb)

            elif len(batch) == 2:  # Online triplet mode
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                embeddings = self.model.get_embedding(images)
                loss = self.loss_fn(embeddings, labels)

            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """Validate on validation set."""
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                if len(batch) == 4:  # Offline mode
                    anchor, positive, negative, labels = batch
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)

                    anchor_emb, pos_emb, neg_emb = self.model(anchor, positive, negative)
                    loss = self.loss_fn(anchor_emb, pos_emb, neg_emb)

                elif len(batch) == 2:  # Online mode
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    embeddings = self.model.get_embedding(images)
                    loss = self.loss_fn(embeddings, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, num_epochs, save_every=5, early_stopping_patience=10):
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement after N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                self.history['val_loss'].append(val_loss)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # TensorBoard logging
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/validation', val_loss, epoch)

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    epochs_without_improvement = 0
                    self.save_checkpoint('best_model.pth', is_best=True)
                    logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
                else:
                    epochs_without_improvement += 1

                # Early stopping
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"(no improvement for {early_stopping_patience} epochs)"
                    )
                    break

            else:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
                self.writer.add_scalar('Loss/train', train_loss, epoch)

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

        # Save final model
        self.save_checkpoint('final_model.pth')

        # Save training history
        self.save_history()

        logger.info("Training completed!")
        self.writer.close()

    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)

        logger.info(f"Training history saved: {history_path}")


def get_default_transforms(augment=True):
    """
    Get default image transforms for training.

    Args:
        augment: Whether to apply data augmentation

    Returns:
        torchvision.transforms.Compose
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def evaluate_embeddings(model, dataloader, device='cuda'):
    """
    Evaluate quality of learned embeddings.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting embeddings'):
            if len(batch) == 2:
                images, labels = batch
            else:
                images, _, _, labels = batch

            images = images.to(device)
            embeddings = model.get_embedding(images)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    # Calculate metrics
    from sklearn.metrics.pairwise import cosine_similarity

    # Intra-class similarity (same label)
    intra_class_sims = []
    # Inter-class similarity (different label)
    inter_class_sims = []

    unique_labels = np.unique(all_labels)

    for label in unique_labels:
        label_mask = all_labels == label
        label_embeddings = all_embeddings[label_mask]

        if len(label_embeddings) > 1:
            # Intra-class similarity
            intra_sim = cosine_similarity(label_embeddings)
            # Remove diagonal
            mask = ~np.eye(intra_sim.shape[0], dtype=bool)
            intra_class_sims.extend(intra_sim[mask].flatten())

        # Inter-class similarity
        other_embeddings = all_embeddings[~label_mask]
        if len(other_embeddings) > 0:
            inter_sim = cosine_similarity(label_embeddings, other_embeddings)
            inter_class_sims.extend(inter_sim.flatten())

    metrics = {
        'intra_class_similarity_mean': np.mean(intra_class_sims),
        'intra_class_similarity_std': np.std(intra_class_sims),
        'inter_class_similarity_mean': np.mean(inter_class_sims),
        'inter_class_similarity_std': np.std(inter_class_sims),
        'separation': np.mean(intra_class_sims) - np.mean(inter_class_sims)
    }

    logger.info("Embedding Evaluation Metrics:")
    logger.info(f"  Intra-class similarity: {metrics['intra_class_similarity_mean']:.4f} "
                f"± {metrics['intra_class_similarity_std']:.4f}")
    logger.info(f"  Inter-class similarity: {metrics['inter_class_similarity_mean']:.4f} "
                f"± {metrics['inter_class_similarity_std']:.4f}")
    logger.info(f"  Separation: {metrics['separation']:.4f}")

    return metrics


def freeze_backbone(model, freeze=True):
    """
    Freeze or unfreeze the backbone (feature extractor) of the model.

    Args:
        model: TripletNetwork model
        freeze: Whether to freeze the backbone
    """
    # Get the feature extractor part (first layer of embedding_net)
    feature_extractor = model.embedding_net[0]

    for param in feature_extractor.parameters():
        param.requires_grad = not freeze

    if freeze:
        logger.info("Backbone frozen - only training projection head")
    else:
        logger.info("Backbone unfrozen - training all layers")