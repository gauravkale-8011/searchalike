"""
Visualization tools for analyzing embeddings.

Usage:
    python visualize_embeddings.py --checkpoint checkpoints/best_model.pth
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from pathlib import Path
import argparse
import logging

from src.triplet_network import TripletNetwork
from src.triplet_dataset import create_triplet_dataloader, generate_labels_from_folders
from src.triplet_trainer import get_default_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_all_embeddings(model, dataloader, device='cuda'):
    """
    Extract embeddings for all images in dataloader.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use

    Returns:
        embeddings: numpy array of embeddings
        labels: numpy array of labels
        paths: list of image paths
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    logger.info("Extracting embeddings...")

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, _, _, labels = batch

            images = images.to(device)
            embeddings = model.get_embedding(images)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)

    logger.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    return embeddings, labels


def plot_tsne(embeddings, labels, save_path='visualizations/tsne.png', n_classes=None):
    """
    Plot t-SNE visualization of embeddings.

    Args:
        embeddings: Embedding vectors
        labels: Class labels
        save_path: Path to save figure
        n_classes: Number of classes to show (None for all)
    """
    logger.info("Computing t-SNE...")

    # Limit to n_classes if specified
    if n_classes is not None:
        unique_labels = np.unique(labels)[:n_classes]
        mask = np.isin(labels, unique_labels)
        embeddings = embeddings[mask]
        labels = labels[mask]

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='tab20',
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Learned Embeddings', fontsize=16)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"t-SNE plot saved to {save_path}")
    plt.close()


def plot_pca(embeddings, labels, save_path='visualizations/pca.png', n_classes=None):
    """
    Plot PCA visualization of embeddings.

    Args:
        embeddings: Embedding vectors
        labels: Class labels
        save_path: Path to save figure
        n_classes: Number of classes to show
    """
    logger.info("Computing PCA...")

    # Limit to n_classes if specified
    if n_classes is not None:
        unique_labels = np.unique(labels)[:n_classes]
        mask = np.isin(labels, unique_labels)
        embeddings = embeddings[mask]
        labels = labels[mask]

    # Compute PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='tab20',
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter, label='Class')
    plt.title(f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})',
              fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"PCA plot saved to {save_path}")
    plt.close()


def plot_similarity_matrix(embeddings, labels, save_path='visualizations/similarity_matrix.png'):
    """
    Plot similarity matrix heatmap.

    Args:
        embeddings: Embedding vectors
        labels: Class labels
        save_path: Path to save figure
    """
    logger.info("Computing similarity matrix...")

    from sklearn.metrics.pairwise import cosine_similarity

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Sort by labels for better visualization
    sorted_indices = np.argsort(labels)
    sim_matrix = sim_matrix[sorted_indices][:, sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sim_matrix,
        cmap='RdYlGn',
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('Cosine Similarity Matrix (Sorted by Class)', fontsize=16)
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Similarity matrix saved to {save_path}")
    plt.close()


def plot_class_distribution(embeddings, labels, save_path='visualizations/class_distribution.png'):
    """
    Plot distribution of embeddings per class.

    Args:
        embeddings: Embedding vectors
        labels: Class labels
        save_path: Path to save figure
    """
    logger.info("Plotting class distribution...")

    from sklearn.metrics.pairwise import euclidean_distances

    unique_labels = np.unique(labels)

    # Compute centroid for each class
    centroids = []
    distances_to_centroid = []

    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings[mask]
        centroid = class_embeddings.mean(axis=0)
        centroids.append(centroid)

        # Compute distances to centroid
        distances = euclidean_distances(class_embeddings, [centroid]).flatten()
        distances_to_centroid.append(distances)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot of distances
    ax1.boxplot(distances_to_centroid, labels=unique_labels)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Distance to Centroid')
    ax1.set_title('Distribution of Distances to Class Centroids')
    ax1.grid(True, alpha=0.3)

    # Histogram of all distances
    all_distances = np.concatenate(distances_to_centroid)
    ax2.hist(all_distances, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Distance to Centroid')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Overall Distribution of Distances')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Class distribution plot saved to {save_path}")
    plt.close()


def plot_training_history(history_path='checkpoints/training_history.json',
                          save_path='visualizations/training_history.png'):
    """
    Plot training history from saved JSON.

    Args:
        history_path: Path to training history JSON
        save_path: Path to save figure
    """
    import json

    logger.info("Plotting training history...")

    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate
    if 'learning_rates' in history:
        ax2.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved to {save_path}")
    plt.close()


def compare_models(pretrained_embeddings, triplet_embeddings, labels,
                   save_path='visualizations/model_comparison.png'):
    """
    Compare pretrained vs triplet model embeddings.

    Args:
        pretrained_embeddings: Embeddings from pretrained model
        triplet_embeddings: Embeddings from triplet model
        labels: Class labels
        save_path: Path to save figure
    """
    logger.info("Comparing models...")

    from sklearn.metrics.pairwise import cosine_similarity

    # Compute metrics
    def compute_metrics(embeddings, labels):
        sim_matrix = cosine_similarity(embeddings)

        intra_class = []
        inter_class = []

        for i, label in enumerate(labels):
            same_class = labels == label
            diff_class = labels != label

            # Remove self-similarity
            same_class[i] = False

            if same_class.sum() > 0:
                intra_class.extend(sim_matrix[i, same_class])
            if diff_class.sum() > 0:
                inter_class.extend(sim_matrix[i, diff_class])

        return {
            'intra_mean': np.mean(intra_class),
            'inter_mean': np.mean(inter_class),
            'separation': np.mean(intra_class) - np.mean(inter_class)
        }

    pretrained_metrics = compute_metrics(pretrained_embeddings, labels)
    triplet_metrics = compute_metrics(triplet_embeddings, labels)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Metrics comparison
    metrics_names = ['Intra-class\nSimilarity', 'Inter-class\nSimilarity', 'Separation']
    pretrained_values = [
        pretrained_metrics['intra_mean'],
        pretrained_metrics['inter_mean'],
        pretrained_metrics['separation']
    ]
    triplet_values = [
        triplet_metrics['intra_mean'],
        triplet_metrics['inter_mean'],
        triplet_metrics['separation']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax1.bar(x - width / 2, pretrained_values, width, label='Pretrained', alpha=0.8)
    ax1.bar(x + width / 2, triplet_values, width, label='Triplet', alpha=0.8)
    ax1.set_ylabel('Score')
    ax1.set_title('Model Comparison Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Improvement percentage
    improvements = [
        (triplet_values[i] - pretrained_values[i]) / abs(pretrained_values[i]) * 100
        for i in range(len(metrics_names))
    ]

    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax2.bar(metrics_names, improvements, color=colors, alpha=0.8)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Triplet Model Improvement over Pretrained')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Model comparison plot saved to {save_path}")
    plt.close()

    # Print metrics
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(f"\nPretrained Model:")
    print(f"  Intra-class similarity: {pretrained_metrics['intra_mean']:.4f}")
    print(f"  Inter-class similarity: {pretrained_metrics['inter_mean']:.4f}")
    print(f"  Separation: {pretrained_metrics['separation']:.4f}")
    print(f"\nTriplet Model:")
    print(f"  Intra-class similarity: {triplet_metrics['intra_mean']:.4f}")
    print(f"  Inter-class similarity: {triplet_metrics['inter_mean']:.4f}")
    print(f"  Separation: {triplet_metrics['separation']:.4f}")
    print(f"\nImprovement:")
    for i, name in enumerate(metrics_names):
        print(f"  {name.replace(chr(10), ' ')}: {improvements[i]:+.2f}%")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize embeddings')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/images',
                        help='Data directory')
    parser.add_argument('--n_classes', type=int, default=None,
                        help='Number of classes to visualize (None for all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    image_paths, labels = generate_labels_from_folders(args.data_dir)

    # Convert labels to numeric
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    numeric_labels = [label_to_idx[label] for label in labels]

    # Create dataloader
    transform = get_default_transforms(augment=False)
    from src.triplet_dataset import create_triplet_dataloader

    dataloader = create_triplet_dataloader(
        image_paths, numeric_labels, transform,
        batch_size=32, mode='online', shuffle=False, num_workers=4
    )

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = TripletNetwork(embedding_dim=128, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    # Extract embeddings
    embeddings, labels_array = extract_all_embeddings(model, dataloader, args.device)

    # Generate visualizations
    logger.info("Generating visualizations...")

    plot_tsne(embeddings, labels_array,
              f'{args.output_dir}/tsne.png', args.n_classes)

    plot_pca(embeddings, labels_array,
             f'{args.output_dir}/pca.png', args.n_classes)

    plot_similarity_matrix(embeddings, labels_array,
                           f'{args.output_dir}/similarity_matrix.png')

    plot_class_distribution(embeddings, labels_array,
                            f'{args.output_dir}/class_distribution.png')

    # Plot training history if available
    history_path = Path(args.checkpoint).parent / 'training_history.json'
    if history_path.exists():
        plot_training_history(history_path,
                              f'{args.output_dir}/training_history.png')

    logger.info(f"All visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    main()