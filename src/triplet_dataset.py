import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from pathlib import Path
from collections import defaultdict
import logging
import config

logger = logging.getLogger(__name__)

# ============================================================
# DATASETS
# ============================================================

class TripletDataset(Dataset):
    """
    Dataset for offline triplet learning.
    Generates (anchor, positive, negative) triplets on-the-fly.
    """

    def __init__(self, image_paths, labels, transform=None, triplets_per_anchor=5):
        self.image_paths = image_paths          # RELATIVE paths
        self.labels = labels
        self.transform = transform
        self.triplets_per_anchor = triplets_per_anchor

        self.label_to_indices = self._build_label_mapping()
        self.unique_labels = list(self.label_to_indices.keys())

        logger.info(
            f"TripletDataset initialized with {len(image_paths)} images, "
            f"{len(self.unique_labels)} unique labels"
        )

    def _build_label_mapping(self):
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            label_to_indices[label].append(idx)
        return dict(label_to_indices)

    def __len__(self):
        return len(self.image_paths) * self.triplets_per_anchor

    def __getitem__(self, idx):
        anchor_idx = idx // self.triplets_per_anchor
        anchor_label = self.labels[anchor_idx]

        positive_indices = self.label_to_indices[anchor_label].copy()
        positive_indices.remove(anchor_idx)

        positive_idx = (
            random.choice(positive_indices)
            if len(positive_indices) > 0
            else anchor_idx
        )

        negative_label = random.choice(
            [l for l in self.unique_labels if l != anchor_label]
        )
        negative_idx = random.choice(self.label_to_indices[negative_label])

        anchor_img = self._load_image(self.image_paths[anchor_idx])
        positive_img = self._load_image(self.image_paths[positive_idx])
        negative_img = self._load_image(self.image_paths[negative_idx])

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_label

    def _load_image(self, image_path):
        """Load image from RELATIVE path safely."""
        try:
            abs_path = config.BASE_DIR / image_path
            img = Image.open(abs_path).convert("RGB")
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return Image.new("RGB", (224, 224), color="white")


class OnlineTripletDataset(Dataset):
    """
    Dataset for online triplet mining.
    Returns (image, label).
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths          # RELATIVE paths
        self.labels = labels
        self.transform = transform

        logger.info(
            f"OnlineTripletDataset initialized with {len(image_paths)} images"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def _load_image(self, image_path):
        """Load image from RELATIVE path safely."""
        try:
            abs_path = config.BASE_DIR / image_path
            img = Image.open(abs_path).convert("RGB")
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return Image.new("RGB", (224, 224), color="white")


# ============================================================
# BALANCED SAMPLER
# ============================================================

class BalancedBatchSampler:
    """
    Sampler that yields batches with P classes × K samples.
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.n_classes = n_classes
        self.n_samples = n_samples

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.unique_labels = list(self.label_to_indices.keys())
        self.n_batches = len(labels) // (n_classes * n_samples)

        logger.info(
            f"BalancedBatchSampler: {n_classes} classes × {n_samples} samples "
            f"= {n_classes * n_samples} per batch"
        )

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            selected_classes = random.sample(
                self.unique_labels,
                min(self.n_classes, len(self.unique_labels))
            )

            for label in selected_classes:
                indices = self.label_to_indices[label]
                selected = (
                    random.sample(indices, self.n_samples)
                    if len(indices) >= self.n_samples
                    else random.choices(indices, k=self.n_samples)
                )
                batch.extend(selected)

            yield batch

    def __len__(self):
        return self.n_batches


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_triplet_dataloader(
    image_paths,
    labels,
    transform,
    batch_size=32,
    mode="online",
    shuffle=True,
    num_workers=4,
):
    if mode == "online":
        dataset = OnlineTripletDataset(image_paths, labels, transform)

        if shuffle:
            n_classes = min(8, len(set(labels)))
            n_samples = batch_size // n_classes

            if n_samples > 0:
                sampler = BalancedBatchSampler(labels, n_classes, n_samples)
                return DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,
                )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    elif mode == "offline":
        dataset = TripletDataset(image_paths, labels, transform)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# LABEL GENERATION (FIXED)
# ============================================================

def generate_labels_from_folders(data_folder):
    """
    Generate labels from folder structure.
    Stores RELATIVE image paths.
    """
    data_folder = Path(data_folder)
    image_paths = []
    labels = []

    subfolders = [f for f in data_folder.iterdir() if f.is_dir()]
    if not subfolders:
        raise ValueError(f"No subfolders found in {data_folder}")

    for folder in subfolders:
        label = folder.name
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            for img_path in folder.glob(ext):
                relative_path = img_path.relative_to(config.BASE_DIR)
                image_paths.append(str(relative_path).replace("\\", "/"))
                labels.append(label)

    logger.info(
        f"Found {len(image_paths)} images across {len(set(labels))} classes"
    )
    return image_paths, labels


def generate_labels_from_filenames(image_paths, delimiter="_", label_position=0):
    labels = []
    for path in image_paths:
        filename = Path(path).stem
        parts = filename.split(delimiter)
        labels.append(parts[label_position] if len(parts) > label_position else "unknown")
    return labels


def create_synthetic_labels(image_paths, n_clusters=10):
    """
    Uses pretrained ResNet to cluster images when labels are unavailable.
    """
    from sklearn.cluster import KMeans
    from torchvision import models, transforms
    from torchvision.models import ResNet50_Weights

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():
        for img_path in image_paths:
            try:
                img = Image.open(config.BASE_DIR / img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)
                feat = model(img).squeeze().cpu().numpy()
                features.append(feat)
            except Exception:
                features.append(np.zeros(2048))

    features = np.array(features)
    labels = KMeans(n_clusters=n_clusters, n_init=10).fit_predict(features)
    return labels.tolist()
