import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import logging
from pathlib import Path
from typing import List, Union
import time
from tqdm import tqdm

import config
from src.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extractor for image similarity.
    Supports:
    - Pretrained CNNs (ResNet50, VGG16, EfficientNet)
    - Trained Triplet Network
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        device: str = config.DEVICE,
        triplet_model_path: str = None
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        self.triplet_model_path = triplet_model_path
        self.use_triplet = triplet_model_path is not None

        if self.use_triplet:
            self.model = self._load_triplet_model(triplet_model_path)
            logger.info(f"✅ Loaded Triplet model: {triplet_model_path}")
        else:
            self.model = self._load_pretrained_model()
            logger.info(f"✅ Loaded pretrained {model_name}")

        self.model.eval()
        self.model.to(self.device)

        # Freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.preprocessor = ImagePreprocessor()

    # ------------------------------------------------------------------
    # MODEL LOADERS
    # ------------------------------------------------------------------

    def _load_pretrained_model(self) -> nn.Module:
        """Load pretrained backbone for feature extraction"""

        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model = nn.Sequential(*list(model.children())[:-1])

        elif self.model_name == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            model = model.features

        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            model = nn.Sequential(*list(model.children())[:-1])

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model

    def _load_triplet_model(self, checkpoint_path: str) -> nn.Module:
        """Load trained Triplet Network"""

        from src.triplet_network import TripletNetwork

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = TripletNetwork(
            embedding_dim=config.TRIPLET_EMBEDDING_DIM,
            pretrained=False,
            backbone=config.TRIPLET_BACKBONE
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------------------------

    def extract_features(self, image: Union[str, torch.Tensor]) -> np.ndarray:
        """Extract features from a single image"""

        if isinstance(image, str):
            tensor = self.preprocessor.preprocess(image)
        else:
            tensor = image

        tensor = tensor.to(self.device)

        with torch.no_grad():
            if self.use_triplet:
                features = self.model.get_embedding(tensor)
            else:
                features = self.model(tensor)

        features = features.squeeze().cpu().numpy()

        # Normalize for cosine similarity
        features = features / (np.linalg.norm(features) + 1e-8)

        return features

    

    def extract_features_batch(self, images, batch_size=64):
        """
        Extract features from images with progress bar and ETA.
        """
        total_images = len(images)
        all_features = []

        num_batches = (total_images + batch_size - 1) // batch_size
        start_time = time.time()

        pbar = tqdm(
            total=num_batches,
            desc="🔍 Extracting embeddings",
            unit="batch"
        )

        for i in range(0, total_images, batch_size):
            batch_paths = images[i:i + batch_size]
            batch_tensors = []

            for path in batch_paths:
                tensor = self.preprocessor.preprocess(path)
                batch_tensors.append(tensor)

            batch = torch.cat(batch_tensors, dim=0).to(self.device)

            with torch.no_grad():
                if self.use_triplet:
                    feats = self.model.get_embedding(batch)
                else:
                    feats = self.model(batch)

            feats = feats.cpu().numpy()
            feats = feats.reshape(len(batch_paths), -1)

            # Normalize
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            feats = feats / norms

            all_features.append(feats)

            # ETA calculation
            batches_done = (i // batch_size) + 1
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / batches_done
            remaining = avg_time_per_batch * (num_batches - batches_done)

            pbar.set_postfix({
                "images": min(i + batch_size, total_images),
                "elapsed": f"{elapsed/60:.1f}m",
                "eta": f"{remaining/60:.1f}m"
            })

            pbar.update(1)

        pbar.close()

        features = np.vstack(all_features)
        return features


    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------

    def save_features(self, features: np.ndarray, path: Path):
        """Save embeddings to disk (FIXES YOUR ISSUE)"""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features)
        logger.info(f"💾 Features saved to {path}")

    # ------------------------------------------------------------------
    # INFO
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        return {
            "model_type": "TripletNetwork" if self.use_triplet else self.model_name,
            "embedding_dim": (
                config.TRIPLET_EMBEDDING_DIM if self.use_triplet else config.FEATURE_DIM
            ),
            "device": str(self.device)
        }