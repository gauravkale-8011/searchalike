import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
import logging

logger = logging.getLogger(__name__)


class TripletNetwork(nn.Module):
    """
    Triplet Network for learning image similarity using triplet loss.
    Uses a shared embedding network for all three inputs (anchor, positive, negative).
    """

    def __init__(self, embedding_dim=128, pretrained=True, backbone='resnet50'):
        """
        Initialize Triplet Network.

        Args:
            embedding_dim: Dimension of output embeddings
            pretrained: Whether to use pretrained weights
            backbone: Base architecture ('resnet50', 'resnet18', 'efficientnet_b0')
        """
        super(TripletNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.backbone = backbone

        # Create embedding network (shared across triplets)
        self.embedding_net = self._create_embedding_network(pretrained)

        logger.info(f"TripletNetwork initialized with {backbone} backbone, "
                    f"embedding_dim={embedding_dim}")

    def _create_embedding_network(self, pretrained):
        """Create the embedding network based on chosen backbone."""

        if self.backbone == 'resnet50':
            # Load ResNet50
            base_model = models.resnet50(
                weights=ResNet50_Weights.DEFAULT if pretrained else None
            )
            # Remove the final classification layer
            feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 2048

        elif self.backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 512

        elif self.backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            feature_dim = 1280

        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

        # Create embedding network with projection head
        embedding_net = nn.Sequential(
            feature_extractor,
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim)
        )

        return embedding_net

    def forward_once(self, x):
        """
        Forward pass for a single image.

        Args:
            x: Input image tensor

        Returns:
            Normalized embedding vector
        """
        output = self.embedding_net(x)
        # L2 normalize embeddings
        output = F.normalize(output, p=2, dim=1)
        return output

    def forward(self, anchor, positive, negative):
        """
        Forward pass for triplet.

        Args:
            anchor: Anchor image tensor
            positive: Positive (similar) image tensor
            negative: Negative (dissimilar) image tensor

        Returns:
            Tuple of (anchor_embedding, positive_embedding, negative_embedding)
        """
        anchor_embedding = self.forward_once(anchor)
        positive_embedding = self.forward_once(positive)
        negative_embedding = self.forward_once(negative)

        return anchor_embedding, positive_embedding, negative_embedding

    def get_embedding(self, x):
        """
        Get embedding for a single image (for inference).

        Args:
            x: Input image tensor

        Returns:
            Embedding vector
        """
        return self.forward_once(x)


class TripletLoss(nn.Module):
    """
    Triplet loss function.
    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    """

    def __init__(self, margin=1.0, distance='euclidean'):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
            distance: Distance metric ('euclidean' or 'cosine')
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance

        logger.info(f"TripletLoss initialized with margin={margin}, distance={distance}")

    def forward(self, anchor, positive, negative):
        """
        Calculate triplet loss.

        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings

        Returns:
            Triplet loss value
        """
        if self.distance == 'euclidean':
            # Euclidean distance
            pos_distance = F.pairwise_distance(anchor, positive, p=2)
            neg_distance = F.pairwise_distance(anchor, negative, p=2)

        elif self.distance == 'cosine':
            # Cosine distance (1 - cosine similarity)
            pos_distance = 1 - F.cosine_similarity(anchor, positive)
            neg_distance = 1 - F.cosine_similarity(anchor, negative)

        else:
            raise ValueError(f"Unsupported distance metric: {self.distance}")

        # Triplet loss with margin
        losses = F.relu(pos_distance - neg_distance + self.margin)

        return losses.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Loss - mines triplets within a batch.
    More efficient than pre-computing all triplets.
    """

    def __init__(self, margin=1.0, triplet_selector='hardest'):
        """
        Initialize online triplet loss.

        Args:
            margin: Margin for triplet loss
            triplet_selector: Strategy for mining triplets
                            ('hardest', 'semi-hard', 'all')
        """
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

        logger.info(f"OnlineTripletLoss initialized with margin={margin}, "
                    f"selector={triplet_selector}")

    def forward(self, embeddings, labels):
        """
        Calculate online triplet loss.

        Args:
            embeddings: Batch of embeddings
            labels: Labels for each embedding

        Returns:
            Triplet loss value
        """
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)

        if self.triplet_selector == 'hardest':
            return self._hardest_triplet_loss(pairwise_dist, labels)
        elif self.triplet_selector == 'semi-hard':
            return self._semi_hard_triplet_loss(pairwise_dist, labels)
        elif self.triplet_selector == 'all':
            return self._all_triplet_loss(pairwise_dist, labels)
        else:
            raise ValueError(f"Unknown triplet selector: {self.triplet_selector}")

    def _pairwise_distances(self, embeddings):
        """Compute pairwise Euclidean distances."""
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)

        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)  # Numerical stability

        # Add small epsilon to avoid division by zero
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)  # Zero diagonal

        return distances

    def _hardest_triplet_loss(self, pairwise_dist, labels):
        """Select hardest positive and hardest negative for each anchor."""
        # Get anchor-positive and anchor-negative masks
        labels = labels.unsqueeze(0)
        mask_anchor_positive = (labels == labels.t()).float()
        mask_anchor_negative = (labels != labels.t()).float()

        # Remove diagonal (self-comparisons)
        mask_anchor_positive = mask_anchor_positive - torch.eye(
            labels.size(1), device=labels.device
        )

        # Hardest positive: maximum distance among positives
        hardest_positive_dist = (pairwise_dist * mask_anchor_positive).max(dim=1)[0]

        # Hardest negative: minimum distance among negatives
        # Add large value to positives to exclude them
        max_anchor_negative_dist = (
                pairwise_dist + 1e6 * (1.0 - mask_anchor_negative)
        ).min(dim=1)[0]

        # Triplet loss
        loss = F.relu(hardest_positive_dist - max_anchor_negative_dist + self.margin)

        return loss.mean()

    def _semi_hard_triplet_loss(self, pairwise_dist, labels):
        """Select semi-hard negatives (harder than positive but within margin)."""
        labels = labels.unsqueeze(0)
        mask_anchor_positive = (labels == labels.t()).float()
        mask_anchor_negative = (labels != labels.t()).float()

        mask_anchor_positive = mask_anchor_positive - torch.eye(
            labels.size(1), device=labels.device
        )

        # Hardest positive
        hardest_positive_dist = (pairwise_dist * mask_anchor_positive).max(dim=1)[0]

        # Semi-hard negative: d(a,n) > d(a,p) but d(a,n) < d(a,p) + margin
        semi_hard_negatives = mask_anchor_negative * (
                (pairwise_dist > hardest_positive_dist.unsqueeze(1)).float() *
                (pairwise_dist < (hardest_positive_dist.unsqueeze(1) + self.margin)).float()
        )

        # If no semi-hard negatives, use hardest negative
        num_semi_hard = semi_hard_negatives.sum(dim=1)
        use_hardest = (num_semi_hard == 0).float()

        # Get semi-hard or hardest negative
        semi_hard_dist = (
                (pairwise_dist + 1e6 * (1.0 - semi_hard_negatives)).min(dim=1)[0] *
                (1.0 - use_hardest)
        )
        hardest_dist = (
                (pairwise_dist + 1e6 * (1.0 - mask_anchor_negative)).min(dim=1)[0] *
                use_hardest
        )

        negative_dist = semi_hard_dist + hardest_dist

        # Triplet loss
        loss = F.relu(hardest_positive_dist - negative_dist + self.margin)

        return loss.mean()

    def _all_triplet_loss(self, pairwise_dist, labels):
        """Use all valid triplets."""
        labels = labels.unsqueeze(0)
        mask_anchor_positive = (labels == labels.t()).float()
        mask_anchor_negative = (labels != labels.t()).float()

        mask_anchor_positive = mask_anchor_positive - torch.eye(
            labels.size(1), device=labels.device
        )

        # Get all valid triplets
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute triplet loss for all combinations
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Mask to get valid triplets
        mask = (
                mask_anchor_positive.unsqueeze(2) *
                mask_anchor_negative.unsqueeze(1)
        )

        triplet_loss = F.relu(triplet_loss) * mask

        # Average over valid triplets
        num_positive_triplets = mask.sum()
        if num_positive_triplets > 0:
            triplet_loss = triplet_loss.sum() / num_positive_triplets
        else:
            triplet_loss = torch.tensor(0.0, device=pairwise_dist.device)

        return triplet_loss


# Utility function to freeze/unfreeze layers
def set_parameter_requires_grad(model, feature_extracting):
    """
    Freeze or unfreeze model parameters.

    Args:
        model: Model to modify
        feature_extracting: If True, freeze parameters for feature extraction
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False