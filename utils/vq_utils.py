import os
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans


def softmax_to_topk_soft_code(logits, k):
    """
    Sparse Coefficient
    """
    # Apply softmax to get probabilities
    y_soft = logits.softmax(dim=1)  # [batch_size, K]

    values, indices = torch.topk(y_soft, k, dim=1)
    mask = torch.zeros_like(y_soft, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    zero_tensor = torch.full_like(y_soft, 0)
    y_soft_topk = torch.where(mask, y_soft, zero_tensor)
    y_soft_topk = y_soft_topk / (y_soft_topk.sum(dim=1).unsqueeze(1) + 1e-10)
    soft_code_topk = y_soft_topk

    return soft_code_topk

def get_weights_and_indices(logits, k):
    # Apply softmax to get probabilities
    y_soft = logits.softmax(dim=1)  # [batch_size, K]
    values, indices = torch.topk(y_soft, k, dim=1)
    mask = torch.zeros_like(y_soft, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    zero_tensor = torch.full_like(y_soft, 0)
    y_soft_topk = torch.where(mask, y_soft, zero_tensor)
    y_soft_topk = y_soft_topk / (y_soft_topk.sum(dim=1).unsqueeze(1) + 1e-10)
    soft_code_topk = y_soft_topk
    non_zero_mask = soft_code_topk != 0
    weights = soft_code_topk[non_zero_mask].view(soft_code_topk.shape[0], k)
    # Keep the codebook index grid on the same device as the CUDA logits.
    # The released CPU-default arange cannot be indexed by a CUDA mask on
    # PyTorch 1.12.  Only placement changes; values/order are identical.
    indices = torch.arange(
        y_soft_topk.shape[1], device=y_soft_topk.device
    ).expand_as(soft_code_topk)[non_zero_mask].view(soft_code_topk.shape[0], k)

    return weights.float(), indices.float()


class ResidualVectorQuantizationWithClustering(nn.Module):
    def __init__(self, num_levels, num_clusters, feature_dim, device):
        super(ResidualVectorQuantizationWithClustering, self).__init__()
        self.num_levels = num_levels
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.device = device
        # Store the quantizers for each level
        self.quantizers = []

    def fit_quantizers(self, features):
        """
        Perform clustering on residuals to initialize quantizers for each level.
        """
        residuals = features.cpu().detach().numpy()  # Start with original features on CPU for clustering

        for level in range(self.num_levels):
            # Perform K-means clustering on the residuals
            print("Level", level)
            kmeans = MiniBatchKMeans(n_clusters=self.num_clusters)
            kmeans.fit(residuals)
            # Save the cluster centers as the quantizer for this level
            self.quantizers.append(torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32))
            # Compute quantized values and update residuals
            quantized = self._quantize_with_centers(residuals, kmeans.cluster_centers_).cpu().numpy()
            residuals = residuals - quantized

    def _quantize_with_centers(self, data, centers):
        """
        Given data and quantization centers, return the quantized data.
        """
        data_tensor = torch.tensor(data, device=self.device)
        centers_tensor = torch.tensor(centers, device=self.device)
        # Calculate distances and find nearest centers
        distances = torch.cdist(data_tensor, centers_tensor, p=2)
        indices = distances.argmin(dim=1)
        quantized_data = centers_tensor[indices]

        return quantized_data

    def forward(self, features):
        residuals = features
        quantized_outputs = []
        quantization_indices = []

        for level, centers in enumerate(self.quantizers):
            # Calculate distances to each cluster center and get the closest one
            print(level)
            print(torch.norm(centers, dim=1))
            print(torch.norm(residuals, dim=1).mean())
            distances = torch.cdist(residuals, centers, p=2)
            indices = distances.argmin(dim=1)
            # Retrieve quantized values based on closest centers
            quantized = centers[indices]
            # Store the quantized output and indices for each level
            quantized_outputs.append(quantized)
            quantization_indices.append(indices)
            # Update residuals for the next level
            residuals = residuals - quantized        
        quantized_result = sum(quantized_outputs)

        return quantized_result, quantization_indices

def load_2d_language_feature(data_dir, device, feature_level=None):
    """
    Load language features from the preprocessed 2D images.

    When ``feature_level`` is provided, only the feature vectors referenced by
    that segmentation level are returned.  This is important for a joint
    multi-scale model: each semantic head must be initialized from its own
    (small/medium/large) feature distribution instead of from the union of all
    scales.
    """
    # Keep codebook initialization independent of filesystem/inode order.
    # preprocess.py emits frame names in sorted order, so preserve that order
    # when concatenating the features consumed by MiniBatchKMeans.
    data_names = sorted(glob.glob(os.path.join(data_dir, '*_f.npy')))
    if not data_names:
        raise FileNotFoundError("No '*_f.npy' language features found in {}".format(data_dir))

    chunks = []
    for feature_path in data_names:
        features = np.load(feature_path)
        if feature_level is not None:
            segmentation_path = feature_path[:-6] + '_s.npy'
            segmentation = np.load(segmentation_path)
            if feature_level < 0 or feature_level >= segmentation.shape[0]:
                raise ValueError(
                    "feature_level {} is outside [0, {}) for {}".format(
                        feature_level, segmentation.shape[0], segmentation_path
                    )
                )
            referenced = np.unique(segmentation[feature_level])
            referenced = referenced[referenced >= 0].astype(np.int64, copy=False)
            if referenced.size == 0:
                continue
            features = features[referenced]
        chunks.append(features)

    if not chunks:
        raise ValueError(
            "No valid language features found for level {} in {}".format(
                feature_level, data_dir
            )
        )
    data = torch.from_numpy(np.concatenate(chunks, axis=0)).to(device)
    
    return data
