import numpy as np
from sklearn.decomposition import PCA
import os

class DimensionalityReducer:
    def __init__(self, target_dimensions=10):
        self.target_dimensions = target_dimensions
        self.pca = PCA(n_components=self.target_dimensions)
        self.is_fitted = False

    def fit_transform(self, features_dict):
        """
        Fits the PCA model on the provided data and transforms it.
        :param features_dict: Dictionary mapping IDs to high-dimensional numpy arrays.
        :return: Dictionary mapping IDs to reduced-dimensional numpy arrays.
        """
        ids = list(features_dict.keys())
        vectors = np.array(list(features_dict.values()))
        
        # Ensure we don't ask for more components than we have samples
        n_samples = vectors.shape[0]
        if n_samples < self.target_dimensions:
            print(f"Warning: Only {n_samples} samples. Adjusting PCA components to {n_samples}.")
            self.pca = PCA(n_components=n_samples)
            
        reduced_vectors = self.pca.fit_transform(vectors)
        self.is_fitted = True
        
        # Re-map the reduced vectors back to their original IDs
        reduced_dict = {ids[i]: reduced_vectors[i] for i in range(len(ids))}
        return reduced_dict

def save_to_flat_file(reduced_dict, output_path):
    """
    Saves the processed dictionary to a flat .npy file for easy loading later.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, reduced_dict)
    print(f"Saved {len(reduced_dict)} feature vectors to {output_path}")

def load_from_flat_file(file_path):
    """
    Loads the dictionary from the flat .npy file.
    """
    # allow_pickle=True is required to load dictionaries saved via numpy
    return np.load(file_path, allow_pickle=True).item()