import os
import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, n_mfcc=40):
        # We keep 40 to match your original "rich representation" intent
        self.n_mfcc = n_mfcc

    def extract_features(self, audio_path):
        try:
            # Fixed sample rate ensures all vectors are calculated on the same scale
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # To match your original 1D format (Length will be exactly n_mfcc)
            # We use mean, but we MUST normalize it so the R-Tree works.
            vec = np.mean(mfccs, axis=1)
            
            # Simple Min-Max normalization to keep dimensions balanced
            # This prevents the 'loudness' dimension from breaking the search
            if np.max(np.abs(vec)) > 0:
                vec = vec / np.max(np.abs(vec))
                
            return vec
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

    def process_directory(self, directory_path):
        """Returns a dictionary mapping filename -> 40-dimensional vector"""
        features_dict = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.wav', '.mp3', '.flac')):
                filepath = os.path.join(directory_path, filename)
                vec = self.extract_features(filepath)
                if vec is not None:
                    # vec is now exactly (40,)
                    features_dict[filename] = vec
        return features_dict