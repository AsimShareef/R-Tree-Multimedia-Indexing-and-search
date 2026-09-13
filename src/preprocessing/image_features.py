import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class ImageFeatureExtractor:
    def __init__(self):
        # Load a pre-trained ResNet18 model
        weights = models.ResNet18_Weights.DEFAULT
        base_model = models.resnet18(weights=weights)
        
        # Remove the final classification layer to get the raw 512-D embedding
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval() # Set to evaluation mode
        
        # Standard preprocessing for PyTorch pre-trained models
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_path):
        """Extracts a 512-D feature vector from a single image."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0) # Add batch dimension
            
            with torch.no_grad():
                features = self.model(img_tensor)
                
            # Flatten the output from [1, 512, 1, 1] to [512]
            return features.squeeze().numpy()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def process_directory(self, directory_path):
        """Processes all images in a directory and returns a dictionary of vectors."""
        features_dict = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(directory_path, filename)
                vec = self.extract_features(filepath)
                if vec is not None:
                    features_dict[filename] = vec
        return features_dict