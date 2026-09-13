import os
import pickle
from src.preprocessing.image_features import ImageFeatureExtractor
from src.preprocessing.audio_features import AudioFeatureExtractor
from src.preprocessing.dimensionality import DimensionalityReducer
from src.index.rtree import RTree
from src.index.storage import save_tree_to_disk

def build_real_database():
    print("🚀 Starting Database Ingestion Engine...")
    os.makedirs("data/processed", exist_ok=True)

    # ==========================================
    # PHASE 1: PROCESS IMAGES
    # ==========================================
    print("\n[1/4] Extracting Image Features (This may take a while)...")
    img_extractor = ImageFeatureExtractor()
    # This reads the folder and returns a dict: {"dog.jpg": [512-D vector], ...}
    raw_image_data = img_extractor.process_directory("data/raw_images") 
    
    if raw_image_data:
        print(f"Extracted {len(raw_image_data)} images. Reducing dimensions to 10D...")
        img_pca = DimensionalityReducer(target_dimensions=10)
        reduced_img_data = img_pca.fit_transform(raw_image_data)
        
        # Save the Image PCA model so the web UI can use it later
        with open("data/processed/image_pca_model.pkl", "wb") as f:
            pickle.dump(img_pca.pca, f)

        print("Building Image R-Tree...")
        image_tree = RTree(capacity=50)
        for point_id, vector in reduced_img_data.items():
            image_tree.insert(point_id, vector)
            
        save_tree_to_disk(image_tree, "data/processed/image_index.pkl")
    else:
        print("⚠️ No images found in data/raw_images/")

    # ==========================================
    # PHASE 2: PROCESS AUDIO
    # ==========================================
    print("\n[2/4] Extracting Audio Features (This may take a while)...")
    audio_extractor = AudioFeatureExtractor()
    raw_audio_data = audio_extractor.process_directory("data/raw_audio")
    
    if raw_audio_data:
        print(f"Extracted {len(raw_audio_data)} audio files. Reducing dimensions to 10D...")
        audio_pca = DimensionalityReducer(target_dimensions=10)
        reduced_audio_data = audio_pca.fit_transform(raw_audio_data)
        
        # Save the Audio PCA model
        with open("data/processed/audio_pca_model.pkl", "wb") as f:
            pickle.dump(audio_pca.pca, f)

        print("Building Audio R-Tree...")
        audio_tree = RTree(capacity=50)
        for point_id, vector in reduced_audio_data.items():
            audio_tree.insert(point_id, vector)
            
        save_tree_to_disk(audio_tree, "data/processed/audio_index.pkl")
    else:
        print("⚠️ No audio files found in data/raw_audio/")

    print("\n✅ Database construction complete! You can now start the web server.")

if __name__ == "__main__":
    build_real_database()