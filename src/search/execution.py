import pickle
from src.preprocessing.image_features import ImageFeatureExtractor
from src.preprocessing.audio_features import AudioFeatureExtractor
from src.search.knn_search import knn_search
from src.index.storage import load_tree_from_disk

def execute_user_query(query_filepath, media_type, k=5):
    """
    Takes a raw file from the user, extracts features, reduces dimensions,
    and searches the pre-built R-Tree database.
    """
    print(f"\n[Search] Processing {media_type} query for {query_filepath}...")
    
    # 1. Initialize variables based on media type
    if media_type == "image":
        extractor = ImageFeatureExtractor()
        pca_path = "data/processed/image_pca_model.pkl"
        tree_path = "data/processed/image_index.pkl"
    elif media_type == "audio":
        extractor = AudioFeatureExtractor()
        pca_path = "data/processed/audio_pca_model.pkl"
        tree_path = "data/processed/audio_index.pkl"
    else:
        raise ValueError("Invalid media type.")

    # 2. Extract raw neural features (e.g., 512-D for image, 40-D for audio)
    raw_vector = extractor.extract_features(query_filepath)
    if raw_vector is None:
        raise ValueError("Could not extract features from the provided file.")

    # 3. Load the EXACT same PCA model used during ingestion to reduce to 10-D
    with open(pca_path, "rb") as f:
        pca_model = pickle.load(f)
    
    # PCA expects a 2D array, so we wrap raw_vector in a list [], then grab the first result [0]
    reduced_query_vector = pca_model.transform([raw_vector])[0]

    # 4. Load the R-Tree and perform the Priority Queue search
    tree = load_tree_from_disk(tree_path)
    raw_results = knn_search(tree, reduced_query_vector, k)

    # 5. Format results into a list of dictionaries for the Flask JSON response
    # knn_search returns tuples: [(distance, point_id, node_type), ...]
    formatted_results = []
    for dist, point_id, _ in raw_results:
        formatted_results.append({
            "distance": float(dist), # Convert numpy float to standard python float for JSON
            "point_id": str(point_id)
        })

    print(f"Found top {k} matches successfully!")
    return formatted_results