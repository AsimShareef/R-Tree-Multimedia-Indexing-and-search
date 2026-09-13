import pickle
import os

def save_tree_to_disk(tree, file_path="data/processed/rtree_index.pkl"):
    """
    Serializes the R-Tree object and saves it to a binary file on disk.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    print(f"Serializing R-Tree to {file_path}...")
    # Open the file in 'wb' (write binary) mode
    with open(file_path, 'wb') as file:
        pickle.dump(tree, file)
    print("Tree successfully saved to disk.")

def load_tree_from_disk(file_path="data/processed/rtree_index.pkl"):
    """
    Loads a serialized R-Tree from disk back into memory.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No index file found at {file_path}")
        
    print(f"Loading R-Tree from {file_path}...")
    # Open the file in 'rb' (read binary) mode
    with open(file_path, 'rb') as file:
        tree = pickle.load(file)
    print("Tree successfully loaded into memory.")
    return tree