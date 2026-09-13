import plotly.graph_objects as go
from sklearn.decomposition import PCA
from src.index.rtree import RTree
from src.index.storage import load_tree_from_disk

def extract_flat_data(node, flat_dict):
    """Recursively extracts all data points from the R-Tree."""
    if node.is_leaf():
        for point_id, vector in node.entries:
            flat_dict[point_id] = vector
    else:
        for child in node.children:
            extract_flat_data(child, flat_dict)

def extract_mbrs(node, depth=0, leaves=None, branches=None):
    """Recursively extracts the bounding box coordinates using your MBR structure."""
    if leaves is None: leaves = []
    if branches is None: branches = []
    
    # Safely grab the MBR object from the node
    bbox = getattr(node, 'mbr', None)
    
    if bbox is not None:
        try:
            # Based on geometry.py, your MBR uses min_bounds and max_bounds
            coords = [
                bbox.min_bounds[0], # Min X
                bbox.max_bounds[0], # Max X
                bbox.min_bounds[1], # Min Y
                bbox.max_bounds[1]  # Max Y
            ]
        except AttributeError:
            print("Warning: Could not find min_bounds/max_bounds on MBR object.")
            coords = None
            
        if coords is not None:
            if node.is_leaf():
                leaves.append(coords)
            else:
                branches.append((coords, depth))
            
    # Continue traversing down the tree
    if not node.is_leaf():
        for child in node.children:
            extract_mbrs(child, depth + 1, leaves, branches)
            
    return leaves, branches

def get_rtree_visualization(index_path, capacity=10, highlight_ids=None):
    """Generates an interactive Plotly HTML string of the R-Tree, highlighting specific points."""
    print(f"Generating visualization for {index_path}...")
    
    if highlight_ids is None:
        highlight_ids = []
    # Ensure all highlight IDs are strings for safe comparison
    highlight_ids = [str(hid) for hid in highlight_ids] 
    
    # 1. Load original tree and extract data
    tree = load_tree_from_disk(index_path)
    dataset_dict = {}
    extract_flat_data(tree.root, dataset_dict)
    
    if not dataset_dict:
        return "<h1>Error: Tree is empty.</h1>"

    vectors = list(dataset_dict.values())
    ids = list(dataset_dict.keys())
    
    # 2. Reduce from high-dim to 2D for human visualization
    pca_2d = PCA(n_components=2)
    vectors_2d = pca_2d.fit_transform(vectors)
    
    # 3. Build a temporary 2D R-Tree
    viz_tree = RTree(capacity=capacity)
    for i, point_id in enumerate(ids):
        viz_tree.insert(point_id, vectors_2d[i])
        
    # 4. Initialize Plotly Figure
    fig = go.Figure()
    
    # Separate data into regular points and highlighted points
    reg_x, reg_y, reg_text = [], [], []
    high_x, high_y, high_text = [], [], []
    
    for i, point_id in enumerate(ids):
        if str(point_id) in highlight_ids:
            high_x.append(vectors_2d[i, 0])
            high_y.append(vectors_2d[i, 1])
            high_text.append(f"<b>★ Top Match:</b> {point_id}")
        else:
            reg_x.append(vectors_2d[i, 0])
            reg_y.append(vectors_2d[i, 1])
            reg_text.append(str(point_id))
    
    # Plot Regular Data Points
    if reg_x:
        fig.add_trace(go.Scatter(
            x=reg_x, y=reg_y, 
            mode='markers', 
            name='Database Media',
            marker=dict(size=6, color='#0066cc', opacity=0.5),
            text=reg_text,
            hoverinfo="text"
        ))
        
    # Plot Highlighted Top-K Points
    if high_x:
        fig.add_trace(go.Scatter(
            x=high_x, y=high_y, 
            mode='markers', 
            name='Top-K Query Results',
            marker=dict(
                size=50, 
                color="#D47706", 
                symbol='star',
                line=dict(width=2, color='black')
            ),
            text=high_text,
            hoverinfo="text"
        ))
    
    # Extract and Draw MBRs
    leaves, branches = extract_mbrs(viz_tree.root)
    
    # Draw Leaf nodes (Light Green Boxes)
    for min_x, max_x, min_y, max_y in leaves:
        fig.add_shape(
            type="rect", x0=min_x, y0=min_y, x1=max_x, y1=max_y,
            line=dict(color="rgba(0, 200, 0, 0.5)", width=1), 
            fillcolor="rgba(0, 200, 0, 0.05)"
        )
        
    # Draw Branch nodes (Thick Red Dashed Boxes)
    for (min_x, max_x, min_y, max_y), depth in branches:
        fig.add_shape(
            type="rect", x0=min_x, y0=min_y, x1=max_x, y1=max_y,
            line=dict(color="rgba(255, 0, 0, 0.8)", width=2, dash="dash")
        )
        
    fig.update_layout(
        title="Interactive 2D R-Tree Spatial Partitioning",
        xaxis=dict(title="PCA Dimension 1"),
        yaxis=dict(title="PCA Dimension 2"),
        height=800,
        template="plotly_white",
        showlegend=True
    )
    
    # Return the raw HTML string
    return fig.to_html(full_html=True, include_plotlyjs='cdn')