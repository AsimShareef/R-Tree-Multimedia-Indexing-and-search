import numpy as np
from .node import LeafNode, InternalNode
from .split import quadratic_split

class RTree:
    """
    Main R-Tree class for high-dimensional spatial indexing.
    """
    def __init__(self, capacity=50):
        """
        :param capacity: Maximum number of entries a node can hold before splitting.
        """
        self.capacity = capacity
        # The tree starts with a single empty leaf node as the root
        self.root = LeafNode(self.capacity)

    def insert(self, point_id, point_vector):
        """
        Inserts a new high-dimensional data point into the R-Tree.
        """
        # Ensure the vector is a numpy array
        point_vector = np.array(point_vector, dtype=np.float32)

        # 1. Choose Leaf: Find the best leaf node to insert the new point
        leaf = self._choose_leaf(self.root, point_vector)

        # 2. Insert the point into the chosen leaf
        leaf.add_point(point_id, point_vector)

        # 3. Handle Overflows and Adjust Tree upwards
        node_to_split = leaf
        new_node = None

        while node_to_split is not None:
            if node_to_split.is_overflow():
                # Split the node using the quadratic split algorithm
                group1, group2 = quadratic_split(node_to_split)
                
                # node_to_split keeps group1, new_node gets group2
                if node_to_split.is_leaf():
                    node_to_split.entries = group1
                    node_to_split.update_mbr()
                    
                    new_node = LeafNode(self.capacity)
                    new_node.entries = group2
                    new_node.update_mbr()
                else:
                    node_to_split.children = group1
                    for child in group1:
                        child.parent = node_to_split
                    node_to_split.update_mbr()
                    
                    new_node = InternalNode(self.capacity)
                    new_node.children = group2
                    for child in group2:
                        child.parent = new_node
                    new_node.update_mbr()
            else:
                # If no split, just update the MBR based on the new point
                node_to_split.update_mbr()
                new_node = None

            # Move up to the parent
            parent = node_to_split.parent

            if parent is None:
                # We reached the root. If it split, we need a new root.
                if new_node is not None:
                    new_root = InternalNode(self.capacity)
                    new_root.add_child(node_to_split)
                    new_root.add_child(new_node)
                    self.root = new_root
                break # Tree adjustment complete
            
            else:
                # If there was a split, add the new node to the parent
                if new_node is not None:
                    parent.add_child(new_node)
                
                node_to_split = parent

    def _choose_leaf(self, node, point_vector):
        """
        Traverses the tree from the root to find the best leaf for a new point.
        The "best" path is the one that requires the least MBR enlargement.
        """
        if node.is_leaf():
            return node

        best_child = None
        min_enlargement = float('inf')
        min_volume = float('inf')

        for child in node.children:
            if child.mbr is None:
                continue
                
            enlargement = child.mbr.enlargement_area(point_vector)
            volume = child.mbr.volume()

            # Choose the child that requires the least MBR enlargement.
            # Resolve ties by choosing the child with the smaller existing volume.
            if enlargement < min_enlargement:
                min_enlargement = enlargement
                min_volume = volume
                best_child = child
            elif enlargement == min_enlargement and volume < min_volume:
                min_volume = volume
                best_child = child

        # Recursively search down the best path
        return self._choose_leaf(best_child, point_vector)