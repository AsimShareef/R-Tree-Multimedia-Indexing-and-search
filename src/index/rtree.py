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

    def delete(self, point_id, point_vector):
        """
        Removes a point from the R-Tree, following Guttman's Delete algorithm:
        FindLeaf -> remove entry -> CondenseTree (re-inserting orphaned entries
        from any node that underflows below the minimum fill) -> shorten the
        tree if the root is left with a single child.

        :param point_id: id of the point to remove.
        :param point_vector: the vector that was used at insert time (needed to
            prune the search, since an R-Tree isn't keyed by id alone).
        :return: True if the point was found and removed, False otherwise.
        """
        point_vector = np.array(point_vector, dtype=np.float32)

        leaf = self._find_leaf(self.root, point_id, point_vector)
        if leaf is None:
            return False

        leaf.remove_point(point_id)
        self._condense_tree(leaf)

        # If the root is an internal node with only one child, that child
        # becomes the new root (keeps the tree from growing needlessly tall).
        if not self.root.is_leaf() and len(self.root.children) == 1:
            self.root = self.root.children[0]
            self.root.parent = None
        elif self.root.is_leaf():
            self.root.update_mbr()

        return True

    def _find_leaf(self, node, point_id, point_vector):
        """
        Searches for the leaf holding point_id. Only descends into children
        whose MBR could plausibly contain the point, but still backtracks:
        overlapping MBRs mean more than one subtree can geometrically contain
        the point even though it's only stored once.
        """
        if node.is_leaf():
            for pid, _ in node.entries:
                if pid == point_id:
                    return node
            return None

        for child in node.children:
            if child.mbr is not None and child.mbr.contains_point(point_vector):
                found = self._find_leaf(child, point_id, point_vector)
                if found is not None:
                    return found
        return None

    def _condense_tree(self, leaf):
        """
        Walks from `leaf` up to the root. Any node that has fallen below the
        minimum fill (M/2) is detached from its parent and all of its entries
        are queued for re-insertion, rather than left underfull in place.
        """
        min_fill = max(1, self.capacity // 2)
        orphans = []
        node = leaf

        while node is not self.root:
            parent = node.parent
            underflow = (
                len(node.entries) < min_fill if node.is_leaf()
                else len(node.children) < min_fill
            )

            if underflow:
                parent.remove_child(node)
                orphans.extend(self._collect_points(node))
            else:
                node.update_mbr()

            parent.update_mbr()
            node = parent

        for point_id, point_vector in orphans:
            self.insert(point_id, point_vector)

    def _collect_points(self, node):
        """Recursively flattens a (detached) subtree into its raw (id, vector) entries."""
        if node.is_leaf():
            return list(node.entries)

        points = []
        for child in node.children:
            points.extend(self._collect_points(child))
        return points

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