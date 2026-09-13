import numpy as np
from .geometry import MBR

class Node:
    """
    Base class for R-Tree nodes.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.mbr = None
        self.parent = None

    def is_leaf(self):
        raise NotImplementedError("Subclasses must implement is_leaf()")

    def is_overflow(self):
        """Check if the node exceeds its maximum capacity."""
        raise NotImplementedError("Subclasses must implement is_overflow()")

    def update_mbr(self):
        """Recalculate the MBR based on current contents."""
        raise NotImplementedError("Subclasses must implement update_mbr()")


class LeafNode(Node):
    """
    Leaf nodes contain the actual high-dimensional data points (e.g., image/audio vectors).
    """
    def __init__(self, capacity):
        super().__init__(capacity)
        # Store tuples of (point_id, vector_data)
        self.entries = []

    def is_leaf(self):
        return True

    def is_overflow(self):
        return len(self.entries) > self.capacity

    def add_point(self, point_id, point_vector):
        """
        Add a data point to the leaf and update the MBR.
        """
        self.entries.append((point_id, point_vector))
        
        if self.mbr is None:
            # Initialize MBR with the exact bounds of the single point
            self.mbr = MBR(point_vector, point_vector)
        else:
            # Expand the existing MBR to include the new point
            self.mbr = self.mbr.enlarge(point_vector)

    def remove_point(self, point_id):
        """
        Removes the entry with the given point_id, if present.
        Does not update the MBR; the caller should call update_mbr() afterwards.
        :return: True if an entry was removed, False if point_id wasn't found here.
        """
        for i, (pid, _) in enumerate(self.entries):
            if pid == point_id:
                del self.entries[i]
                return True
        return False

    def update_mbr(self):
        """
        Recomputes the MBR from scratch based on all stored points.
        Useful after a node split or deletion.
        """
        if not self.entries:
            self.mbr = None
            return
        
        # Extract just the vectors from the entries
        vectors = np.array([entry[1] for entry in self.entries])
        min_bounds = np.min(vectors, axis=0)
        max_bounds = np.max(vectors, axis=0)
        self.mbr = MBR(min_bounds, max_bounds)


class InternalNode(Node):
    """
    Internal nodes contain child nodes (which can be other InternalNodes or LeafNodes)
    and an MBR that tightly bounds all children.
    """
    def __init__(self, capacity):
        super().__init__(capacity)
        self.children = []

    def is_leaf(self):
        return False

    def is_overflow(self):
        return len(self.children) > self.capacity

    def add_child(self, child_node):
        """
        Adds a child node and updates the parent-child relationship and MBR.
        """
        self.children.append(child_node)
        child_node.parent = self
        
        if self.mbr is None:
            self.mbr = child_node.mbr
        elif child_node.mbr is not None:
            self.mbr = self.mbr.enlarge(child_node.mbr)

    def remove_child(self, child_node):
        """
        Detaches a child node (used by deletion when a node underflows).
        Does not update the MBR; the caller is expected to call update_mbr()
        once all removals for this node are finished.
        """
        self.children.remove(child_node)
        child_node.parent = None

    def update_mbr(self):
        """
        Recomputes the MBR from scratch based on all child MBRs.
        """
        if not self.children:
            self.mbr = None
            return

        # Start with the MBR of the first child
        current_mbr = self.children[0].mbr

        # Iteratively enlarge it with the remaining children
        for child in self.children[1:]:
            if child.mbr is not None:
                current_mbr = current_mbr.enlarge(child.mbr)

        self.mbr = current_mbr