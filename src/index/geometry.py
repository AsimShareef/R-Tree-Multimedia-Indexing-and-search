import numpy as np

class MBR:
    """
    Minimum Bounding Rectangle (MBR) for d-dimensional space.
    Used to partition spatial data in the R-Tree.
    """
    def __init__(self, min_bounds, max_bounds):
        """
        :param min_bounds: 1D numpy array representing the lower bounds in all dimensions.
        :param max_bounds: 1D numpy array representing the upper bounds in all dimensions.
        """
        self.min_bounds = np.array(min_bounds, dtype=np.float32)
        self.max_bounds = np.array(max_bounds, dtype=np.float32)

    def volume(self):
        """
        Calculates the d-dimensional volume of the MBR. 
        In standard 2D R-Tree literature, this is referred to as 'area'.
        """
        # np.prod multiplies all elements together (e.g., width * height * depth...)
        return np.prod(self.max_bounds - self.min_bounds)

    def enlarge(self, other):
        """
        Returns a newly sized MBR that encompasses both this MBR and another MBR (or point).
        """
        if isinstance(other, MBR):
            new_min = np.minimum(self.min_bounds, other.min_bounds)
            new_max = np.maximum(self.max_bounds, other.max_bounds)
        else:
            # Assumes 'other' is a point (1D numpy array)
            new_min = np.minimum(self.min_bounds, other)
            new_max = np.maximum(self.max_bounds, other)
        return MBR(new_min, new_max)

    def enlargement_area(self, other):
        """
        Calculates the increase in n-dimensional volume if 'other' is added to this MBR.
        This is a crucial heuristic for deciding which node to insert a new point into.
        """
        enlarged_mbr = self.enlarge(other)
        return enlarged_mbr.volume() - self.volume()

    def intersects(self, other_mbr):
        """
        Checks if this MBR intersects with another MBR.
        Used heavily during branch-pruning in searches.
        """
        return np.all(self.min_bounds <= other_mbr.max_bounds) and \
               np.all(self.max_bounds >= other_mbr.min_bounds)

    def min_distance(self, point):
        """
        Calculates the minimum Euclidean distance from a given point to this MBR.
        Essential for the Priority Queue-based KNN search.
        """
        # Find the closest point inside or on the boundary of the MBR to the target point
        closest_point = np.maximum(self.min_bounds, np.minimum(point, self.max_bounds))
        return np.linalg.norm(point - closest_point)
        
    def __repr__(self):
        return f"MBR(min={self.min_bounds}, max={self.max_bounds})"