import numpy as np
from .geometry import MBR

def get_entry_mbr(entry, is_leaf):
    """
    Helper function to extract an MBR from an entry.
    - If it's a leaf entry, it's a tuple (id, vector), so the MBR is just the point.
    - If it's an internal entry, it's a Node object which has an .mbr attribute.
    """
    if is_leaf:
        point = entry[1]
        return MBR(point, point)
    else:
        return entry.mbr

def pick_seeds(entries, is_leaf):
    """
    Selects two entries to be the first elements of two new groups.
    Chooses the pair that would result in the most 'dead space' if grouped together.
    """
    max_wasted_area = -1.0
    seed1, seed2 = 0, 1

    n = len(entries)
    for i in range(n):
        mbr_i = get_entry_mbr(entries[i], is_leaf)
        for j in range(i + 1, n):
            mbr_j = get_entry_mbr(entries[j], is_leaf)
            
            # Calculate the area of the MBR containing both entries
            enlarged_mbr = mbr_i.enlarge(mbr_j)
            enlarged_area = enlarged_mbr.volume()
            
            # Wasted area = area(combined) - area(i) - area(j)
            wasted_area = enlarged_area - mbr_i.volume() - mbr_j.volume()
            
            if wasted_area > max_wasted_area:
                max_wasted_area = wasted_area
                seed1, seed2 = i, j

    return seed1, seed2

def pick_next(entries, mbr1, mbr2, is_leaf):
    """
    Selects the next entry to assign to a group.
    To minimize overlap, we find the entry that has the largest difference in 
    enlargement area between the two MBRs, and assign it to the one it enlarges least.
    """
    max_diff = -1.0
    best_entry_idx = 0
    best_group = 1 # 1 for group1, 2 for group2

    for i, entry in enumerate(entries):
        entry_mbr = get_entry_mbr(entry, is_leaf)
        
        d1 = mbr1.enlargement_area(entry_mbr if not is_leaf else entry[1])
        d2 = mbr2.enlargement_area(entry_mbr if not is_leaf else entry[1])
        
        diff = abs(d1 - d2)
        
        if diff > max_diff:
            max_diff = diff
            best_entry_idx = i
            # Assign to the group that requires the least enlargement.
            # If there's a tie, assign to the one with the smaller total area.
            if d1 < d2:
                best_group = 1
            elif d2 < d1:
                best_group = 2
            else:
                best_group = 1 if mbr1.volume() < mbr2.volume() else 2

    return best_entry_idx, best_group

def quadratic_split(node):
    """
    Splits an overflowing node into two sets of entries using the quadratic cost algorithm.
    Returns two lists of entries (group1, group2).
    """
    is_leaf = node.is_leaf()
    entries = node.entries if is_leaf else node.children
    
    # 1. Pick Seeds
    seed1_idx, seed2_idx = pick_seeds(entries, is_leaf)
    
    # Extract seeds and initialize groups
    # Pop the higher index first to avoid shifting the lower index
    if seed1_idx > seed2_idx:
        seed1_idx, seed2_idx = seed2_idx, seed1_idx
        
    entry2 = entries.pop(seed2_idx)
    entry1 = entries.pop(seed1_idx)
    
    group1 = [entry1]
    group2 = [entry2]
    
    mbr1 = get_entry_mbr(entry1, is_leaf)
    mbr2 = get_entry_mbr(entry2, is_leaf)

    # Minimum fill requirement (usually m = M/2, where M is capacity)
    min_fill = max(1, node.capacity // 2)

    # 2. Pick Next for remaining entries
    while entries:
        # Check if remaining entries MUST be assigned to a group to meet min_fill
        if len(group1) + len(entries) == min_fill:
            group1.extend(entries)
            break
        if len(group2) + len(entries) == min_fill:
            group2.extend(entries)
            break
            
        next_idx, group_choice = pick_next(entries, mbr1, mbr2, is_leaf)
        chosen_entry = entries.pop(next_idx)
        
        entry_mbr = get_entry_mbr(chosen_entry, is_leaf)
        
        if group_choice == 1:
            group1.append(chosen_entry)
            mbr1 = mbr1.enlarge(entry_mbr if not is_leaf else chosen_entry[1])
        else:
            group2.append(chosen_entry)
            mbr2 = mbr2.enlarge(entry_mbr if not is_leaf else chosen_entry[1])

    return group1, group2