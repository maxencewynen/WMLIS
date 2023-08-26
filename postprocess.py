import numpy as np
from scipy.ndimage import maximum_filter, generate_binary_structure, label, labeled_comprehension


def remove_connected_components(segmentation, l_min=9):
    """
    Remove all lesions with less or equal amount of voxels than `l_min` from a
    binary segmentation mask `segmentation`.
    Args:
      segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
      l_min:  `int`, minimal amount of voxels in a lesion.
    Returns:
      Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
      only with connected components that have more than `l_min` voxels.
    """
    labeled_seg, num_labels = label(segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = labeled_comprehension(segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        if n_el > l_min:
            current_voxels = np.stack(np.where(labeled_seg == i_el), axis=1)
            seg2[current_voxels[:, 0],
                 current_voxels[:, 1],
                 current_voxels[:, 2]] = 1
    return seg2

def simple_instance_representation(heatmap, pool_size=3, threshold=0.1, k=200):
    """
    Apply NMS on the heatmap prediction to find instance centers.

    Args:
        heatmap: 3D array containing the instance center heatmap prediction.
        pool_size: Integer, the size of the max pooling window.
        threshold: Float, hard threshold to filter out low confidence predictions.
        k: Integer, only keep top-k highest confidence scores.

    Returns:
        instance_centers: List of tuples containing the (z, y, x) coordinates of instance centers.
    """
    # Apply max pooling
    pooled = maximum_filter(heatmap, size=(pool_size, pool_size, pool_size))
    
    # Mask of the local maxima
    maxima = (heatmap == pooled)
    
    # Apply threshold and get coordinates of the remaining maxima
    coordinates = np.argwhere(maxima & (heatmap >= threshold))
    
    # Sort coordinates based on heatmap values and keep only top-k
    coordinates = sorted(coordinates, key=lambda coord: heatmap[tuple(coord)], reverse=True)[:k]

    return maxima, coordinates

def simple_instance_grouping(heatmap, offsets, instance_centers, semantic_mask, min_lesion_size=9):
    """
    Assign instance IDs based on offset vectors and instance centers.

    Args:
        heatmap: 3D array containing the instance center heatmap prediction.
        offsets: 4D array (3 x depth x height x width) containing the regression offsets.
        instance_centers: List of tuples containing the (z, y, x) coordinates of instance centers.
        semantic_mask: 3D array where 'background' voxels are marked with 0 and 'lesion' voxels are marked with 1.
        min_lesion_size: Integer. All lesions with less or equal amount of voxels than min_lesion_size will 
                be replaced by a voting of their closest neighbours or to background (0) if no other lesion is around.

    Returns:
        instance_map: 3D array containing the instance IDs for each pixel.
    """
    if offsets.shape[-1] == 3:
        offsets = offsets.transpose(3,0,1,2)
    
    instance_map = np.zeros(heatmap.shape)

    # Get the 3D grids for coordinates
    z_grid, y_grid, x_grid = np.meshgrid(np.arange(heatmap.shape[0]).astype(np.float64),
                                         np.arange(heatmap.shape[1]).astype(np.float64),
                                         np.arange(heatmap.shape[2]).astype(np.float64),
                                         indexing='ij')
    
    # Update the grids based on the offsets
    z_grid += offsets[0]
    y_grid += offsets[1]
    x_grid += offsets[2]

    # For each voxel in 'lesion' class, assign the instance ID based on the closest instance center
    lesions_voxels = np.argwhere(semantic_mask == 1)
    for z, y, x in lesions_voxels:
        min_distance = float('inf')
        best_instance_id = 0
        for idx, (cz, cy, cx) in enumerate(instance_centers):
            distance = (z_grid[z, y, x] - cz)**2 + (y_grid[z, y, x] - cy)**2 + (x_grid[z, y, x] - cx)**2
            if distance < min_distance:
                min_distance = distance
                best_instance_id = idx + 1  # +1 because 0 is reserved for 'stuff'
        instance_map[z, y, x] = best_instance_id
    
    instance_map = instance_map.astype(np.int32)
    # before_postprocessing = np.copy(instance_map)
    # changed_values = np.zeros_like(instance_map)
    
    lesion_ids = np.unique(instance_map)
    highest_id = max(lesion_ids)
    
    for lid in lesion_ids:
        if lid == 0:  # Skip background
            continue
        
        current_lesion = (instance_map == lid)
        structure = generate_binary_structure(3, 3)  # 3D 6-connectivity structure
        labeled_array, num_features = label(current_lesion, structure)
        
        # Get sizes of connected components and sort by size
        component_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
        sorted_components = np.argsort(component_sizes)
        
        largest_component_id = sorted_components[-1]
        for idx in sorted_components:
            component_voxels = np.argwhere(labeled_array == idx + 1)
            if component_sizes[idx] > min_lesion_size and idx != largest_component_id:
                highest_id += 1
                new_id = highest_id 
                instance_map[component_voxels[:, 0], component_voxels[:, 1], component_voxels[:, 2]] = new_id
                # changed_values[component_voxels[:, 0], component_voxels[:, 1], component_voxels[:, 2]] = 1
            elif component_sizes[idx] > min_lesion_size and idx != largest_component_id:
                pass
            elif component_sizes[idx] <= min_lesion_size and idx != largest_component_id:
                for z, y, x in component_voxels:
                    neighbors = instance_map[max(z-1,0):z+2, max(y-1,0):y+2, max(x-1,0):x+2].flatten()
                    neighbors = neighbors[(neighbors != 0) & (neighbors != lid)]
                    if neighbors.size == 0:
                        instance_map[z, y, x] = 0
                        semantic_mask[z, y, x] = 0
                        # changed_values[z, y, x] = 1
                    else:
                        instance_map[z, y, x] = np.bincount(neighbors).argmax()
                        # changed_values[z, y, x] = 1

    # return instance_map, before_postprocessing, changed_values
    return instance_map, semantic_mask

def postprocess(semantic_mask, heatmap, offsets):
    semantic_mask = remove_connected_components(semantic_mask)
    instance_centers, coordinates = simple_instance_representation(heatmap)
    instance_mask, semantic_mask = simple_instance_grouping(heatmap, offsets, coordinates, semantic_mask)
    
    return semantic_mask, instance_mask
