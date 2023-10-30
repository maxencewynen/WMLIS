import numpy as np
from scipy.ndimage import maximum_filter, generate_binary_structure, label, labeled_comprehension
import argparse
import torch
import torch.nn.functional as F


def postprocess_binary_segmentation(binary_segmentation, threshold=0.5, min_lesion_size=9):
    # Threshold the image
    binary_data = np.where(binary_segmentation >= threshold, 1, 0).astype(np.uint8)

    # Remove objects smaller than 9 voxels
    binary_data = remove_connected_components(binary_data, l_min=min_lesion_size)

    # Find connected components larger than 9 voxels
    labeled_array, num_features = label(binary_data)

    return labeled_array


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

# def simple_instance_representation(heatmap, pool_size=3, threshold=0.1, k=200):
#     """
#     Apply NMS on the heatmap prediction to find instance centers.
#
#     Args:
#         heatmap: 3D array containing the instance center heatmap prediction.
#         pool_size: Integer, the size of the max pooling window.
#         threshold: Float, hard threshold to filter out low confidence predictions.
#         k: Integer, only keep top-k highest confidence scores.
#
#     Returns:
#         instance_centers: List of tuples containing the (z, y, x) coordinates of instance centers.
#     """
#     # Apply max pooling
#     pooled = maximum_filter(heatmap, size=(pool_size, pool_size, pool_size))
#
#     # Mask of the local maxima
#     maxima = (heatmap == pooled)
#
#     # Apply threshold and get coordinates of the remaining maxima
#     coordinates = np.argwhere(maxima & (heatmap >= threshold))
#
#     # Sort coordinates based on heatmap values and keep only top-k
#     coordinates = sorted(coordinates, key=lambda coord: heatmap[tuple(coord)], reverse=True)[:k]
#
#     return maxima, coordinates


def compute_voting_image(offsets, binary_lesion_mask):
    """
    Compute a voting image based on the offsets matrix for voxels inside the binary lesion mask.

    Args:
        offsets: 4D array (3 x depth x height x width) containing the regression offsets.
        binary_lesion_mask: 3D binary array where 'background' voxels are marked with 0 and 'lesion' voxels are marked with 1.

    Returns:
        voting_image: 3D array containing the log of the number of votes received by each voxel inside the lesion mask.
    """
    depth, height, width = offsets.shape[1:]

    # Create an empty voting matrix with the same shape as the offsets matrix
    voting_matrix = np.zeros((depth, height, width))

    # Iterate through all voxels in the offsets matrix
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # Check if the voxel is inside the lesion mask
                if binary_lesion_mask[z, y, x] == 1:
                    # Calculate the coordinates pointed to by the offsets
                    dz, dy, dx = offsets[:, z, y, x]
                    new_z, new_y, new_x = int(z + dz), int(y + dy), int(x + dx)

                    # Check if the new coordinates are within bounds
                    if 0 <= new_z < depth and 0 <= new_y < height and 0 <= new_x < width:
                        # Update the voting matrix at the new coordinates
                        voting_matrix[new_z, new_y, new_x] += 1

    # Compute the log of the number of votes for each voxel inside the lesion mask
    voting_image = np.log(voting_matrix + 1)  # Adding 1 to avoid log(0)

    return voting_image


def simple_instance_grouping(heatmap, offsets, instance_centers, semantic_mask, min_lesion_size=9, compute_voting=False):
    """
    Assign instance IDs based on offset vectors and instance centers.

    Args:
        heatmap: 3D array containing the instance center heatmap prediction.
        offsets: 4D array (3 x depth x height x width) containing the regression offsets.
        instance_centers: List of tuples containing the (z, y, x) coordinates of instance centers.
        semantic_mask: 3D array where 'background' voxels are marked with 0 and 'lesion' voxels are marked with 1.
        min_lesion_size: Integer. All lesions with less or equal amount of voxels than min_lesion_size will 
                be replaced by a voting of their closest neighbours or to background (0) if no other lesion is around.
        compute_voting: Boolean. If True, the voting image will be computed and returned.

    Returns:
        semantic_mask: 3D array where 'background' voxels are marked with 0 and 'lesion' voxels are marked with 1.
        instance_map: 3D array containing the instance IDs for each pixel.
        (if compute_voting) voting_image: 3D array containing the log of the number of votes received by each voxel.
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
    if compute_voting:
        voting_image = compute_voting_image(offsets, semantic_mask)
        return instance_map, semantic_mask, voting_image
    return instance_map, semantic_mask


# def postprocess(semantic_mask, heatmap, offsets, compute_voting=False):
#     semantic_mask = remove_connected_components(semantic_mask)
#     instance_centers, coordinates = simple_instance_representation(heatmap)
#     if compute_voting:
#         instance_mask, semantic_mask, voting_image = simple_instance_grouping(heatmap, offsets, coordinates, semantic_mask, compute_voting=compute_voting)
#         return semantic_mask, instance_mask, instance_centers, voting_image
#
#     instance_mask, semantic_mask = simple_instance_grouping(heatmap, offsets, coordinates, semantic_mask)
#     return semantic_mask, instance_mask, instance_centers

def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=None):
    """
    from https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W, D] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 3] where K is the number of center points. The order of second dim is (z, y, x).
    """
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool3d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 3, 'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1])


def group_pixels(ctr, offsets, compute_voting=False):
    """
    from https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 3] where K is the number of center points. The order of second dim is (z, y, x).
        offsets: A Tensor of shape [N, 3, H, W, D] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_z, offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W, D] (to be gathered by distributed data parallel).
    """
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0)
    depth, height, width = offsets.size()[1:]

    # generates a 3D coordinate map, where each location is the coordinate of that loc
    z_coord = torch.arange(depth, dtype=offsets.dtype, device=offsets.device).unsqueeze(1).unsqueeze(2).repeat(1, height, width).unsqueeze(0)
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).unsqueeze(0).unsqueeze(2).repeat(depth, 1, width).unsqueeze(0)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).unsqueeze(0).unsqueeze(1).repeat(depth, height, 1).unsqueeze(0)

    coord = torch.cat((z_coord, y_coord, x_coord), dim=0)

    ctr_loc = coord + offsets
    if compute_voting:
        votes = torch.round(ctr_loc).long()
        votes[2, :] = torch.clamp(votes[2, :], 0, width - 1)
        votes[1, :] = torch.clamp(votes[1, :], 0, height - 1)
        votes[0, :] = torch.clamp(votes[0, :], 0, depth - 1)

        flat_votes = votes.view(3, -1)
        # Calculate unique coordinate values and their counts
        unique_coords, counts = torch.unique(flat_votes, dim=1, return_counts=True)
        # Create a result tensor with zeros
        votes = torch.zeros(1, votes.shape[1], votes.shape[2], votes.shape[3], dtype=torch.long, device=votes.device)
        # Use advanced indexing to set counts in the result tensor
        votes[0, unique_coords[0], unique_coords[1], unique_coords[2]] = counts
        votes = torch.log(votes + 1, out=torch.zeros_like(votes, dtype=torch.float32))
        votes = F.avg_pool3d(votes, kernel_size=3, stride=1, padding=1)
        votes*=100

    ctr_loc = ctr_loc.view(3, depth * height * width).transpose(1, 0)

    # ctr: [K, 3] -> [K, 1, 3]
    # ctr_loc = [D*H*W, 3] -> [1, D*H*W, 3]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    # Compute the Euclidean distance in 3D space
    distance = torch.norm(ctr - ctr_loc, dim=-1)

    # Find the center with the minimum distance at each voxel, offset by 1, to reserve id=0 for stuff
    instance_id = torch.argmin(distance, dim=0).view(1, depth, height, width) + 1
    if compute_voting:
        return instance_id, votes
    return instance_id


def postprocess(semantic_mask, heatmap, offsets, threshold=0.5, compute_voting=False):
    """
    Postprocesses the semantic mask, center heatmap and the offsets.
    Arguments:
        semantic_mask: a binary numpy.ndarray of shape [W, H, D].
        heatmap: A Tensor of shape [N, 1, W, H, D] of raw center heatmap output
        offsets: A Tensor of shape [N, 3, W, H, D] of raw offset output
        threshold: A Float, threshold applied to the semantic mask if to be applied.
        compute_voting: A Boolean, whether to compute the votes image.
    Returns:
        A tuple of:
            semantic_mask: A binary numpy.ndarray of shape [W, H, D].
            instance_mask: A numpy.ndarray of shape [W, H, D].
            instance_centers: A numpy.ndarray of shape [W, H, D].
            (Optional: voting_image: A numpy.ndarray of shape [W, H, D].)
    """
    assert len(np.unique(semantic_mask)) == 2, "Semantic mask should be binary"
    semantic_mask = remove_connected_components(semantic_mask)

    instance_centers = find_instance_center(heatmap)

    instance_ids = group_pixels(instance_centers, offsets, compute_voting=compute_voting)
    if compute_voting:
        instance_ids, voting_image = instance_ids
    else:
        voting_image = None

    instance_mask = np.transpose(np.squeeze(instance_ids.cpu().numpy().astype(np.int32)), (2,0,1)) * semantic_mask

    ret = (instance_mask, np.squeeze(instance_centers.cpu().numpy().astype(np.uint8)))
    ret += (voting_image.cpu().numpy().astype(np.int16),) if compute_voting else ()
    return (semantic_mask,) + tuple(np.transpose(x, (2, 0, 1)) for x in ret)


def compute_all_voting_image(path_pred):
    import os
    import nibabel as nib

    for f in sorted(os.listdir(path_pred)):
        if f.endswith('pred-offsets.nii.gz') or f.endswith('pred_offsets.nii.gz'):
            offsets_img = nib.load(os.path.join(path_pred, f))
            offsets  = offsets_img.get_fdata()

            binary_segmentation = os.path.join(path_pred, f[:-len('pred-offsets.nii.gz')] + 'seg-binary.nii.gz')
            if not os.path.exists(binary_segmentation):
                binary_segmentation = os.path.join(path_pred, f[:-len('pred_offsets.nii.gz')] + 'seg_binary.nii.gz')
            binary_segmentation = nib.load(binary_segmentation).get_fdata()

            voting_image = compute_voting_image(offsets.transpose(3,0,1,2), binary_segmentation)
            filename = f[:-len('pred-offsets.nii.gz')] + 'voting-image.nii.gz'
            filepath = os.path.join(path_pred, filename)
            nib.save(nib.Nifti1Image(voting_image, offsets_img.affine), filepath)
            print(f"Saved {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--compute_voting', action="store_true", default=False,
                        help="Whether to compute the voting image")
    parser.add_argument('--path_pred', help="Path to the predictions")

    args = parser.parse_args()

    if args.compute_voting:
        compute_all_voting_image(path_pred=args.path_pred)
    else:
        print("Nothing asked, nothing done.")

