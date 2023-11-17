import numpy as np
from scipy.ndimage import maximum_filter, generate_binary_structure, label, labeled_comprehension
import argparse
import torch
import torch.nn.functional as F


def postprocess_probability_segmentation(probability_segmentation, threshold=0.5, voxel_size=(1,1,1), l_min=14):
    """
    Constructs an instance segmentation mask from a lesion probability matrix, by applying a threshold
     and removing all lesions with less volume than `l_min`.
    Args:
        probability_segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
        threshold: `float`, threshold to apply to the binary segmentation mask.
        voxel_size: `tuple` of length 3, with the voxel size in mm.
        l_min:  `int`, minimal volume of a lesion.
    Returns:
        Instance lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
    """
    # Threshold the image
    binary_data = np.where(probability_segmentation >= threshold, 1, 0).astype(np.uint8)

    # Remove objects smaller than l_min voxels
    binary_data = remove_small_lesions_from_binary_segmentation(binary_data, voxel_size=voxel_size, l_min=l_min)

    # Find connected components larger than l_min voxels
    labeled_array, num_features = label(binary_data)

    return labeled_array


def remove_small_lesions_from_instance_segmentation(instance_segmentation, voxel_size, l_min=14):
    """
    Remove all lesions with less volume than `l_min` from an instance segmentation mask `instance_segmentation`.
    Args:
        instance_segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
        voxel_size: `tuple` of length 3, with the voxel size in mm.
        l_min:  `int`, minimal volume of a lesion.
    Returns:
        Instance lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
    """

    assert type(voxel_size) == tuple, "Voxel size should be a tuple"
    assert len(voxel_size) == 3, "Voxel size should be a tuple of length 3"

    label_list, label_counts = np.unique(instance_segmentation, return_counts=True)

    instance_seg2 = np.zeros_like(instance_segmentation)

    for lid, lvoxels in zip(label_list, label_counts):
        if lid == 0: continue

        this_instance_indices = np.where(instance_segmentation == lid)
        size_along_x = (1 + max(this_instance_indices[0]) - min(this_instance_indices[0])) * voxel_size[0]
        size_along_y = (1 + max(this_instance_indices[1]) - min(this_instance_indices[1])) * voxel_size[1]
        size_along_z = (1 + max(this_instance_indices[2]) - min(this_instance_indices[2])) * voxel_size[2]

        # if the connected component is smaller than 3 voxels in any direction, skip it as it is not
        # clinically considered a lesion
        if size_along_x < 3 or size_along_y < 3 or size_along_z < 3:
            continue

        if lvoxels * np.prod(voxel_size) > l_min:
            instance_seg2[instance_segmentation == lid] = lid

    return instance_seg2


def remove_small_lesions_from_binary_segmentation(binary_segmentation, voxel_size, l_min=14):
    """
    Remove all lesions with less volume than `l_min` from a binary segmentation mask `binary_segmentation`.
    Args:
        binary_segmentation: `numpy.ndarray` of shape [H, W, D], with a binary lesions segmentation mask.
        voxel_size: `tuple` of length 3, with the voxel size in mm.
        l_min:  `int`, minimal volume of a lesion.
    Returns:
        Binary lesion segmentation mask (`numpy.ndarray` of shape [H, W, D])
    """

    assert type(voxel_size) == tuple, "Voxel size should be a tuple"
    assert len(voxel_size) == 3, "Voxel size should be a tuple of length 3"
    assert np.unique(binary_segmentation).tolist() == [0, 1], "Segmentation should be binary"

    labeled_seg, num_labels = label(binary_segmentation)
    label_list = np.unique(labeled_seg)
    num_elements_by_lesion = labeled_comprehension(binary_segmentation, labeled_seg, label_list, np.sum, float, 0)

    seg2 = np.zeros_like(binary_segmentation)
    for i_el, n_el in enumerate(num_elements_by_lesion):
        this_instance_indices = np.where(labeled_seg == i_el)
        this_instance_mask = np.stack(this_instance_indices, axis=1)

        size_along_x = (1 + max(this_instance_indices[0]) - min(this_instance_indices[0])) * voxel_size[0]
        size_along_y = (1 + max(this_instance_indices[1]) - min(this_instance_indices[1])) * voxel_size[1]
        size_along_z = (1 + max(this_instance_indices[2]) - min(this_instance_indices[2])) * voxel_size[2]

        # if the connected component is smaller than 3 voxels in any direction, skip it as it is not
        # clinically considered a lesion
        if size_along_x < 3 or size_along_y < 3 or size_along_z < 3:
            continue

        lesion_size = n_el * np.prod(voxel_size)
        if lesion_size > l_min:
            current_voxels = this_instance_mask
            seg2[current_voxels[:, 0],
            current_voxels[:, 1],
            current_voxels[:, 2]] = 1
    return seg2


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


def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=100):
    """
    from https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W, D] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep. If None, all centers > threshold are kept
    Returns:
        A Tensor of shape [K, 3] where K is the number of center points. The order of second dim is (x, y, z).
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
    nonzeros = (ctr_hmp > 0).short()
    
    # Find clusters of centers to consider them as one center instead of two
    centers_labeled, num_centers = label(nonzeros.cpu().numpy())
    centers_labeled = torch.from_numpy(centers_labeled).to(nonzeros.device)
    for c in list(range(1, num_centers + 1)):
        coords_cx, coords_cy, coords_cz = torch.where(centers_labeled == c)
        
        # if center is made of two voxels or more 
        if len(coords_cx) > 1: 
            # keep only one center voxel at random, since all of them have the same probability
            # of being a center
            coord_to_keep = np.random.choice(list(range(len(coords_cx))))

            # set all the other center voxels to zero
            for i in range(len(coords_cx)):
                if i != coord_to_keep:
                    ctr_hmp[coords_cx[i], coords_cy[i], coords_cz[i]] = -1
    
    # Make the list of centers from the updated ctr_hmp
    ctr_all = torch.nonzero(ctr_hmp > 0).short()

    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1]).short()


def make_votes_readable(votes):
    votes = torch.log(votes + 1, out=torch.zeros_like(votes, dtype=torch.float32))
    votes = F.avg_pool3d(votes, kernel_size=3, stride=1, padding=1)
    return votes * 100


def group_pixels(ctr, offsets, compute_voting=False):
    """
    Inspired from https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/post_processing/instance_post_processing.py
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
    z_coord, y_coord, x_coord = torch.meshgrid(
        torch.arange(depth),
        torch.arange(height),
        torch.arange(width),
        indexing="ij"
    )
    z_coord = z_coord[None, :].to(offsets.device)
    y_coord = y_coord[None, :].to(offsets.device)
    x_coord = x_coord[None, :].to(offsets.device)

    coord = torch.cat((z_coord, y_coord, x_coord), dim=0)

    ctr_loc = (coord + offsets).half()

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

    if ctr.shape[0] == 0:
        if compute_voting:
            return torch.zeros_like(coord), torch.squeeze(votes)
        else:
            return torch.zeros_like(coord)

    ctr_loc = ctr_loc.view(3, depth * height * width).transpose(1, 0)

    del z_coord, y_coord, x_coord, coord
    torch.cuda.empty_cache()

    # ctr: [K, 3] -> [K, 1, 3]
    # ctr_loc = [D*H*W, 3] -> [1, D*H*W, 3]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)
    
    # Compute the distances in batches to avoid memory issues
    total_elements = ctr_loc.shape[1]  
    batch_size = 1e6
    num_batches = (total_elements + batch_size - 1) // batch_size

    # Initialize a list to store the results for each batch
    instance_id_batches = []

    for batch_idx in range(int(num_batches)):
        start_idx = int(batch_idx * batch_size)
        end_idx = int(min((batch_idx + 1) * batch_size, total_elements))

        # Process a batch of elements
        ctr_loc_batch = ctr_loc[:, start_idx:end_idx]  # Slice along dim=1
        distance_batch = torch.norm(ctr - ctr_loc_batch, dim=-1)  # [K, batch_size]

        # Find the center with the minimum distance at each voxel, offset by 1
        instance_id_batch = torch.argmin(distance_batch, dim=0).short() + 1
        instance_id_batches.append(instance_id_batch)

    # Concatenate the results along the batch dimension
    instance_id = torch.cat(instance_id_batches, dim=0).view(1, depth, height, width)

    if compute_voting:
        return instance_id, torch.squeeze(votes)
    return instance_id


def refine_instance_segmentation(instance_mask, l_min=14):
    """
    Refines the instance segmentation by relabeling disconnected components in instances
    and removing instances smaller than l_min
    Args:
        instance_maks: np.ndarray of dimension (H,W,D), array of instance ids
        l_min: minimum lesion size
    """
    iids = np.unique(instance_mask)[1:]
    max_instance_id = np.max(instance_mask)
    # for every instance id
    for iid in iids:
        # get the mask
        mask = (instance_mask == iid)
        components, n_components = label(mask)
        if n_components > 1: # if the lesion is split in n components
            biggest_lesion_size = 0
            biggest_lesion_id = -1
            for cid in range(1, n_components + 1): # go through each component
                component_mask = (components == cid)
                this_size = np.sum(component_mask)
                if this_size > biggest_lesion_size:
                    biggest_lesion_size = this_size
                    biggest_lesion_id = cid
            for cid in range(1, n_components + 1):
                if cid == biggest_lesion_id: continue
                instance_mask[components == cid] = 0

        elif np.sum(mask) < l_min:  # check if lesion size is too small or not
            instance_mask[mask] = 0
            instance_mask[instance_mask == max_instance_id] = iid
            max_instance_id -= 1
    return instance_mask


def calibrate_offsets(offsets, centers):
    """
    Calibrates the offsets by subtracting the mean offset at center locations
    Args:
        offsets: A Tensor of shape [N, 3, W, H, D] of raw offset output, where N is the batch size (N=1 expected)
        centers: Binary np.ndarray of dimension (H, W, D), array of centers
    """
    bias_x, bias_y, bias_z = torch.mean(offsets[:,:,centers == 1], axis=2).squeeze()
    offsets[:, 0, :, :, :] = offsets[:, 0, :, :, :] - bias_x
    offsets[:, 1, :, :, :] = offsets[:, 1, :, :, :] - bias_y
    offsets[:, 2, :, :, :] = offsets[:, 2, :, :, :] - bias_z
    return offsets


def postprocess(semantic_mask, heatmap, offsets, compute_voting=False, heatmap_threshold=0.1,
                voxel_size=(1, 1, 1), l_min=14):
    """
    Postprocesses the semantic mask, center heatmap and the offsets.
    Arguments:
        semantic_mask: a binary numpy.ndarray of shape [W, H, D].
        heatmap: A Tensor of shape [N, 1, W, H, D] of raw center heatmap output
        offsets: A Tensor of shape [N, 3, W, H, D] of raw offset output
        compute_voting: A Boolean, whether to compute the votes image.
        heatmap_threshold: A Float, threshold applied to the center heatmap score.
        voxel_size: A tuple of length 3, with the voxel size in mm.
        l_min:  An Integer, minimal volume of a lesion.
    Returns:
        A tuple of:
            semantic_mask: A binary numpy.ndarray of shape [W, H, D].
            instance_mask: A numpy.ndarray of shape [W, H, D].
            instance_centers: A numpy.ndarray of shape [W, H, D].
            (Optional: voting_image: A numpy.ndarray of shape [W, H, D].)
    """
    assert len(np.unique(semantic_mask)) <= 2, "Semantic mask should be binary"
    semantic_mask = remove_small_lesions_from_binary_segmentation(semantic_mask, voxel_size=voxel_size, l_min=l_min)

    instance_centers = find_instance_center(heatmap, threshold=heatmap_threshold)

    centers_mx = np.zeros_like(semantic_mask)
    ic = instance_centers.cpu().numpy()
    centers_mx[ic[:, 0], ic[:, 1], ic[:, 2]] = 1

    offsets = calibrate_offsets(offsets, centers_mx)

    instance_ids = group_pixels(instance_centers, offsets, compute_voting=compute_voting)
    if compute_voting:
        instance_ids, voting_image = instance_ids
    else:
        voting_image = None

    instance_mask = np.squeeze(instance_ids.cpu().numpy().astype(np.int32)) * semantic_mask
    instance_mask = remove_small_lesions_from_instance_segmentation(instance_mask, voxel_size=voxel_size, l_min=l_min)
    instance_mask = refine_instance_segmentation(instance_mask, l_min=l_min)

    ret = (instance_mask, centers_mx.astype(np.uint8))
    ret += (voting_image.cpu().numpy().astype(np.int16),) if compute_voting else ()
    return (semantic_mask,) + ret


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
