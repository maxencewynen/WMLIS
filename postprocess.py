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


def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=200):
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
    return votes*100



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
    z_coord = torch.arange(depth, dtype=torch.float16, device=offsets.device).unsqueeze(1).unsqueeze(2).repeat(1, height, width).unsqueeze(0)
    y_coord = torch.arange(height, dtype=torch.float16, device=offsets.device).unsqueeze(0).unsqueeze(2).repeat(depth, 1, width).unsqueeze(0)
    x_coord = torch.arange(width, dtype=torch.float16, device=offsets.device).unsqueeze(0).unsqueeze(1).repeat(depth, height, 1).unsqueeze(0)

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
        if compute_voting: return torch.zeros_like(coord), torch.squeeze(votes)
        else: return torch.zeros_like(coord)

    ctr_loc = ctr_loc.view(3, depth * height * width).transpose(1, 0)
    
    del z_coord, y_coord, x_coord, coord
    torch.cuda.empty_cache()

    # ctr: [K, 3] -> [K, 1, 3]
    # ctr_loc = [D*H*W, 3] -> [1, D*H*W, 3]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    # Compute the memory needed to store the distance matrix
    memory_needed = (ctr.shape[0] * ctr_loc.shape[1] * ctr_loc.element_size())
    free, total = torch.cuda.mem_get_info()
    if memory_needed < free - 0.5e9:
        # Compute the Euclidean distance in 3D space
        distance = torch.norm(ctr - ctr_loc, dim=-1)  # [K, D*H*W]

        # Find the center with the minimum distance at each voxel, offset by 1, to reserve id=0 for stuff
        instance_id = torch.argmin(distance, dim=0).short().view(1, depth, height, width) + 1 # [1, D, H, W]
        del distance, ctr_loc, ctr
        torch.cuda.empty_cache()
    else:
        total_elements = ctr_loc.shape[1]  # Assuming dim=1 is the element dimension
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



def postprocess(semantic_mask, heatmap, offsets, compute_voting=False):
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
    assert len(np.unique(semantic_mask)) <= 2, "Semantic mask should be binary"
    semantic_mask = remove_connected_components(semantic_mask)

    instance_centers = find_instance_center(heatmap)

    instance_ids = group_pixels(instance_centers, offsets, compute_voting=compute_voting)
    if compute_voting:
        instance_ids, voting_image = instance_ids
    else:
        voting_image = None

    instance_mask = np.squeeze(instance_ids.cpu().numpy().astype(np.int32)) * semantic_mask

    centers_mx = np.zeros_like(semantic_mask)
    instance_centers = instance_centers.cpu().numpy()
    centers_mx[instance_centers[:, 0], instance_centers[:, 1], instance_centers[:, 2]] = 1

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

