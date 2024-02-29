import torch

def filter_points(points, shape, device='cpu', return_mask=False):
    """
    Remove points close to the border of the image.
    input:
        points: (N,2)
        shape: torch tensor (H,W)
    output:
        points: (N,2)
    """
    if len(points)!=0:
        H,W = shape
        mask  = (points[:,0] >= 0) & (points[:,0] < H-1) & (points[:,1] >= 0) & (points[:,1] < W-1)
        if return_mask:
            return points[mask], mask
        return points[mask]
    else:
        return points

def compute_keypoint_map(points, shape, device='cpu'):
    """
    input:
        points: (N,2)
        shape: torch tensor (H,W)
    output:
        kmap: (H,W)
    """
    H, W = shape
    coord = torch.round(points).to(torch.int32)
    mask = (coord[:, 0] >= 0) & (coord[:, 0] < H-1) & (coord[:, 1] >= 0) & (coord[:, 1] < W-1)
    k_map = torch.zeros(shape, dtype=torch.int32, device=device)
    k_map[coord[mask, 0], coord[mask, 1]] = 1
    return k_map

def warp_points(points, homography, device='cpu'):
    """
    :param points: (N,2), tensor
    :param homographies: [B, 3, 3], batch of homographies
    :return: warped points B,N,2
    """
    if len(points.shape)==0:
        return points

    points = torch.fliplr(points)
    
    batch_size = homography.shape[0]
    
    points = torch.cat((points, torch.ones((points.shape[0], 1),device=device)),dim=1)

    warped_points = torch.tensordot(homography, points.transpose(1,0),dims=([2], [0]))

    warped_points = warped_points.reshape([batch_size, 3, -1])
    
    warped_points = warped_points.transpose(2, 1)
    
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    
    warped_points = torch.flip(warped_points,dims=(2,))
    
    warped_points = warped_points.squeeze(0)
       
    return warped_points

def warp_points_NeRF(points: torch.Tensor,
                     depth: torch.Tensor,
                     cam_intrinsic_matrix: torch.Tensor,
                     input_rotation: torch.Tensor,
                     input_translation: torch.Tensor, 
                     warp_rotation: torch.Tensor,
                     warp_translation: torch.Tensor, 
                     device='cpu') -> torch.Tensor:
    """
    Warp keypoints from the input frame to a different viewpoint frame.
    - Input
        - points: (N, 2) tensor
        - depth: (B, H, W) tensor
        - cam_intrinsic_matrix: (B, 3, 3) tensor
        - input_rotation: (B, 3, 3) tensor
        - input_translation: (B, 3, 1) tensor
        - warp_rotation: (B, 3, 3) tensor
        - warp_translation: (B, 3, 1) tensor
    - Output
        - warped_points: (B, N, 2) tensor
    """
    
    if len(points.shape)==0:
        return points

    B, H, W = depth.shape
    
    points_temp = points.floor().to(torch.int32)

    flat_points = points_temp[:, 0] * W + points_temp[:, 1]

    # Create 5x5 (flattned) patch around each feature point
    offset = torch.arange(-2, 3, device=device)
    offset = torch.stack((offset.repeat_interleave(5), 
                            offset.repeat(5)), dim=0).T
    
    # mask points that are close to the border
    mask = (points_temp[:, 0] <= 2) | (points_temp[:, 1] <= 2) | (points_temp[:, 0] >= H-2) | (points_temp[:, 1] >= W-2)
    
    depth_batch = torch.zeros((B, points_temp.shape[0]), device=device)

    for i, dp in enumerate(depth):

        # Take depth values at feature points location if they are close to the border
        depth_batch[i, mask] = dp[points_temp[mask, 0], points_temp[mask, 1]]

        dp = dp.flatten()
    
        depth_values = torch.empty((points_temp[~mask].shape[0], len(offset)), device=device)

        for j, off in enumerate(offset):
            patch = flat_points[~mask] + (off[0] * W + off[1])
            depth_values[:, j] = dp[patch]
        
        min_depth, max_depth = torch.min(depth_values, dim=1)[0], torch.max(depth_values, dim=1)[0]
        
        # If there is a large difference between the min and max depth values of the patch, take the min depth value,
        # otherwise take the depth value at the feature point location
        depth_batch[i, ~mask] = torch.where((max_depth - min_depth) >= 0.03, min_depth, dp[flat_points[~mask]].flatten())

    depth_values = depth_batch.unsqueeze(1).to(device)

    points = torch.fliplr(points)
    
    points = torch.cat((points, torch.ones((points.shape[0], 1), device=device)), dim=1)
    warped_points = torch.tensordot(torch.linalg.inv(cam_intrinsic_matrix), points, dims=([2], [1]))
    warped_points /= torch.linalg.norm(warped_points, dim=(1), keepdim=True)
    warped_points *= depth_values
    warped_points = input_rotation @ warped_points + input_translation    
    warped_points = torch.linalg.inv(warp_rotation) @ warped_points - (torch.linalg.inv(warp_rotation) @ warp_translation)
    warped_points = cam_intrinsic_matrix @ warped_points

    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:,:, :2] / warped_points[:,:, 2:]
    warped_points = torch.flip(warped_points, dims=(2,))

    warped_points = warped_points.squeeze(0)

    return warped_points