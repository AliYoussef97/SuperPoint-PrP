import torch

def match_nn_double(match_prob_dist: torch.Tensor,
                    cross_check: bool = True) -> torch.Tensor:
    """
    Compute the nearest neighbor matches from the match probabilities.
    Inputs:
        match_prob_dist: (N0, N1) torch.Tensor
        dist_thresh: float
        cross_check: bool
    Outputs:
        matches: (N, 2) torch.Tensor
    """
    indices_0 = torch.arange(match_prob_dist.shape[0], device=match_prob_dist.device)
    indices_1 = torch.argmin(match_prob_dist, dim=1) # each index in desc_0 has a corresponding index in desc_1

    if cross_check:
        matches_0 = torch.argmin(match_prob_dist, dim=0) # each index in desc_1 has a corresponding index in desc_0
        mask = torch.eq(indices_0, matches_0[indices_1]) # cross-checking
        indices_0 = indices_0[mask] 
        indices_1 = indices_1[mask]
    
    matches = torch.vstack((indices_0, indices_1)).T

    return matches

def match(desc_0: torch.Tensor,
          desc_1: torch.Tensor,
          cross_check: bool = True) -> torch.Tensor:
    """
    Wrapper function for the matching methods.
    Inputs:
        matching_method: Literal["double_softmax","ratio_test"]
        dist_thresh: float
        cross_check: bool
    Outputs:
        matches: (N, 2) torch.Tensor
        confidence: (N, 1) torch.Tensor
    """    
    distance = 1 - torch.matmul(desc_0, desc_1.T)
    
    matches = match_nn_double(distance, cross_check)

    return matches, distance[matches[:, 0], matches[:, 1]]


def extract_matched_points(kpts: torch.Tensor,
                      matches: torch.Tensor) -> torch.Tensor:
    """
    Extract the keypoints from the matches.
    Inputs:
        kpts: (N, 2) torch.Tensor
        matches: (M) torch.Tensor
    Outputs:
        m_kpts: (M, 2) torch.Tensor
    """
    if len(kpts.shape) == 3:
        kpts = kpts.squeeze()
    m_kpts = kpts[matches]
    matched_kpts = m_kpts[:, [1, 0]]
    return matched_kpts


def matcher(desc_0: torch.Tensor,
            desc_1: torch.Tensor,
            logits_0: torch.Tensor,
            logits_1: torch.Tensor,
            cross_check: bool = True):
    """
    Wrapper function for the matching methods.
    Inputs:
        matching_method: Literal["double_softmax", "ratio_test" , "mnn"]
        dist_thresh: float
        cross_check: bool
    Outputs:
        dict: {"kpts_0": (N, 3) torch.Tensor,
                "kpts_1": (N, 3) torch.Tensor,
                "m_kpts_0": (M, 2) torch.Tensor,
                "m_kpts_1": (M, 2) torch.Tensor,
                "confidence": (M) torch.Tensor}
    """
    
    logits_0, logits_1 = logits_0.squeeze(), logits_1.squeeze()

    desc_0, desc_1 = desc_0.squeeze(0), desc_1.squeeze(0)
    
    kpts_0, kpts_1 = torch.nonzero(logits_0 > 0.0, as_tuple=False), torch.nonzero(logits_1 > 0.0, as_tuple=False)
    
    sparse_desc_0, sparse_desc_1 = desc_0[:, kpts_0[:, 0].long(), kpts_0[:, 1].long()].T, desc_1[:, kpts_1[:, 0].long(), kpts_1[:, 1].long()].T
    
    matches, confidence = match(sparse_desc_0, sparse_desc_1, cross_check)
    
    m_kpts0, m_kpts1 = extract_matched_points(kpts_0, matches[:, 0]), extract_matched_points(kpts_1, matches[:, 1])
    
    return {"kpts_0": kpts_0[:, [1, 0]].float().squeeze().detach().cpu().numpy(),
            "kpts_1": kpts_1[:, [1, 0]].float().squeeze().detach().cpu().numpy(),
            "m_kpts_0": m_kpts0.float().detach().cpu().numpy(),
            "m_kpts_1": m_kpts1.float().detach().cpu().numpy(),
            "confidence": confidence.detach().cpu().numpy()}