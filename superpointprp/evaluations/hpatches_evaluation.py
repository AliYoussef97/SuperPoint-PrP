import torch
from tqdm import tqdm
from superpointprp.utils.match import matcher
from superpointprp.evaluations.eval_utils.hpatches_utils import compute_repeatability, estimate_homography, mean_matching_acc, hpatches_metrics


@torch.no_grad()
def estimate_hpatches_metrics(config, model, data_loader, device):
    
    repeatability = []
    homogaphy_est_acc = []
    homography_est_err = []
    avg_pre_match_points = 0.0
    avg_post_match_points = 0.0
    MMA = 0.0
    num_matches = 0.0

    for batch in tqdm(data_loader):
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        output = model(batch['image'])

        warped_output = model(batch['warped_image'])

        logits_0, desc_0 = output["prob_heatmap_nms"], output["desc"]

        logits_1, desc_1 = warped_output["prob_heatmap_nms"], warped_output["desc"]

        m_output = matcher(desc_0, 
                           desc_1,
                           logits_0,
                           logits_1,
                           config["matcher"]["cross_check"])
                
        rep = compute_repeatability(m_output["kpts_0"][:, :2], m_output["kpts_1"][:, :2], batch['H'], batch["size"], config["data"]["dist_thresh"])
        
        h_est_acc, h_est_err, inliers = estimate_homography(m_output["m_kpts_0"], m_output["m_kpts_1"], batch['H'], batch["size"], config["data"]["dist_thresh"])

        mm_acc = mean_matching_acc(m_output["m_kpts_0"], m_output["m_kpts_1"], batch['H'], config["data"]["dist_thresh"])

        avg_pre_match_points += int(m_output["kpts_0"].shape[0])
        avg_post_match_points += int(m_output["m_kpts_0"].shape[0])
        repeatability.append(rep)
        homogaphy_est_acc.append(h_est_acc)
        homography_est_err.append(h_est_err)
        MMA += mm_acc
        num_matches += 1.0

    hpatches_metrics(repeatability, 
                     homogaphy_est_acc, 
                     homography_est_err, 
                     MMA, 
                     num_matches, 
                     avg_pre_match_points,
                     avg_post_match_points,
                     len(data_loader.dataset), 
                     config["data"]["dist_thresh"])