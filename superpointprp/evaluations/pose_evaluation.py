import torch
import numpy as np
from tqdm import tqdm
from superpointprp.utils.match import matcher
from superpointprp.evaluations.eval_utils.pose_utils import compute_epipolar_error, estimate_pose, compute_pose_error, pose_estimation_metrics


@torch.no_grad()
def estimate_pose_errors(config, model, data_loader, device):
    
    all_metrics = []
    len_pairs = len(data_loader.dataset)

    for batch in tqdm(data_loader):
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        output = model(batch['inp0'])

        warped_output = model(batch['inp1'])

        logits_0, desc_0 = output["detector_output"]["prob_heatmap_nms"], output["descriptor_output"]["desc"]

        logits_1, desc_1 = warped_output["detector_output"]["prob_heatmap_nms"], warped_output["descriptor_output"]["desc"]

        m_output = matcher(desc_0, 
                           desc_1,
                           logits_0,
                           logits_1,
                           config["matcher"]["cross_check"])

        epi_errs = compute_epipolar_error(m_output["m_kpts_0"], m_output["m_kpts_1"], batch['T_0to1'], batch['K0'], batch['K1'])
        correct = epi_errs < config["data"]["epi_thrsehold"]
        num_correct = np.sum(correct)
        precision = np.mean(correct) if len(correct) > 0 else 0
        matching_score = num_correct / len(m_output["kpts_0"]) if len(m_output["kpts_0"]) > 0 else 0

        ret = estimate_pose(m_output["m_kpts_0"], m_output["m_kpts_1"], batch['K0'], batch['K1'], config["matcher"]["ransac_thresh"])
        if ret is None:
            err_t, err_R = np.inf, np.inf
        else:
            R, t, inliers = ret
            err_t, err_R = compute_pose_error(batch['T_0to1'], R, t)
        
        out_eval = {'error_t': err_t,
                    'error_R': err_R,
                    'precision': precision,
                    'matching_score': matching_score,
                    'num_correct': num_correct,
                    'epipolar_errors': epi_errs}
        
        all_metrics.append(out_eval)
    
    pose_estimation_metrics(all_metrics, len_pairs)