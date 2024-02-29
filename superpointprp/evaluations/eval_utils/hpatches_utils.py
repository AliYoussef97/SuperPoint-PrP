# Parts of this code is from SuperGlue[https://github.com/magicleap/SuperGluePretrainedNetwork]

import numpy as np
import cv2
from tabulate import tabulate

def warp_keypoints(keypoints: np.ndarray, 
                   H: np.ndarray) -> np.ndarray:
    """ Warp keypoints with given homography.
    Inputs:
        keypoints: (N, 2) array of (x, y) keypoint coordinates
        H: (3, 3) array
    Outputs:
        warped_keypoints: (N, 2) array of warped (x, y) keypoint coordinates
    """

    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                        axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]


def keep_true_keypoints(points: np.ndarray,
                        H: np.ndarray,
                        shape: tuple) -> np.ndarray:
    """ 
    Keep only the keypoints that can be warped by the homography and
    lie inside the image boundaries.
    Inputs:
        points: (N, 2) array of (x, y) keypoint coordinates
        H: (3, 3) array
        shape: (height, width) of the image
    Outputs:
        warped_points: (N, 2) array of warped (x, y) keypoint coordinates
    """
    
    warped_points = warp_keypoints(points, H)
    mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[1]) &\
            (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[0])
    return points[mask, :]


def compute_repeatability(mkpts0: np.ndarray, 
                          mkpts1: np.ndarray, 
                          H: np.ndarray,
                          img_shape: tuple,
                          dist_thresh: float = 3.0) -> float:
        """
        Compute repeatability between two sets of keypoints.
        Inputs:
            mkpts0: (N, 2) array of (x, y) keypoint coordinates
            mkpts1: (N, 2) array of (x, y) keypoint coordinates
            H: (3, 3) array
            img_shape: (height, width) of the image
            dist_thresh: threshold on matching distance
        Outputs:
            rep: repeatability
        """
        warped_points = keep_true_keypoints(mkpts1, 
                                            np.linalg.inv(H), 
                                            img_shape)
        
        true_warped_points = keep_true_keypoints(mkpts0,
                                                 H,
                                                 img_shape)
        
        true_warped_points = warp_keypoints(true_warped_points, H)
        
        N1 = true_warped_points.shape[0]
        N2 = warped_points.shape[0]

        true_warped_points = np.expand_dims(true_warped_points, axis=1)
        warped_points = np.expand_dims(warped_points, axis=0)
        
        norm = np.linalg.norm(true_warped_points-warped_points, ord=None, axis=2)

        count_0 = 0
        count_1 = 0

        if N1 != 0:
            min_0 = np.min(norm, axis=1)
            count_0 = np.sum(min_0 <= dist_thresh)
        if N2 != 0:
            min_1 = np.min(norm, axis=0)
            count_1 = np.sum(min_1 <= dist_thresh)
        if N1 + N2 > 0:
            rep = (count_0 + count_1) / (N1 + N2)
        
        return rep


def hpatches_auc(errors: list,
                 thresholds: list) -> list:
    """
    Compute AUC scores for different thresholds.
    Inputs:
        errors: list of errors
        thresholds: list of thresholds
    Outputs:
        aucs: list of auc scores
    """
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round((np.trapz(r, x=e) / t), 4))
    return aucs


def estimate_homography(m_kpts0: np.ndarray,
                        m_kpts1: np.ndarray,
                        H: np.ndarray,
                        shape: tuple,
                        dist_thresh: float = 3.0) -> np.ndarray:
    """
    Estimate homography between two sets of keypoints.
    Inputs:
        m_kpts0: (N, 2) array of (x, y) keypoint coordinates
        m_kpts1: (N, 2) array of (x, y) keypoint coordinates
        H: (3, 3) array
        shape: (height, width) of the image
        dist_thresh: threshold on matching distance
    Outputs:
        correctness: 1 if estimated homography is correct, 0 otherwise
        mean_dist: mean distance between true warped points and estimated warped points
        inliers: (N, ) array of inliers
    """
     
    estimated_H, inliers = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC)
    
    if estimated_H is None:
        return 0, dist_thresh+1., np.zeros(m_kpts0.shape[0])

    inliers = inliers.flatten()

    corners = np.array([[0, 0, 1],
                        [shape[1] - 1, 0, 1],
                        [0, shape[0] - 1, 1],
                        [shape[1] - 1, shape[0] - 1, 1]])
    
    real_warped_corners = np.dot(corners, np.transpose(H))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
   
    warped_corners = np.dot(corners, np.transpose(estimated_H))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    
    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    
    correctness = float(mean_dist <= dist_thresh)

    return correctness, mean_dist, inliers


def mean_matching_acc(m_kpts0: np.ndarray,
                      m_kpts1: np.ndarray,
                      H: np.ndarray,
                      thresh: float = 3.0) -> float:
    """
    Compute mean matching accuracy.
    Inputs:
        m_kpts0: (N, 2) array of (x, y) keypoint coordinates
        m_kpts1: (N, 2) array of (x, y) keypoint coordinates
        H: (3, 3) array
        thresh: threshold on matching distance  
    Outputs:
        mean_matching_acc: mean matching accuracy
    """

    true_warped_points = warp_keypoints(m_kpts0, H)

    mask_keep = np.sqrt(np.sum((true_warped_points - m_kpts1)**2, axis=1)) <= thresh

    if mask_keep.size > 0:
        mean_matching_acc = np.mean(mask_keep).astype(np.float32)
    else:
        mean_matching_acc = 0.0
    
    return mean_matching_acc


def hpatches_metrics(repeatability: list,
                     homogaphy_est_acc: list,
                     homography_est_err: list,
                     MMA: float,
                     num_matches: float,
                     avg_pre_match_points: float,
                     avg_post_match_points: float,
                     len_data: int,
                     threshold: float = 3.0) -> None:
        """
        Print hpatches metrics.
        Inputs:
            repeatability: list of repeatability scores
            homogaphy_est_acc: list of homography estimation accuracies
            homography_est_err: list of homography estimation errors
            MMA: mean matching accuracy
            num_matches: number of matches
        """
        repeatability = np.mean(repeatability)
        homogaphy_est_acc = np.mean(homogaphy_est_acc)
        homography_est_auc = hpatches_auc(homography_est_err, [threshold])
        MMA = MMA / num_matches
        avg_pre_match_points = avg_pre_match_points / int(len_data)
        avg_post_match_points = avg_post_match_points / int(len_data)

        print('Evaluation Results (mean over {} pairs) at threshold {}: \n'.format(len_data, threshold))
        print(tabulate([[repeatability, homogaphy_est_acc, homography_est_auc[0], MMA, avg_pre_match_points, avg_post_match_points]],
                       headers=['Repeatability', 'Homography Estimation Accuracy', 
                                'Homography Estimation AUC', 'Mean Matching Accuracy' , 
                                '# Keypoints Pre Match', '# Keypoints Post Match'],
                       tablefmt='orgtbl'))