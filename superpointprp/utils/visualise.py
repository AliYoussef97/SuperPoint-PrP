import cv2
import numpy as np

def img_pair_visual(image1: np.ndarray, 
                    image2: np.ndarray,
                    matched_keypoints: np.ndarray,
                    matched_warped_keypoints: np.ndarray) -> np.ndarray:
    """
    Visualise the matching of keypoints between two images.
    - Inputs:
        - image1: (H, W)
        - image2: (H, W) 
        - matched_keypoints: (N, 2) x,y coordinates. 
        - matched_warped_keypoints: (M, 2) x,y coordinates.
    Output:
        - image_pair: (2*H, W, 3)
    """
    H = image1.shape[0]
    image1 = np.stack([image1] * 3, axis=-1)
    image2 = np.stack([image2] * 3, axis=-1)
    image_pair = np.vstack((image1, image2))
    matched_keypoints = matched_keypoints.astype(int)
    matched_warped_keypoints = matched_warped_keypoints.astype(int)
    for point1, point2 in zip(matched_keypoints, matched_warped_keypoints):
        point2[1] += H
        image_pair = cv2.circle(image_pair, tuple(point1), 1, (255, 0, 0), -1)
        image_pair = cv2.circle(image_pair, tuple(point2), 1, (0, 0, 255), -1)
        image_pair = cv2.line(image_pair, tuple(point1), tuple(point2), (0, 255, 0), 1)
    return image_pair