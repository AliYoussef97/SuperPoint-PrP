# This code is from Superpoint[https://github.com/rpautrat/SuperPoint]

import torch
import torchvision

def remove_borders(keypoints,scores, border: int, height: int, width: int):
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0, remove_bord=False):

    pts = torch.nonzero(prob >= min_prob, as_tuple=False).to(torch.float32)

    size = torch.tensor(size/2.)

    boxes = torch.cat((pts - size, pts + size), dim=1).to(torch.float32)

    scores = prob[pts[:, 0].long(), pts[:, 1].long()]

    indices = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou)

    pts = torch.index_select(pts,0,indices) #pts[indices, :]
    
    scores = torch.index_select(scores,0,indices) #scores[indices]

    if remove_bord:
        pts, scores = remove_borders(pts, scores, border=remove_bord, height=prob.shape[0], width=prob.shape[1])

    if keep_top_k:
        k = min(scores.shape[0], keep_top_k)
        scores, indices = torch.topk(scores, k, dim=0, sorted=True)
        pts = torch.index_select(pts, 0, indices) #pts[indices, :]
       
    
    nms_prob = torch.zeros_like(prob)
    nms_prob[pts[:, 0].long(), pts[:, 1].long()] = scores
    
    return nms_prob