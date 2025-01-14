from pathlib import Path
import os
import torch
import kornia.geometry.transform as K
import kornia
import cv2
import numpy as np
from tqdm import tqdm
from superpointprp.data.data_utils.homographic_augmentation import Homographic_aug
from superpointprp.models.model_utils.sp_utils import box_nms
from superpointprp.settings import EXPER_PATH
from superpointprp.utils.train_utils import move_to_device
from superpointprp.data.data_utils.kp_utils import warp_points_NeRF, filter_points
import random

class ExportDetections():
    def __init__(self, config, model, dataloader, split, enable_HA, device):
        
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.split = split
        self.enable_HA = enable_HA
        if self.enable_HA:
            print(f'\033[92m✅ Homography Adaptation enabled \033[0m')
        self.device = device
        self.output_dir = self._init_output_dir()
        self.one_homography = Homographic_aug(config['homography_adaptation'], self.device)
        self.homography_adaptation()

    def _init_output_dir(self):
        """
        Where to save the outputs.
        """
        output_dir = Path(EXPER_PATH, 'outputs', self.config['data']['experiment_name'], self.split)
        if not output_dir.exists():
            os.makedirs(output_dir)
        return output_dir
    

    @torch.no_grad()
    def step(self,image,probs,counts):
        
        image_shape = image.shape[2:]
        
        H = self.one_homography.sample_homography(shape=image_shape,
                                                  **self.config['homography_adaptation']['params']) # 1,3,3
        H_inv = torch.inverse(H) # 1,3,3

        warped_image = K.warp_perspective(image, H, dsize=(image_shape), align_corners=True) # 1,1,H,W
       
        mask = K.warp_perspective(torch.ones_like(warped_image,device=self.device), H, dsize=(image_shape), mode='nearest', align_corners=True) # 1,1,H,W

        count = K.warp_perspective(torch.ones_like(warped_image,device=self.device), H_inv, dsize=(image_shape), mode='nearest', align_corners=True) # 1,1,H,W

        if self.config['homography_adaptation']['valid_border_margin']:
            erosion = self.config['homography_adaptation']['valid_border_margin']
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion*2,)*2)
            kernel = torch.as_tensor(kernel,device=self.device, dtype=torch.float32)
            
            mask = kornia.morphology.erosion(mask,kernel).to(torch.int32) # 1,1,H,W
            mask = mask.squeeze(1) # 1,H,W

            count = kornia.morphology.erosion(count,kernel).to(torch.int32) # 1,1,H,W
            count = count.squeeze(1) # 1,H,W

        
        prob = self.model(warped_image)['detector_output']['prob_heatmap'] # 1,H,W
        prob *= mask

        prob_proj = K.warp_perspective(prob.unsqueeze(0), H_inv, dsize=(image_shape), mode='bilinear', align_corners=True) # 1,1,H,W
        prob_proj = prob_proj.squeeze(1) # 1,H,W
        prob_proj *= count # 1,H,W
        
        probs = torch.concat([probs, prob_proj.unsqueeze(1)], dim=1)
        counts = torch.concat([counts, count.unsqueeze(1)], dim=1)
        
        return probs, counts   

    
    @torch.no_grad()
    def homography_adaptation(self):
        for data in tqdm(self.dataloader, desc=f"Exporting detections",colour="green"):

            data = move_to_device(data,self.device)

            name = data["name"][0]
            save_path = Path(self.output_dir, '{}.npy'.format(name))
            if save_path.exists():
                continue
           
            probs = self.model(data["raw"]["image"])["detector_output"]["prob_heatmap"] # 1,H,W
            

            if self.enable_HA:

                counts = torch.ones_like(probs,device=self.device) # 1,H,W
                
                probs = probs.unsqueeze(1) # 1,1,H,W
                counts = counts.unsqueeze(1) # 1,1,H,W
                
                for _ in range(self.config["homography_adaptation"]["num"]-1):
                    probs, counts = self.step(data["raw"]["image"], probs, counts) # 1,num,H,W, 1,num,H,W

                counts = torch.sum(counts, dim=1) # 1,H,W
                max_prob, _ = torch.max(probs, dim=1) # 1,H,W
                mean_prob = torch.sum(probs, dim=1) / counts # 1,H,W
            
                if self.config["homography_adaptation"]["aggregation"] == "max":
                    probs = max_prob # 1,H,W
                
                if self.config["homography_adaptation"]["aggregation"] == "sum":
                    probs = mean_prob # 1,H,W
            
            probs = [box_nms(prob=pb,
                             size=self.config["model"]["detector_head"]["nms"],
                             min_prob=self.config["model"]["detector_head"]["det_thresh"],
                             keep_top_k=self.config["model"]["detector_head"]["top_k"],
                             remove_bord=self.config["model"]["detector_head"]["remove_border"]) for pb in probs]
            
            probs = torch.stack(probs) # 1,H,W
            
            pred = torch.ge(probs,self.config["model"]["detector_head"]["det_thresh"]).to(torch.int32) # 1,H,W

            pred = torch.nonzero(pred.squeeze(0), as_tuple=False) # N,2

            pred = pred.cpu().numpy()

            np.save(save_path, pred)


class ExportNeRFDetections():
    def __init__(self, config, model, dataloader, split, device):
        
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.split = split
        self.device = device
        self.output_dir = self._init_output_dir()
        self.export_NeRF()

    def _init_output_dir(self):
        """
        Where to save the outputs.
        """
        output_dir = Path(EXPER_PATH, 'outputs', self.config['data']['experiment_name'], self.split)
        if not output_dir.exists():
            os.makedirs(output_dir)
        return output_dir
    
    @torch.no_grad()
    def step(self,
             warped_image,
             probs,
             counts,
             input_rotation,
             input_translation,
             warped_rotation,
             warped_translation,
             warped_depth,
             input_intrinsics,
             warped_intrinsic):

        warped_output = self.model(warped_image)["detector_output"]["prob_heatmap"]# 1,H,W
        
        warped_output_nms = [box_nms(prob=pb,
                                     size=self.config["model"]["detector_head"]["nms"],
                                     min_prob=self.config["model"]["detector_head"]["det_thresh"],
                                     keep_top_k=self.config["model"]["detector_head"]["top_k"],
                                     remove_bord=self.config["model"]["detector_head"]["remove_border"]) for pb in warped_output]
        
        warped_output_nms = torch.stack(warped_output_nms) # 1,H,W
        
        warped_pts = torch.ge(warped_output_nms,self.config["model"]["detector_head"]["det_thresh"]).to(torch.int32) # 1,H,W

        warped_pts = torch.nonzero(warped_pts.squeeze(0), as_tuple=False) # N,2
        
        binary_heatmap = torch.zeros(size=(warped_output_nms.shape),device=self.device).squeeze()

        unwarped_pts = warp_points_NeRF(warped_pts,
                                        warped_depth.unsqueeze(0),
                                        warped_intrinsic.unsqueeze(0),
                                        input_intrinsics.unsqueeze(0),
                                        warped_rotation.unsqueeze(0),
                                        warped_translation.unsqueeze(0),
                                        input_rotation.unsqueeze(0),
                                        input_translation.unsqueeze(0),
                                        self.device) # N,2
        
        warped_output = warped_output.squeeze()

        unwarped_pts = filter_points(unwarped_pts, binary_heatmap.shape,self.device)
        
        for unwarped_pt, warped_pt in zip(unwarped_pts, warped_pts):
            
            if (int(unwarped_pt[0]) <= 1 or int(unwarped_pt[1]) <= 1 or int(unwarped_pt[0]) >= binary_heatmap.shape[0]-1 or int(unwarped_pt[1]) >= binary_heatmap.shape[1]-1 or 
                int(warped_pt[0]) <= 1 or int(warped_pt[1]) <= 1 or int(warped_pt[0]) >= binary_heatmap.shape[0]-1 or int(warped_pt[1]) >= binary_heatmap.shape[1]-1):

                binary_heatmap[ int(unwarped_pt[0]),int(unwarped_pt[1])] = warped_output[int(warped_pt[0]),int(warped_pt[1])]
    
            else:
                binary_heatmap[ int(unwarped_pt[0])-1:int(unwarped_pt[0])+2, 
                                int(unwarped_pt[1])-1:int(unwarped_pt[1])+2] = warped_output[ int(warped_pt[0])-1:int(warped_pt[0])+2, 
                                                                                              int(warped_pt[1])-1:int(warped_pt[1])+2]
            
        binary_heatmap = binary_heatmap.unsqueeze(0)
        count = torch.ones_like(binary_heatmap,device=self.device) # 1,1,H,W
        
        probs = torch.concat([probs, binary_heatmap.unsqueeze(1)], dim=1)
        counts = torch.concat([counts, count.unsqueeze(1)], dim=1)

        return probs, counts
        
    
    @torch.no_grad()
    def export_NeRF(self):
        for _, data in enumerate(tqdm(self.dataloader, desc=f"Exporting NeRF Labels", colour="green")):

            data = move_to_device(data,self.device)

            for j in range(len(data["name"])):
                
                name = data["name"][j]
                save_path = Path(self.output_dir, '{}.npy'.format(name))
                if save_path.exists():
                     continue

                input_index = j
                other_index = [k for k in range(len(data["name"])) if k != input_index]
                other_index = random.sample(other_index, k=int(0.75*len(other_index)))

                input_image = data["raw"]["image"][j,...].unsqueeze(0)
                input_rotation = data["raw"]["input_rotation"][j,...]
                input_translation = data["raw"]["input_translation"][j,...]
                input_intrinsics = data["camera_intrinsic_matrix"][j,...]
                
                probs = self.model(input_image)["detector_output"]["prob_heatmap"] # 1,H,W
                
                counts = torch.ones_like(probs,device=self.device) # 1,H,W
                
                probs = probs.unsqueeze(1) # 1,1,H,W
                counts = counts.unsqueeze(1) # 1,1,H,W
                
                for k in other_index:
                    warped_image = data["raw"]["image"][k,...].unsqueeze(0)
                    warped_rotation = data["raw"]["input_rotation"][k,...]
                    warped_translation = data["raw"]["input_translation"][k,...]
                    warped_depth = data["raw"]["input_depth"][k,...]
                    warped_intrinsic = data["camera_intrinsic_matrix"][k,...]
                    
                    probs, counts = self.step(warped_image,
                                              probs,
                                              counts,
                                              input_rotation,
                                              input_translation,
                                              warped_rotation,
                                              warped_translation,
                                              warped_depth,
                                              input_intrinsics,
                                              warped_intrinsic)
                
                counts = torch.sum(counts, dim=1) # 1,H,W
                probs = torch.sum(probs, dim=1) / counts # 1,H,W
                
                probs = [box_nms(prob=pb,
                             size=self.config["model"]["detector_head"]["nms"],
                             min_prob=self.config["model"]["detector_head"]["det_thresh"],
                             keep_top_k=self.config["model"]["detector_head"]["top_k"],
                             remove_bord=self.config["model"]["detector_head"]["remove_border"]) for pb in probs]
                
                probs = torch.stack(probs) # 1,H,W
                
                pred = torch.ge(probs,self.config["model"]["detector_head"]["det_thresh"]).to(torch.int32) # 1,H,W
                
                pred = torch.nonzero(pred.squeeze(0), as_tuple=False) # N,2
                
                pred = pred.cpu().numpy()
                
                np.save(save_path, pred)