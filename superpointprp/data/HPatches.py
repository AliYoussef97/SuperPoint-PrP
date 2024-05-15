import torch
import kornia.geometry.transform as K
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from superpointprp.settings import DATA_PATH


class HPatches(Dataset):
    def __init__(self, data_config, device="cpu") -> None:
        super(HPatches,self).__init__()

        self.config = data_config
        self.device = device
        self.samples = self._init_dataset()
    
    def _init_dataset(self):
        ignored_scenes = (
                "i_contruction",
                "i_crownnight",
                "i_dc",
                "i_pencils",
                "i_whitebuilding",
                "v_artisans",
                "v_astronautis",
                "v_talent",
            )
        data_dir = Path(DATA_PATH, self.config["name"])
        folder_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
        
        image_paths = []
        warped_image_paths = []
        homographies = []
        names = []

        for folder_dir in folder_dirs:
            if (folder_dir.stem in ignored_scenes) and self.config["ignore_scenes"]:
                continue
            if self.config["alteration"] == 'i' != folder_dir.stem[0] != 'i':
                continue
            if self.config["alteration"] == 'v' != folder_dir.stem[0] != 'v':
                continue

            num_images = 5
            file_ext = '.ppm' 

            for i in range(2, 2 + num_images):
                image_paths.append(str(Path(folder_dir, "1" + file_ext)))
                warped_image_paths.append(str(Path(folder_dir, str(i) + file_ext)))
                homographies.append(np.loadtxt(str(Path(folder_dir, "H_1_" + str(i)))))
                names.append(f"{folder_dir.stem}_{1}_{i}")
        
        files = {'image_paths': image_paths,
                 'warped_image_paths': warped_image_paths,
                 'homography': homographies,
                 'names': names} 
        
        return files


    def __len__(self):
        return len(self.samples['image_paths'])
    

    def read_image(self, image):
        image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        image = image[None]
        image = torch.as_tensor(image/255., dtype=torch.float32)
        return image  
    
    def get_resize_shape(self, H, W):
        
        side = self.config["resize_side"]
        side_size = self.config["resize"]
        aspect_ratio = W / H
        
        if isinstance(side_size, list):
            size = side_size[0], side_size[1]
            return size
        
        if side == "vert":
            size = side_size, int(side_size * aspect_ratio)
        elif side == "horz":
            size = int(side_size / aspect_ratio), side_size
        elif (side == "short") ^ (aspect_ratio < 1.0):
            size = side_size, int(side_size * aspect_ratio)
        else:
            size = int(side_size / aspect_ratio), side_size
        return size
    
    def preprocess(self, image):
        H, W = image.shape[-2:]
        size = self.get_resize_shape(H, W)
        image = K.resize(image, size, side= self.config["resize_side"], interpolation='bilinear', align_corners=None)
        scale = torch.Tensor([image.shape[-1] / W, image.shape[-2] / H]).to(torch.float32)
        T = np.diag([scale[0], scale[1], 1.0])
        return image, T

    def __getitem__(self, index):

        image = self.read_image(self.samples['image_paths'][index])
        warped_image = self.read_image(self.samples['warped_image_paths'][index])
        
        image, T0 = self.preprocess(image)
        warped_image, T1 = self.preprocess(warped_image)
        
        homography = self.samples['homography'][index].astype(np.float32)
        homography = T1 @ homography @ np.linalg.inv(T0)

        name = self.samples['names'][index]
        
        size = image.shape[-2:]

        data = {"image": image.unsqueeze(0),
                "warped_image": warped_image.unsqueeze(0),
                "H": homography.astype(np.float32),
                "name": name,
                "size": [size[0], size[1]]}
        
        return data
    
    def batch_collator(self, batch):

        batch_0 = batch[0]

        data_output = batch_0.copy()

        return data_output