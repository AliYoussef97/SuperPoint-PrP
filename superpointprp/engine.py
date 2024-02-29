import tyro
import yaml
import torch
from pathlib import Path
from typing import Literal
from dataclasses import dataclass 
from superpointprp.settings import CKPT_PATH
from superpointprp.utils.get_model import get_model, load_checkpoint
from superpointprp.utils.data_loaders import get_loader
from superpointprp.engine_solvers.train import train_val
from superpointprp.engine_solvers.export import ExportDetections, ExportNeRFDetections
from superpointprp.evaluations.pose_evaluation import estimate_pose_errors
from superpointprp.evaluations.hpatches_evaluation import estimate_hpatches_metrics


@dataclass
class options:
    """Training options.

    Args:
        validate_training: Validate during training.
        include_mask_loss: Apply mask or no mask during loss (Do not include bordering artifacts by applying mask).
        nerf_loss: Whether to use Descriptor NeRF loss or normal SuperPoint Descriptor loss.
        train_nerf: Whether to enable training of NeRF datasets (Multiple datasets can be used for training)
    """
    validate_training: bool = False
    include_mask_loss: bool = False
    nerf_loss: bool = False

@dataclass
class export_pseudo_labels_split:
    """Export pseudo labels on train, validation or test split.

    Args:
        enable_Homography_Adaptation: Enable homography adaptation duing export.
        split: The split to export pseudo labels on.
    """
    enable_Homography_Adaptation: bool = True
    split: Literal["training", "validation", "test"] = "training"


@dataclass
class pose_options:
    """Pose options.
    Args:
        validate_training: configuation path
    """
    shuffle: bool = False
    max_length: int = -1


@dataclass
class hpatches_options:
    """Hpatches options.
    Args:
        validate_training: configuation path
    """
    alteration: Literal["i", "v", "all"] = "all"



@tyro.conf.configure(tyro.conf.FlagConversionOff)
class main():
    """main class, script backbone.
    
    Args:
        config_path: Path to configuration.
        task: The task to be performed.
    """
    def __init__(self,
                 config_path: str,
                 task: Literal["train",
                               "export_pseudo_labels",
                               "export_NeRF_labels",
                               "pose_evaluation",
                               "hpatches_evaluation"],
                training:options,
                pseudo_labels:export_pseudo_labels_split,
                pose:pose_options,
                hpatches:hpatches_options) -> None:

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f) 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if task == "train":

            self.model = get_model(self.config["model"], device=self.device)
            
            self.dataloader = get_loader(self.config, task, 
                                         device="cpu", validate_training=training.validate_training)

            self.mask_loss = training.include_mask_loss
            self.nerf_loss = training.nerf_loss

            if self.config["pretrained"]:
                                
                pretrained_dict = torch.load(Path(CKPT_PATH, self.config["pretrained"]), map_location=self.device)
                self.model = load_checkpoint(self.model, pretrained_dict, eval=False)

                self.iteration = pretrained_dict["iteration"]
                self.optimizer_state_dict = pretrained_dict["optimizer_state_dict"]

            if self.config["continue_training"]:
                iteration = self.iteration
                optmizer_state = self.optimizer_state_dict
            else:
                iteration = 0
                optmizer_state = None

            train_val(self.config, self.model,
                      self.dataloader["train"], self.dataloader["validation"],
                      self.mask_loss, iteration,
                      optmizer_state, self.nerf_loss,
                      self.device)

        if task == "export_pseudo_labels" or task == "export_NeRF_labels":

            self.pseudo_split = pseudo_labels.split
            self.enable_Homography_Adaptation = pseudo_labels.enable_Homography_Adaptation

            self.model = get_model(self.config["model"], device=self.device)
            self.dataloader = get_loader(self.config, task, device="cpu", export_split=self.pseudo_split)

            assert self.config["pretrained"], "Use pretrained model to export pseudo labels."
                        
            pretrained_dict = torch.load(Path(CKPT_PATH, self.config["pretrained"]), map_location=self.device)
            self.model = load_checkpoint(self.model, pretrained_dict, eval=True)

            if task == "export_pseudo_labels":
                ExportDetections(self.config, self.model, self.dataloader, self.pseudo_split, self.enable_Homography_Adaptation, self.device)
            else:
                ExportNeRFDetections(self.config, self.model, self.dataloader, self.pseudo_split, self.device)

        if task == "pose_evaluation" or task == "hpatches_evaluation":
            
            self.model = get_model(self.config["model"], device=self.device)

            pretrained_dict = torch.load(Path(CKPT_PATH, self.config["pretrained"]), map_location=self.device)
            self.model = load_checkpoint(self.model, pretrained_dict, eval=True)
            
            if task == "pose_evaluation":
                if pose.shuffle:
                    self.config["data"]["shuffle"] = True
            
                if pose.max_length > -1:
                    self.config["data"]["max_length"] = pose.max_length
                
                self.dataloader = get_loader(self.config,
                                             task,
                                             device = self.device,
                                             validate_training = False)
            
                estimate_pose_errors(self.config, 
                                     self.model, 
                                     self.dataloader, 
                                     self.device)
                
            if task == "hpatches_evaluation":
                
                self.config["data"]["alteration"] = hpatches.alteration
                
                self.dataloader = get_loader(self.config,
                                             task,
                                             device = self.device,
                                             validate_training = False)
            
                estimate_hpatches_metrics(self.config, 
                                          self.model, 
                                          self.dataloader, 
                                          self.device)

if __name__ == '__main__':
    tyro.cli(main)