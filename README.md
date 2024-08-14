  
## SuperPoint-PrP ##
This project is part of the codebase for [NeRF-Supervised Feature Point Detection and Description](https://arxiv.org/abs/2403.08156).
  
  The SuperPoint-PrP implementation is based on:

  - The [SuperPoint](https://arxiv.org/abs/1712.07629) paper, [Tensorflow implementation](https://github.com/rpautrat/SuperPoint) and [GlueFactory's implementation](https://github.com/cvg/glue-factory). 


  ### 1. Setup ###
  
  In order to install the requirements and setup SuperPoint-PrP and the paths, run:
 
  ```
  make install
  ```

  You will be required to provide three different paths:

    1. Data_PATH: The path to the folder which will contain the datasets.
    2. CKPT_PATH: The path where the model's checkpoints are saved.
    3. EXPER_PATH: The path of the directory where experiments are written.

The Dataset can be downloaded through the following [link](https://drive.google.com/file/d/1lwee3hKPL-4LMmexB2iwHShEKaKQZq18/view?usp=sharing).

The folder containing the datasets should be structured as follows:
```
| datasets
|   |-- NeRF
|   |  |-- images
|   |  |   |-- training
|   |  |   |-- validation
|   |  |-- camera_transforms
|   |  |   |-- training
|   |  |   |-- validation
|   |  |-- depth
|   |  |   |-- training
|   |  |   |-- validation
|   |-- HPatches
|   |   |-- i_ajustment
|   |   |   |--1.ppm
|   |   |   |--...
|   |   |   |--H_1_2
|   |   |-- ...
|   |-- ScanNet
|   |   |-- ....
```


### 2. Configurations ###

To display all available training options run:
  ```
  python engine.py -h
  ```


### 3. Training SuperPoint-PrP ###
```
python engine.py --config_path .\configs\train.yaml --task train
```

### 4. Evaluating HPatches Homography Estimation and Repeatability ###

Run the following to run the HPatches evaluation:

```
python engine.py --config_path .\configs\hpatches_eval.yaml --task hpatches_evaluation
```

In the configuration file, change the `alteration` argument as `v` to evaluate using varying viewpoint scenes only, `i` to evaluate on varying illumination scenes, or `all` to run full Hpatches evaluation.

### 5. Relative Pose Estimation Evaluation ###

Run the following to run the relative pose estimation on the ScanNet dataset:

```
python engine.py --config_path .\configs\scannet_pose.yaml --task pose_evaluation
```

For YFCC outdoor relative pose estimation evaluation, run the following:
```
python engine.py --config_path .\configs\YFCC_pose.yaml --task pose_evaluation
```

## Credits

Special thanks to Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich
 the authors of [SuperPoint](https://arxiv.org/abs/1712.07629), Rémi Pautrat for the [Tensorflow implementation](https://github.com/rpautrat/SuperPoint) and the authors of Superpoint's [GlueFactory implementation](https://github.com/cvg/glue-factory).