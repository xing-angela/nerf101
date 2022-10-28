## NeRF
Here is the link to the original NeRF paper: [NeRF](https://arxiv.org/abs/2003.08934)

This code is taken from the [nerf-pytorch repo](https://github.com/yenchenlin/nerf-pytorch). In order to run the code, you can reference the documentation in that repo. 

## About NeRF-Pytorch
NeRF-Pytorch can take in several different types of scenes as input (refer to the different `load_{data type}.py` files in the repo). We only dealt with loading synthetic data. The synthetic data we used was created by Chandradeep in Blender.

## Batch Script Command
`python run_nerf.py --config configs/<config-file>.txt`

## Config File
### Location
```
├── configs
│    └── <config_file>.txt
```
### Content
This config file is based on the the synthetic blender data we used.
```
expname = blender_<dataset name>
basedir = ./logs
datadir = ./data/nerf-synthetic/<dataset_name>

no_batching = True
use_viewdirs = True
lrate_decay = 500

N_samples = 64
N_importance = 128
N_rand = 1024

precrop_iters = 500
precrop_frac = 0.5
```

## Dataset
NeRF-Pytorch requires as input RGB images from each camera view and their associated camera data (intrinsics and extrinsics). 
### Folder Setup
```
├── test
│    └── <image from dataset>
├── train
│    └── <image 0>
│    ...
│    └── <image n>
├── transformations_train.json
├── val
│    └── <image from dataset>
├── transformations_test.json
├── transformations_val.json
```
### transformations_train.json
(transformation matrices for images in the train folder)
```
{
  'camera_angle_x': <camera_angle>
  'frames': [ { 'file_path': <image 0 file path>, 
                'transformation_matrix': <4x4 matrix> }, 
              ...
              { 'file_path': <image n file path>, 
                'transformation_matrix': <4x4 matrix> } ]
}
```
### transformations_test.json / transformation_val.json
(transformation matrices for images in the test and val folders)
```
{
  'camera_angle_x': <camera_angle>,
  'frames': [ { 'file_path': <image file path>, 
                'transformation_matrix': <4x4 matrix> } ]
}
```

## Cylindrical/Corkscrew Path
This is the code can be found in the `load_blender.py` file. 

## Color Depth Mapping
The color depth mapping code (the `color_map.py` file) was taken from the Nerfies code found [here](https://github.com/google/nerfies). You can change the color depth mapping back to the regular depth mapping in lines 173-180 in the `run_nerf.py` file. 

## Problems Encountered
- For synthetic data, the near and far planes are hard-coded in lines 622 and 623 in the `run_nerf.py` file. The original near and far planes are set to 2 and 6 respectively. For Chandradeep’s data, the near and far planes are 0.1 and 20 respectively.
- The original code output images and a video where the camera moves in a circular path around the object. For the synthetic data, there were images (and parts of the video) that were completely white. We believe that this occurs because camera goes outside the bounds of the box, so decreasing the radius the camera movement resolves this issues.
- The code uses the Blender coordinate system rather than the OpenCV coordinate system. 






