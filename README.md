# roller-grasper-v3

This repository contains implementations for the paper titled "Design and Control of Roller Grasper V3 for In-Hand 
Manipulation".

## System requirements

* Operating system: Linux or macOS. The code has been tested on Linux Ubuntu 20.04 and macOS Monterey (12.6).

* Software dependencies: see `environment.yml`. Please refer to the following guide to set up the development 
environment.

* Hardware: a CUDA-enabled GPU is recommend for training neural network models.



## Environment Setup & Install

Create a conda environment named `RGV3` and activate it.
```
conda env create -f environment.yml
conda activate RGV3
```
The simulation environment is based on MuJoCo physics engine and mujoco-py python wrapper. 
You can follow the instructions [here](https://github.com/openai/mujoco-py#install-mujoco) to get MuJoCo ready.

Typical setup & install time is ~5min.

## Datasets and Models

We provide the datasets collected in the simulation environment 
to train neural network models that predict reachability between poses. 
The datasets can be 
downloaded [here](https://drive.google.com/drive/folders/1k25juqbasv0HoCAiyCnyMz9hO-Wvn_La?usp=sharing).

We also provide the models we have trained and evaluated in the paper, which can be 
downloaded [here](https://drive.google.com/drive/folders/1_ffIY77FMKnAXc1JVeUucR4RdRECFcax?usp=sharing).

The code assumes the datasets and models are organized as follows:
```
data
├── checkpoints
│   ├── cube
│   │   └── model
│   │       ├── 100000_actor
│   │       └── 100000_actor_optimizer
│   ├── ... // for other objects
│   └── unified
│       └── model
│           ├── 700000_actor
│           └── 700000_actor_optimizer
└── collision_dataset
    ├── cube
    │   ├── train.npy
    │   └── val.npy
    └── ... // for other objects

```

## Run Experiments

#### Train neural network models that predict reachability between poses
*   To train a model on a single object with type `<obj_type>` (e.g., `cube`):
    ```
    python collision_oracle_pc.py --obj_type <obj_type> --output_dir <output_dir>
    ```
    The models and log files will be saved in the directory `<output_dir>`.
*   To train a unified model for multiple objects:
    ```
    python collision_oracle_pc_uni.py --output_dir <output_dir>
    ```
    You can use `--obj_types <obj_types>` to specify the objects (split by `,`). 
*   To visualize the learning curves with tensorboard:
    ```
    tensorboard --logdir <output_dir>/runs_log
    ```
    The checkpoints will be saved at `<output_dir>/model`.

#### Test high-level planner

Run `rrt_pc.py`, and use `--obj_type` to specify object type and `--model_type` to specify using either a `single` 
model or the `unified` model, e.g.,
```
python rrt_pc.py --obj_type cube --model_type single --output_dir rrt_pc_cube
```
This will start search for a feasible path from a `start_frame` and an `end_frame` using the high-level planner 
proposed in the paper. By default, the search will be run with 10 different seeds, and the path with highest confidence 
will be selected.
The path will be output to the terminal as a sequence of waypoints at termination, each waypoint consisting of position 
(in xyz coordinates) and orientation (in quaternion).

You can also change the `start_frame` and `end_frame` in `rrt_pc.py`. The run time may vary from seconds to a few 
minutes, depending on the object type, start frame and end frame. In most cases it is expected to finish within a 
minute.
