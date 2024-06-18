# underwater_depth_estimation
This project aims to investigate the ability and efficiency of the depth estimation model, DepthAnything, on underwater images.

`depthUnderwater.ipynb`: This notebook is the code base that does some initial tests on the depthEstimation dataset, processing on the Eiffel Dataset and depth estimation on this dataset. 

`Other_Dataset_Exploration/`: This folder contains the notebooks for the work done on the FleSea dataset. 

`Training/`: This folder contains the final code files for the parameter tuning done on the DepthAnything model with the Eiffel Tower dataset.

The notebooks can be run in interactive jobs on the cluster, each cell being run individually.

The training can be run in an interactive job or with a batch script utilising the command ``python run_training.py``. 

All other folders and files are auxillary as research, notes or scratch code files. 
