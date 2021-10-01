# XOSight_project

**Overview of the scripts:**

- _general_testing.py_: \
A general testing script for generating the dataset
- _network testing.py_:\
Script that converts the dataset to a pytorch dataset and runs the main loop of the network
- _hrnet_bacbone.py_:\
Contains the HRNet bacbone of the network
- _mti_net.py_:\
Contains the MTI-Net segment of the network
- _network_modules.py_:\
Contains modules used throughout the network
- _full_network.py_:\
Combines the different parts of the network and adds the YOLO object detection head
- _utils.py_:\
Contains utility functions such as metrics
- _train.py_:\
Contains the NetworkTrainer class which traines the network and shows results.
- _config.py_:\
Here the configuration of the whole network is stored.
