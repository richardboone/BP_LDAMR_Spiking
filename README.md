
## Requirements
### Dependencies and Libraries
* python 3.10
* pytorch
* torchvision

### Installation
To install requirements:

```setup
pip install -r requirements.txt
```

### Datasets
MNIST 
NMNIST
FashionMNIST
DVS Gesture

## Training
### Before running
Modify the data path and network settings in the config files in the Networks folder. 

Select the index of GPU in the [main.py] (0 by default)

### Run the code
```sh
$ python main.py -config Networks/config_file.yaml
$ python main.py -config Networks/config_file.yaml -checkpoint checkpoint/ckpt.pth // load the checkpoint
```

In the config files:
The learning urle is set in the "rule" section of the config file.  The proposed rule is set as "TPA"
The end of time temporal reconstruction method for TPA is set in "tpa_filler" section
Other hyperparameters such as the decay rate and the learning rate are listed 
