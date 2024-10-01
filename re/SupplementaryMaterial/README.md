# Pareto Deep Long-Tailed Recognition: A Conflict-Averse Solution
This is the simple implementation of our paper "Pareto Deep Long-Tailed Recognition: A Conflict-Averse Solution". 

## Installation

**Requirements**
* python 3.8
* PyTorch 1.4.0
* torchvision 0.5.0
* numpy 1.19.5

## Training
### CIFAR10-LT
Specify the data path ("data_root") in configs/Cifar10_200.json. Then running the following commend:
```bash
$ python3 train_cifar.py --config ./configs/Cifar10_200.json
```