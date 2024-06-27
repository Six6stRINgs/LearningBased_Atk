# Learning-Based Attack
![image](./picture/LBA_adversarial_attack_framework.png)

## Introduction
This repository contains all the code for the paper.

Learning-Based Attack is a black box attack method that uses a model to generate adversarial examples. 
The model is trained based on the the existing adversarial examples.
It can generate sparse adversarial examples that are similar to the existing adversarial examples with fewer queries and more efficient.

The original models are already trained and saved in the ```saved_models``` folder.

## Usage
## Requirements
Run the ```requirements.txt``` to install the required packages.
```bash
pip install -r requirements.txt
```

## Begin Attack
LBA needs an original attack, thus we provide the original attack methods in the ```attack``` folder.

## Original Attack
In this paper, we use NVITA to generate the original adversarial examples.

FGSM and BIM are also supported.
### Example: NVITA
```bash
python ./exp.py 
  --dataset Electricity
  --model CNN
  --attack NVITA
  --n 1
  --epsilon 0.1
  --save_csv
  --print_info
  --plot
```

## Learning-Based Attack
When the original adversarial examples are generated, we can use the LBA to generate the adversarial examples.

### Example: LBA
```bash
python ./exp.py 
  --dataset Electricity
  --model CNN
  --attack LBA
  --epsilon 0.1
  --save_csv
  --LBA_ori_atk_load_from_csv
  --print_info
  --plot
```

Note that all the results(*.csv) of the original attack are saved in the ```results``` folder.

By setting the ```LBA_ori_atk_load_from_csv``` parameter(Default: False),
the LBA can directly use the csv file to load the original adversarial examples.
If the parameter is set to False, the LBA will generate the original adversarial examples first.

**The descriptions of all the parameters can be found in the ```exp.py``` file.**