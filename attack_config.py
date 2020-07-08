from easydict import EasyDict
import sys
import os
import argparse
import numpy as np
import torch

def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

# abs_current_path = os.path.realpath('./')
# root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join('/home/mjli/Code/Implicit-ResNet/lib')
add_path(lib_dir)

from training.config import TrainingConfigBase, SGDOptimizerMaker, \
    PieceWiseConstantLrSchedulerMaker, IPGDAttackMethodMaker

class AttackConfig(TrainingConfigBase):

    lib_dir = lib_dir

    num_epochs = 105
    val_interval = 5

    create_optimizer = SGDOptimizerMaker(lr =5e-2, momentum = 0.9, weight_decay = 5e-4)
    create_lr_scheduler = PieceWiseConstantLrSchedulerMaker(milestones = [75, 90, 100], gamma = 0.1)

    create_loss_function = torch.nn.CrossEntropyLoss

    create_attack_method = \
        IPGDAttackMethodMaker(eps = 8/255.0, sigma = 2/255.0, nb_iters = 3, norm = np.inf,
                              mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std = torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))

    create_evaluation_attack_method = \
        IPGDAttackMethodMaker(eps = 8/255.0, sigma = 2/255.0, nb_iters = 1, norm = np.inf,
                              mean=torch.tensor(
                                  np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std=torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))
    create_evaluation_attack_method2 = \
        IPGDAttackMethodMaker(eps = 0.031, sigma = 0.003, nb_iters = 20, norm = np.inf,
                              mean=torch.tensor(
                                  np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std=torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))
    create_evaluation_attack_method2_2 = \
        IPGDAttackMethodMaker(eps = 8/255.0, sigma = 1/255.0, nb_iters = 20, norm = np.inf,
                              mean=torch.tensor(
                                  np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std=torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))
    create_evaluation_attack_method3 = \
        IPGDAttackMethodMaker(eps = 0.031, sigma = 0.003, nb_iters = 100, norm = np.inf,
                              mean=torch.tensor(
                                  np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                              std=torch.tensor(np.array([1]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]))


config = AttackConfig()