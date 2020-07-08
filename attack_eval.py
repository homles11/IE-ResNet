# from utils.misc import load_checkpoint

import argparse
import torch
import numpy as np
import os
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
import torchvision.transforms as transforms
import torchvision
from attack_config import config
from training.train import eval_one_epoch,eval_one_epoch2


parser = argparse.ArgumentParser()
parser.add_argument('--resume', '--resume', default=,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default:log/last.checkpoint)')
parser.add_argument('-d', type=int, default=0, help='Which gpu to use')
parser.add_argument('-net',type=str, default='')
parser.add_argument('-wd',type=float, default=0.05)
parser.add_argument('-it',type=int, default=1)
args = parser.parse_args()

def create_test_dataset(batch_size = 128, root = './data'):
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    #  transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    #testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    return testloader

DEVICE = torch.device('cuda:{}'.format(args.d))
torch.backends.cudnn.benchmark = False

net = get_network(args,use_gpu=True)
net.to(DEVICE)

ds_val = create_test_dataset(512)

AttackMethod = config.create_attack_method_fgsm(DEVICE)
AttackMethod2 = config.create_evaluation_attack_method2(DEVICE)
AttackMethod3 = config.create_evaluation_attack_method3(DEVICE)

if os.path.isfile(args.resume):
    # load_checkpoint(args.resume, net)
    checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.d))
    # print(checkpoint)
    net.load_state_dict(checkpoint)
    print("Checkpoint has been loaded:",args.resume)
    #learning_rate=checkpoint['lr'] #############################
    # momen=checkpoint['momen']
    # weightdecay=checkpoint['wd']
    # start=checkpoint['it']
else:
    print("Sorry, the checkpoint doesnot exist.")
print('Evaluating')
# clean_acc, adv_acc = eval_one_epoch(net, ds_val, DEVICE, AttackMethod)
# print('clean acc -- {}     adv acc -- {}'.format(clean_acc, adv_acc))
# clean_acc, adv_acc = eval_one_epoch2(net, ds_val, DEVICE, AttackMethod)
# print('clean acc -- {}     adv acc -- {} in training mode'.format(clean_acc, adv_acc))
clean_acc, adv_acc = eval_one_epoch(net, ds_val, DEVICE, AttackMethod2)
print('clean acc -- {}     adv acc -- {} in eval mode'.format(clean_acc, adv_acc))