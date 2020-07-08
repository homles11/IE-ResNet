""" helper function

author baiyu
"""

import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

#from dataset import cifar10Train, cifar10Test

def get_network(args, num_classes=10,use_gpu=True):
    """ return given network
    """
    if args.net == 'sir_resnet18':
        from models.sirResNet import sir_resnet18
        net = sir_resnet18(wd=args.wd, it=args.it,num_classes=num_classes)
    elif args.net == 'sir_resnet34':
        from models.sirResNet import sir_resnet34
        net = sir_resnet34(wd=args.wd, it=args.it,num_classes=num_classes)
    elif args.net == 'sir_resnet50':
        from models.sirResNet import sir_resnet50
        net = sir_resnet50(wd=args.wd, it=args.it,num_classes=num_classes)
    elif args.net == 's-resnet18':
        from models.sResNet import resnet18
        net = resnet18(num_classes=num_classes)
    elif args.net == 's-resnet34':
        from models.sResNet import resnet34
        net = resnet34(num_classes=num_classes)
    elif args.net == 's-resnet50':
        from models.sResNet import resnet50
        net = resnet50(num_classes=num_classes)
    elif args.net == 's-resnet101':
        from models.sResNet import resnet101
        net = resnet101(num_classes=num_classes)   
    elif args.net == 's2-resnet18':
        from models.s2ResNet import resnet18
        net = resnet18(num_classes=num_classes)
    elif args.net == 's2-resnet34':
        from models.s2ResNet import resnet34
        net = resnet34(num_classes=num_classes)
    elif args.net == 's2-resnet50':
        from models.s2ResNet import resnet50
        net = resnet50(num_classes=num_classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu:
        if torch.cuda.device_count()>=1:
            net = nn.DataParallel(net)
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    #cifar10_training = cifar10Train(path, transform=transform_train)
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar10 test dataset
        std: std of cifar10 test dataset
        path: path to cifar10 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar10_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    #cifar10_test = cifar10Test(path, transform=transform_test)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

def get_training_dataloader_100(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
        # transforms.Normalize(mean, std)
    ])
    #cifar10_training = cifar10Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader_100(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar10 test dataset
        std: std of cifar10 test dataset
        path: path to cifar10 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar10_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize(mean, std)
    ])
    #cifar10_test = cifar10Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar10_dataset):
    """compute the mean and std of cifar10 dataset
    Args:
        cifar10_training_dataset or cifar10_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar10_dataset[i][1][:, :, 0] for i in range(len(cifar10_dataset))])
    data_g = numpy.dstack([cifar10_dataset[i][1][:, :, 1] for i in range(len(cifar10_dataset))])
    data_b = numpy.dstack([cifar10_dataset[i][1][:, :, 2] for i in range(len(cifar10_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]