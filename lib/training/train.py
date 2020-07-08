import os
import sys
# father_dir = os.path.join('/',  *os.path.realpath(__file__).split(os.path.sep)[:-2])
# #print(father_dir)
# if not father_dir in sys.path:
#     sys.path.append(father_dir)
def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

# abs_current_path = os.path.realpath('./')
# root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-3])
lib_dir = os.path.join('/home/mjli/Code/Implicit-ResNet/lib')
add_path(lib_dir)
from lib.utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
from tqdm import tqdm
from torch.autograd import Variable

def train_one_epoch(net, batch_generator, optimizer,
                    criterion, DEVICE=torch.device('cuda:0'),
                    descrip_str='Training', AttackMethod = None, oracle=None, adv_coef = 1.0):
    '''

    :param attack_freq:  Frequencies of training with adversarial examples. -1 indicates natural training
    :param AttackMethod: the attack method, None represents natural training
    :return:  None    #(clean_acc, adv_acc)
    '''
    net.train()
    pbar = tqdm(batch_generator)
    advacc = -1
    advloss = -1
    cleanacc = -1
    cleanloss = -1
    pbar.set_description(descrip_str)
    for i, (data, label) in enumerate(pbar):
        # data = data.to(DEVICE)
        data = data.cuda()
        label = label.cuda()
        # label = label.to(DEVICE)

        optimizer.zero_grad()

        pbar_dic = OrderedDict()
        TotalLoss = 0

        if AttackMethod is not None:
            if oracle is None:
                adv_inp = AttackMethod.attack(net, data, label)
            else:
                adv_inp = AttackMethod.attack(oracle, data, label)
            optimizer.zero_grad()
            net.train()
            pred = net(adv_inp)
            loss = criterion(pred, label)

            acc = torch_accuracy(pred, label, (1,))
            advacc = acc[0].item()
            advloss = loss.item()
            #TotalLoss = TotalLoss + loss * adv_coef
            (loss * adv_coef).backward()


        pred = net(data)

        loss = criterion(pred, label)
        #TotalLoss = TotalLoss + loss
        loss.backward()
        #TotalLoss.backward()
        #param = next(net.parameters())
        #grad_mean = torch.mean(param.grad)

        optimizer.step()
        acc = torch_accuracy(pred, label, (1,))
        cleanacc = acc[0].item()
        cleanloss = loss.item()
        #pbar_dic['grad'] = '{}'.format(grad_mean)
        pbar_dic['Acc'] = '{:.2f}'.format(cleanacc)
        pbar_dic['loss'] = '{:.2f}'.format(cleanloss)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(advacc)
        pbar_dic['Advloss'] = '{:.2f}'.format(advloss)
        pbar.set_postfix(pbar_dic)


def eval_one_epoch(net, batch_generator,  DEVICE=torch.device('cuda:0'), AttackMethod = None):
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()
    adv_accuracy = AvgMeter()

    pbar.set_description('Evaluating')
    for (data, label) in pbar:
        # data = data.to(DEVICE)
        # label = label.to(DEVICE)
        data = data.cuda()
        label = label.cuda()

        # with torch.no_grad():
        pred = net(data)
        acc = torch_accuracy(pred, label, (1,))
        clean_accuracy.update(acc[0].item())

        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(net, data, label)

            # with torch.no_grad():
            pred = net(adv_inp)
            acc = torch_accuracy(pred, label, (1,))
            adv_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['AdvAcc'] = '{:.2f}'.format(adv_accuracy.mean)

        pbar.set_postfix(pbar_dic)

        adv_acc = adv_accuracy.mean if AttackMethod is not None else 0
    return clean_accuracy.mean, adv_acc

def eval_one_epoch2(net, batch_generator,  DEVICE=torch.device('cuda:0'), AttackMethod = None):
    net.train()
    # pbar = tqdm(batch_generator)
    clean_accuracy = 0.0
    adv_accuracy = 0.0
    correct = 0.0
    adv_correct = 0.0
    for (images, labels) in batch_generator:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        if AttackMethod is not None:
            adv_inp = AttackMethod.attack(net, images, labels)
        outputs = net(adv_inp)
        _, preds = outputs.max(1)
        adv_correct += preds.eq(labels).sum()

    print('Test set Accuracy: {:.4f}, Adv Accuracy:{:.4f}'.format(
        correct.float() / len(batch_generator.dataset),
        adv_correct.float()  / len(batch_generator.dataset)
    ))
    return correct.float() / len(batch_generator.dataset), adv_correct.float()  / len(batch_generator.dataset)