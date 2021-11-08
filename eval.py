'''Test CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import foolbox as fb
import pdb

from nearest_neighbor import nnclass, create_cifar_coreset_tensor
from utils import model_picker

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
parser.add_argument('--model_name', default='googlenet', type=str, help='Model name')
parser.add_argument('--cifar_ckpt_dir', default='/vulcanscratch/psando/cifar_model_ckpts/', 
                    help='resume from checkpoint')
parser.add_argument('--cifar_dir', default='./data',
                    help='location of the cifar-10 dataset')
parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')
parser.add_argument('--batch_size', default=128, type=int, help='input batch size')
parser.add_argument('--runs', default=1, type=int, help='number of runs')
parser.add_argument('--no_progress_bar', action='store_true', help='whether to show progress bar')
parser.add_argument('--no_download_data', action='store_true', help='whether to download data')

# Adversarial attack settings
parser.add_argument('--adversarial', action='store_true', help='Whether or not to perform adversarial attack during testing')
parser.add_argument('--attack_iters', type=int, default=20, help='Number of iterations for the attack')
parser.add_argument('--epsilon', type=float, default=8., help='Epsilon (default=8) for the attack. Script will divide by 255.')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'==> Using device {device}..')

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Dataset mean and std for normalization
dm = torch.tensor([0.4914, 0.4822, 0.4465])[None, :, None, None].cuda()
ds = torch.tensor([0.2023, 0.1994, 0.2010])[None, :, None, None].cuda()

trainset = torchvision.datasets.CIFAR10(
    root=args.cifar_dir, train=True, download=(not args.no_download_data), transform=transform_test)
testset = torchvision.datasets.CIFAR10(
    root=args.cifar_dir, train=False, download=(not args.no_download_data), transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = model_picker(args.model_name)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.cifar_ckpt_dir:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.cifar_ckpt_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(args.cifar_ckpt_dir, f'{args.model_name}.pt'))
    net.load_state_dict(checkpoint['model'])

criterion = nn.CrossEntropyLoss()

def test(epoch, adversarial=False, epsilon=(8./255), attack_iters=10):
    global best_acc
    net.eval()
    test_loss = 0

    attack_success = 0
    standard_correct = 0
    coreset_correct = 0
    total = 0

    # Create coreset loader
    coreset_matrix, coreset_target = create_cifar_coreset_tensor(net, trainset)
    nnc = nnclass(coreset_matrix, coreset_target)

    if adversarial:
        # no preprocessing since data from testloader is already normalized
        # no bounds since adversarial examples are in normalized range
        fmodel = fb.models.PyTorchModel(net, bounds=(-np.inf, np.inf), preprocessing=None) 
        attack = fb.attacks.LinfPGD(abs_stepsize=(epsilon / attack_iters * 2.5), steps=attack_iters, random_start=False)
        print(f'LinfPGD Attack Parameters: epsilon={epsilon} iters={attack_iters}')
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if adversarial:
            _, inputs, success = attack(fmodel, inputs, targets, epsilons=epsilon)
            attack_success += success.sum().item()

        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)            
            standard_correct += predicted.eq(targets).sum().item()

            # Compute coreset accuracy
            embedded = net(inputs, last_layer=True)
            outputs_nn = nnc.classify(embedded)
            coreset_correct += outputs_nn.eq(targets).sum().item()

        if not args.no_progress_bar:
            from utils import progress_bar
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*standard_correct/total, standard_correct, total))

    print('\n')
    acc = 100.*standard_correct/total
    print(f'[Epoch {epoch}] Test Accuracy: {acc:.2f} %')
    coreset_acc = 100.*coreset_correct/total
    print(f'[Epoch {epoch}] Coreset Accuracy: {coreset_acc:.2f} %')
    if adversarial:
        attack_success_rate = 100.*attack_success/total
        print(f'[Epoch {epoch}] Attack Success: {attack_success_rate:.2f} %')
    return acc


acc = test(0, adversarial=args.adversarial, epsilon=(args.epsilon / 255), attack_iters=args.attack_iters)

