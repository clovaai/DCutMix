# DCutMix-PyTorch
# Copyright (c) 2023-present NAVER Cloud Corp.
# Apache-2.0

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar
from utils import AverageMeter

def build_model(model):
    if model == 'pyramidnet' or model == 'pyramidnet_110_64':
        net = PyramidNet(dataset='cifar10', alpha=64, depth=110, bottleneck=True, num_classes=num_classes)
    elif model == 'pyramidnet_200_240':
        net = PyramidNet(dataset='cifar10', alpha=240, depth=200, bottleneck=True, num_classes=num_classes)
    else:
        print('invalid network architecture %s' % args.model)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(net)
    print("number of parameters (M): ", pytorch_total_params/1e6)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    return net

def get_dirichlet_prior(size, trial, alpha_label=0.01):
    dirichlet_priors = []
    for idx in range(size):
        alpha = alpha_label * np.ones(trial)
        dirichlet_priors.append(np.random.dirichlet(alpha))
    dirichlet_priors = torch.FloatTensor(np.asarray(dirichlet_priors))

    return dirichlet_priors

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target, mode='avg'):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        if mode=='avg':
            return loss.mean()
        elif mode=='element':
            return loss
        else:
            print('no return')

def rand_bbox(x1, y1, x2, y2, lam):
    W = x2 - x1
    H = y2 - y1
    cut_rat = np.sqrt(np.maximum(lam,0))
    cut_ws = np.round((W * cut_rat))
    cut_hs = np.round((H * cut_rat))

    cut_ws = cut_ws.astype(np.int)
    cut_hs = cut_hs.astype(np.int)
    # uniform
    cxs = []
    cys = []
    cxs_b = []
    cys_b = []
    for k, (cut_w, cut_h) in enumerate(zip(cut_ws, cut_hs)):
        if W[k] >= 0 and H[k] >= 0:
            if cut_w < W[k] and cut_h < H[k]:
                cx = np.random.randint(low=0, high=W[k] - cut_w + 1, size=1)[0]
                cy = np.random.randint(low=0, high=H[k] - cut_h + 1, size=1)[0]
            else:
                cx = 0
                cy = 0

            cxs.append(cx)
            cys.append(cy)
            cxs_b.append(np.clip(cx + cut_ws[k], 0, W[k]))
            cys_b.append(np.clip(cy + cut_hs[k], 0, H[k]))

        else:
            cx = 0
            cy = 0
            cut_ws[k] = 0
            cut_hs[k] = 0

            cxs.append(cx)
            cys.append(cy)
            cxs_b.append(cx)
            cys_b.append(cy)

    cxs = np.asarray(cxs) + x1
    cys = np.asarray(cys) + y1
    cxs_b = np.asarray(cxs_b) + x1
    cys_b = np.asarray(cys_b) + y1

    bbx1 = cxs
    bby1 = cys
    bbx2 = cxs_b
    bby2 = cys_b

    return bbx1, bby1, bbx2, bby2

def joint_dirichlet(inputs, targets, phis, beta, mode='dcutmix'):
    r = np.random.rand()

    if r < beta:

        size = inputs.size()
        W = size[-2]
        H = size[-1]
        phi_transepose = np.transpose(phis)
        batch_size = inputs.size(0)

        init_bbx1 = np.zeros(shape=batch_size, dtype=np.int)
        init_bby1 = np.zeros(shape=batch_size, dtype=np.int)
        init_bbx2 = W * np.ones(shape=batch_size, dtype=np.int)
        init_bby2 = H * np.ones(shape=batch_size, dtype=np.int)

        mul_v_k = np.ones(shape=batch_size)

        input_mixed = inputs.clone()

        targets_lists = []
        targets_lists.append(targets)

        dirichlet_approx = []
        for k, phi in enumerate(phi_transepose):
            rand_index = torch.randperm(inputs.size(0))
            targets_b = targets.clone()
            targets_b = targets_b[rand_index]
            if k == 0:
                v_k = phi
                mul_v_k = (1 - v_k)
                bbx1, bby1, bbx2, bby2 = rand_bbox(init_bbx1, init_bby1, init_bbx2, init_bby2, 1 - v_k)

                area_or = (init_bby2 - init_bby1) * (init_bbx2 - init_bbx1)
                area_or = area_or.astype(np.float)
                area_rem = (bby2 - bby1) * (bbx2 - bbx1)
                area_rem = area_rem.astype(np.float)

                dirichlet_approx.append((area_or - area_rem) / (W * H))
                init_bbx1, init_bby1, init_bbx2, init_bby2 = bbx1, bby1, bbx2, bby2

                if mode == 'dcutmix':
                    for b in range(input_mixed.size(0)):
                        input_mixed[b, :, init_bbx1[b]:init_bbx2[b], init_bby1[b]:init_bby2[b]] = inputs[rand_index[b],
                                                                                                  :,
                                                                                                  init_bbx1[b]:
                                                                                                  init_bbx2[b],
                                                                                                  init_bby1[b]:
                                                                                                  init_bby2[b]]
                elif mode == 'dmixup':
                    for b in range(input_mixed.size(0)):
                        input_mixed[b, :, :, :] = phi[b] * input_mixed[b, :, :, :] \
                                                  + phi_transepose[k + 1, b] * inputs[rand_index[b], :,:,:]


            elif k < args.trial - 1:
                v_k = phi / (mul_v_k + 1e-10)
                mul_v_k *= (1 - v_k)
                bbx1, bby1, bbx2, bby2 = rand_bbox(init_bbx1, init_bby1, init_bbx2, init_bby2, 1 - v_k)

                area_or = (init_bby2 - init_bby1) * (init_bbx2 - init_bbx1)
                area_or = area_or.astype(np.float)
                area_rem = (bby2 - bby1) * (bbx2 - bbx1)
                area_rem = area_rem.astype(np.float)

                # area rem denote the regions for next images
                dirichlet_approx.append((area_or - area_rem) / (W * H))
                init_bbx1, init_bby1, init_bbx2, init_bby2 = bbx1, bby1, bbx2, bby2

                if mode == 'dcutmix':
                    for b in range(input_mixed.size(0)):
                        input_mixed[b, :, init_bbx1[b]:init_bbx2[b], init_bby1[b]:init_bby2[b]] = inputs[rand_index[b],
                                                                                                  :,
                                                                                                  init_bbx1[b]:
                                                                                                  init_bbx2[b],
                                                                                                  init_bby1[b]:
                                                                                                  init_bby2[b]]
                elif mode == 'dmixup':
                    for b in range(input_mixed.size(0)):
                        input_mixed[b, :, :, :] += phi_transepose[k + 1, b] * inputs[rand_index[b], :,:,:]

            targets_lists.append(targets_b)

        if mode == 'dcutmix':
            # add approximated prior for the last class
            dirichlet_approx.append((area_rem) / (W * H))

            dirichlet_approx_class = []
            dirichlet_approx = np.transpose(dirichlet_approx)
            for b, prior in enumerate(dirichlet_approx):
                dirichlet_approx_class_per_batch = np.zeros(shape=args.num_classes)
                for k, prob in enumerate(prior):
                    dirichlet_approx_class_per_batch[targets_lists[k][b].item()] += prob

                dirichlet_approx_class.append(dirichlet_approx_class_per_batch)

            dirichlet_approx_class = np.asarray(dirichlet_approx_class)

            return input_mixed, torch.from_numpy(np.array(dirichlet_approx_class))
        elif mode == 'dmixup':
            dirichlet_approx_class = []
            for b, prior in enumerate(phis): # distribution per priors
                dirichlet_approx_class_per_batch = np.zeros(shape=args.num_classes)
                for k, prob in enumerate(prior):
                    dirichlet_approx_class_per_batch[targets_lists[k][b].item()] += prob

                dirichlet_approx_class.append(dirichlet_approx_class_per_batch)

            dirichlet_approx_class = np.asarray(dirichlet_approx_class)

            return input_mixed, torch.from_numpy(np.array(dirichlet_approx_class))

    else:
        dirichlet_approx_class = []
        for b, target in enumerate(targets):
            dirichlet_approx_class_per_batch = np.zeros(shape=args.num_classes)
            dirichlet_approx_class_per_batch[target.item()] = 1.

            dirichlet_approx_class.append(dirichlet_approx_class_per_batch)

        dirichlet_approx_class = np.asarray(dirichlet_approx_class)

        return inputs, torch.from_numpy(np.array(dirichlet_approx_class))

def train(epoch, criterion, criterion_onehot, tot_iter, optimizer, alpha=0.01, beta=0.5):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0

    loss_meter = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if args.mode == 'dcutmix' or args.mode == 'dmixup':
            dirichlet_priors = get_dirichlet_prior(size=inputs.size(0), trial=args.trial, alpha_label=alpha)
            input_mixed, dirichlet_approx = joint_dirichlet(inputs, targets, dirichlet_priors.data.numpy(),
                                                            beta=beta,
                                                            mode=args.mode)

            dirichlet_approx = dirichlet_approx.to(device)
            input_mixed = input_mixed.to(device)

            optimizer.zero_grad()

            outputs = net(input_mixed)
            loss = criterion(outputs, dirichlet_approx)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % (train_loss / (batch_idx + 1)))

    return batch_idx + tot_iter

def test(epoch, criterion):
    global best_acc
    global best_acc5
    net.eval()
    test_loss = 0

    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
            top1.update(err1.item(), inputs.size(0))
            top5.update(err5.item(), inputs.size(0))
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Top 1: %.3f | Top 5 : %.3f'
                            % (test_loss / (batch_idx + 1), top1.avg, top5.avg))

    # Save checkpoint.
    acc = top1.avg
    acc5 = top5.avg
    if acc < best_acc:
        print('Saving.. best Top 1 error: %.3f Top 5 error %.3f' % (acc, acc5))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if args.num_classes == 100:
            dataset = 'cifar100'
        elif args.num_classes == 10:
            dataset = 'cifar10'

        if not os.path.exists('checkpoint/%s' % dataset):
            os.makedirs('checkpoint/%s' % dataset, exist_ok=True)
        torch.save(state, './checkpoint/%s/ckpt.pth' % dataset)
        best_acc = acc
        best_acc5 = acc5
    else:
        print('best Top 1 error: %.3f Top 5 error: %.3f' % (best_acc, best_acc5))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        try:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        except:
            # for pytorch >= 1.7.0
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = float(batch_size) - correct_k
        res.append(wrong_k.mul_(100.0 / float(batch_size)))

    return res

def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
    parser.add_argument('--root', required=True, type=str, help='root directory of dataset.')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--trial', default=5, type=int, help='number of mixing images')
    parser.add_argument('--num_classes', default=100, type=int, help='num_classes')
    parser.add_argument('--epoch', default=300, type=int, help='epoch')
    parser.add_argument('--model', default='pyramidnet', type=str, help='[resnet18, pyramidnet]')
    parser.add_argument('--mode', default='dcutmix', type=str, help='[dcutmix, dmixup]')
    parser.add_argument('--beta', default=1.0, type=float, help='probability of applying d-mix augmentation')
    parser.add_argument('--alpha', default=1, type=float, help='alpha')
    parser.add_argument('--wd', default=0., type=float, help='weight decay')
    parser.add_argument('--eval_ckpt_path', default=None, type=str, help='checkpoint path of the model weight for evaluation.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_option()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    best_acc = 100.  # best test accuracy
    best_acc5 = 100.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    num_classes = args.num_classes

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if num_classes == 100:
        trainset = torchvision.datasets.CIFAR100(
            root=args.root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root=args.root, train=False, download=True, transform=transform_test)
    elif num_classes == 10:
        trainset = torchvision.datasets.CIFAR10(
            root=args.root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=args.root, train=False, download=True, transform=transform_test)
    else:
        print('select one of 10 or 100')
        exit()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    # Model
    print('==> Building model..')
    net = build_model(args.model)

    criterion_soft = SoftTargetCrossEntropy()
    criterion_onehot = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                milestones=[int(args.epoch * 0.75), int(args.epoch * 0.90)],
                                                gamma=0.1)

    if args.eval_ckpt_path is not None:
        state_dict = torch.load(args.eval_ckpt_path)
        net.load_state_dict(state_dict['net'])
        test(0, criterion_onehot)
    else:
        tot_iter = 0
        for epoch in range(args.epoch):
            tot_iter = train(epoch, criterion_soft, criterion_onehot, tot_iter, optimizer, alpha=args.alpha, beta=args.beta)
            test(epoch, criterion_onehot)
            scheduler.step()






