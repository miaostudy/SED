import os
import sys
import argparse
import math
import time
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from utils.builder import *
from model.MLPHeader import MLPHead
from util import *
from utils.eval import *
from model.SevenCNN import CNN
from data.imbalance_cifar import *
from data.Clothing1M import *
from utils.ema import EMA
from utils.SCS import SCS
from utils.SCR import SCR

# 导入accelerate库
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings

warnings.filterwarnings('ignore')


def conf_penalty(outputs):
    outputs = outputs.clamp(min=1e-12)
    probs = torch.softmax(outputs, dim=1)
    return torch.mean(torch.sum(probs.log() * probs, dim=1))


def warmup(net, scs, scr, net_ema, ema, optimizer, trainloader, train_loss_meter, train_accuracy_meter, accelerator):
    net.train()
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='WARMUP TRAINING',
                disable=not accelerator.is_main_process)
    for it, sample in enumerate(pbar):
        curr_lr = [group['lr'] for group in optimizer.param_groups][0]

        x, _ = sample['data']
        y = sample['label']
        outputs = net(x)
        logits = outputs['logits'] if type(outputs) is dict else outputs
        # prob = outputs['prob']
        loss_ce = F.cross_entropy(logits, y, label_smoothing=0.3)
        penalty = conf_penalty(logits)
        loss = loss_ce + penalty

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if accelerator.is_main_process:
            ema.update_params(net)
            ema.apply_shadow(net_ema)

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')
        pbar.set_description(f'WARMUP TRAINING (lr={curr_lr:.3e})')


def robust_train(net, scs, scr, n_samples, net_ema, ema, optimizer, trainloader, train_loss_meter, train_accuracy_meter,
                 num_class, params, accelerator):
    net.train()
    device = accelerator.device
    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='eval training',
                disable=not accelerator.is_main_process)
    temp_logits = torch.zeros((n_samples, num_class)).to(device)
    temp_logits_ema = torch.zeros((n_samples, num_class)).to(device)
    label = torch.zeros(n_samples).to(device)
    psedu_label = torch.zeros(n_samples).to(device)

    with torch.no_grad():
        for it, sample in enumerate(pbar):
            indices = sample['index'].cpu().numpy()
            x, _ = sample['data']
            y = sample['label']
            outputs = net(x)

            outputs_ema = net_ema(x)
            logits_ema = outputs_ema['logits'] if type(outputs_ema) is dict else outputs_ema
            px = logits_ema.softmax(dim=1)
            temp_logits_ema[indices] = px
            _, pesudo = torch.max(px, dim=-1)
            psedu_label[indices] = pesudo.float()

            logits = outputs['logits'] if type(outputs) is dict else outputs
            px = logits.softmax(dim=1)
            temp_logits[indices] = px
            label[indices] = y.float()

    if accelerator.is_main_process:
        clean_idx, noise_idx = scs.forward(params, temp_logits, label)
        print(f"the length of clean subset and noisy subset are {len(clean_idx)} and {len(noise_idx)}")
        weight = scr.forward(temp_logits_ema)
    else:
        clean_idx, noise_idx = [], []
        weight = torch.zeros(n_samples).to(device)

    clean_idx = accelerate.utils.broadcast_object_list([clean_idx])[0]
    noise_idx = accelerate.utils.broadcast_object_list([noise_idx])[0]
    weight = accelerate.utils.broadcast(weight, from_process=0)

    pbar = tqdm(trainloader, ncols=150, ascii=' >', leave=False, desc='robust training',
                disable=not accelerator.is_main_process)
    for it, sample in enumerate(pbar):
        indices = sample['index'].cpu().numpy()
        x, x_s = sample['data']
        y = sample['label']
        outputs = net(x)
        outputs_s = net(x_s)
        pesudo = psedu_label[indices].long()

        logits = outputs['logits'] if type(outputs) is dict else outputs
        logits_s = outputs_s['logits'] if type(outputs_s) is dict else outputs_s
        ind_in_clean = []
        ind_in_noise = []
        for i in range(len(indices)):
            if indices[i] in clean_idx:
                ind_in_clean.append(int(i))
            else:
                ind_in_noise.append(int(i))

        if params.use_mixup and ind_in_clean:
            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            idx2 = torch.randperm(len(ind_in_clean))
            loss_clean = torch.mean(
                F.cross_entropy(logits[ind_in_clean], y[ind_in_clean], reduction="none", label_smoothing=0.3) * l +
                (1 - l) * F.cross_entropy(
                    logits[ind_in_clean][idx2], y[ind_in_clean][idx2], reduction="none", label_smoothing=0.3)
            )
        elif ind_in_clean:
            loss_clean = F.cross_entropy(logits[ind_in_clean], y[ind_in_clean], label_smoothing=0.3)
        else:
            loss_clean = torch.tensor(0.0, device=device)

        loss = loss_clean

        loss_SSL = F.cross_entropy(logits_s, pesudo, reduction="none", label_smoothing=0.3) * weight[indices]
        loss = loss + loss_SSL.mean() * params.alpha

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if accelerator.is_main_process:
            ema.update_params(net)
            ema.apply_shadow(net_ema)

        train_acc = accuracy(logits, y, topk=(1,))
        train_accuracy_meter.update(train_acc[0], x.size(0))
        train_loss_meter.update(loss.detach().cpu().item(), x.size(0))
        pbar.set_postfix_str(f'TrainAcc: {train_accuracy_meter.avg:3.2f}%; TrainLoss: {train_loss_meter.avg:3.2f}')


class ResNet(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear'):
        super().__init__()
        import torchvision
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.classfier_head = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.classfier_head, init_method='He')
        elif classifier.startswith('mlp'):
            sf = float(classifier.split('-')[1])
            self.classfier_head = MLPHead(self.feat_dim, mlp_scale_factor=sf, projection_size=num_classes,
                                          init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.classfier_head(x)
        return {'logits': logits}


def build_model(num_classes, params_init, config):
    if config.dataset.startswith('web-'):
        net = ResNet(arch="resnet50", num_classes=num_classes, pretrained=True)
    else:
        net = CNN(input_channel=3, n_outputs=num_classes)
    return net


def build_optimizer(net, params):
    if params.opt == 'adam':
        return build_adam_optimizer(net.parameters(), params.lr, params.weight_decay, amsgrad=False)
    elif params.opt == 'sgd':
        return build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet!')


def build_loader(params):
    dataset_n = params.dataset
    if dataset_n in ["cifar100nc", "cifar80no"]:
        num_classes = int(100 * (1 - params.openset_ratio))
        transform = build_transform(rescale_size=32, crop_size=32)
        dataset = build_cifar100n_dataset("./data/cifar100",
                                          CLDataTransform(transform['cifar_train'],
                                                          transform['cifar_train_strong_aug']),
                                          transform['cifar_test'], noise_type=params.noise_type,
                                          openset_ratio=params.openset_ratio, closeset_ratio=params.closeset_ratio)
        trainloader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True)
        test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    if dataset_n.startswith('web-'):
        class_ = {"web-aircraft": 100, "web-bird": 200, "web-car": 196, "web-fg": 400, "web-nat": 5000}
        num_classes = class_[dataset_n]
        transform = build_transform(rescale_size=448, crop_size=448)
        root = os.path.join('/data2/wlf/datasets/', dataset_n)
        dataset = build_webfg_dataset(root,
                                      CLDataTransform(transform['train'], transform["train_strong_aug"]),
                                      transform['test'])
        trainloader = DataLoader(dataset["train"], batch_size=params.batch_size, shuffle=True, num_workers=4,
                                 pin_memory=True)
        test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=4,
                                 pin_memory=False)

    num_samples = len(trainloader.dataset)
    return_dict = {'trainloader': trainloader, 'num_classes': num_classes, 'num_samples': num_samples,
                   'dataset': dataset_n}
    return_dict['test_loader'] = test_loader
    return return_dict


# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")  # 仍然保留，但accelerate会覆盖此设置
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=str, default='cosine:20,5e-4,100')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--warmup-lr', type=float, default=0.001)
    parser.add_argument('--warmup-gradual', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--params-init', type=str, default='none')

    parser.add_argument('--aph', type=float, default=0.95)

    parser.add_argument('--dataset', type=str, default='cifar100nc')
    parser.add_argument('--noise-type', type=str, default='symmetric')
    parser.add_argument('--closeset-ratio', type=float, default=0.2)

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--save-weights', type=bool, default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--restart-epoch', type=int, default=0)

    parser.add_argument('--use-quantile', type=bool, default=True)
    parser.add_argument('--clip-thresh', type=bool, default=True)
    parser.add_argument('--use-mixup', type=bool, default=False)
    parser.add_argument('--momentum_scs', type=float, default=0.999)
    parser.add_argument('--momentum_scr', type=float, default=0.99)

    args = parser.parse_args()

    if Accelerator().is_main_process:
        print(args)
    return args


if __name__ == '__main__':
    accelerator = Accelerator()

    config = parse_args()

    config.openset_ratio = 0.0 if config.dataset == 'cifar100nc' else 0.2

    set_seed(config.seed)

    loader_dict = build_loader(config)
    dataset_name, n_classes, n_samples = loader_dict['dataset'], loader_dict['num_classes'], loader_dict['num_samples']

    if accelerator.is_main_process:
        scs = SCS(num_classes=n_classes, momentum=config.momentum_scs)
        scr = SCR(num_classes=n_classes, momentum=config.momentum_scr)
    else:
        scs = None
        scr = None

    model = build_model(n_classes, config.params_init, config)

    if config.resume is not None:
        if accelerator.is_main_process:
            dict_s = torch.load(config.resume, map_location='cpu')
            new_state_dict = {}
            for k, v in dict_s.items():
                if k.startswith('module.'):
                    new_k = k[7:]
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
        model = accelerator.prepare(model)
        accelerator.wait_for_everyone()
    else:
        model = accelerator.prepare(model)

    optim = build_optimizer(model, config)

    model_ema = copy.deepcopy(model)
    ema = EMA(model_ema, alpha=config.aph)
    ema.apply_shadow(model_ema)

    trainloader, test_loader = loader_dict['trainloader'], loader_dict['test_loader']
    trainloader, test_loader, optim = accelerator.prepare(
        trainloader, test_loader, optim
    )

    lr_plan = build_lr_plan(config.lr, config.epochs, config.warmup_epochs, config.warmup_lr,
                            decay=config.lr_decay, warmup_gradual=config.warmup_gradual)

    best_accuracy, best_epoch = 0.0, None
    train_loss_meter = AverageMeter()
    train_accuracy_meter = AverageMeter()

    epoch = 0
    if config.restart_epoch != 0:
        epoch = config.restart_epoch
        config.restart_epoch = 0

    while epoch < config.epochs:
        train_loss_meter.reset()
        train_accuracy_meter.reset()
        adjust_lr(optim, lr_plan[epoch])

        if epoch < config.warmup_epochs:
            warmup(model, scs, scr, model_ema, ema, optim, trainloader,
                   train_loss_meter, train_accuracy_meter, accelerator)
        else:
            robust_train(model, scs, scr, n_samples, model_ema, ema, optim, trainloader,
                         train_loss_meter, train_accuracy_meter, n_classes, config, accelerator)

        # 评估
        eval_result = evaluate_cls_acc(test_loader, model, accelerator)
        test_accuracy = eval_result['accuracy']

        if accelerator.is_main_process:
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch + 1
                if config.save_weights:
                    accelerator.save(model.state_dict(),
                                     f'./result_log/{dataset_name}_{epoch}_best_model.pth')

            print(
                f'>> Epoch {epoch}: loss {train_loss_meter.avg:.2f} ,train acc {train_accuracy_meter.avg:.2f} ,test acc {test_accuracy:.2f}, best acc {best_accuracy:.2f}')

        epoch += 1
