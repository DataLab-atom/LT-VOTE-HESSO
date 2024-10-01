from __future__ import print_function

import argparse, os, shutil, time, random, math
import numpy as np
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F

from only_train_once import OTO

import losses

from datasets.cifar100 import *
from datasets.Samplers import ClassAwareSampler

from train.train_fn.base import train_base, update_score_base
from utils.accuracy import AverageMeter
from train.validate import *

from models.net import *
from losses.loss import *

from utils.config import *
from utils.plot import *
from utils.common import hms_string,adjust_learning_rate

from utils.logger import logger

from MultiBackward import MBACK

import onnx

print(onnx.__version__, " opset=", onnx.defs.onnx_opset_version())


args = parse_args()
reproducibility(args.seed)
args = dataset_argument(args)
args.logger = logger(args)

best_acc = 0 # best test accuracy
OUT_DIR = './cache'

def get_wd_params(model: nn.Module):
    no_wd_params = list()
    wd_params = list()
    for n, p in model.named_parameters():
        if '__no_wd__' in n:
            no_wd_params.append(p)
        else:
            wd_params.append(p)
        
    return wd_params, no_wd_params


def train_sample(train_loader, model, optimizer):

    model.train()
    
    # ----- RECORD LOSS AND ACC -----
    tl = []

    for step, data_tuple in enumerate(train_loader):
        
        x = data_tuple[0].cuda()
        y = data_tuple[1].cuda()
        indexs = data_tuple[2]
        o = model(x)['ce'][0]
        # fea.requires_grad = True

        # o = model.module.classifier(fea)
        loss = F.cross_entropy(o, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        tl.append(loss.item()) 

    loss_ave = torch.tensor(tl).mean().item()
    
    return loss_ave

def main():
    global best_acc

    try:
        assert args.num_max <= 50000. / args.num_class
    except AssertionError:
        args.num_max = int(50000 / args.num_class)
    
    print(f'==> Preparing imbalanced CIFAR-100')
    # N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, args.num_class, args.imb_ratio)
    trainset, testset = get_cifar100(os.path.join(args.data_dir, 'cifar100/'), args)
    N_SAMPLES_PER_CLASS = trainset.img_num_list
    args.cls_num_list = N_SAMPLES_PER_CLASS
    print(args.cls_num_list)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last= args.loss_fn == 'ncl', pin_memory=True, sampler=None)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True) 
    casampler = ClassAwareSampler(trainset)
    CS_loader = data.DataLoader(dataset=trainset, sampler = casampler, batch_size=args.batch_size, num_workers=args.workers, drop_last= args.loss_fn == 'ncl', pin_memory=True,)
    trainloader = CS_loader
    if args.cmo: # 数据增强
        cls_num_list = N_SAMPLES_PER_CLASS
        cls_weight = 1.0 / (np.array(cls_num_list))
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        labels = trainloader.dataset.targets
        samples_weight = np.array([cls_weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(labels), replacement=True)
        weighted_trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=sampler)
    else:
        weighted_trainloader = None
    
    print ("==> creating {}".format(args.network))
    
    tasks = args.tasks
    print(tasks)

    print ("==> creating {}".format(args.network))

    model = Net(args)
    model.load_state_dict(torch.load(args.out + 'out.pt'))
    for param in model.parameters():
        param.requires_grad = False
    for name in model.decoders:
        for param in model.decoders[name].parameters():
            param.requires_grad = True
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
                     nesterov=args.nesterov)
    
    dummy_input = torch.zeros(128, 3, 32, 32).cuda()#torch.zeros_like(next(iter(trainloader))[0])
    
    oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())
    oto.visualize(view=False, out_dir=OUT_DIR)
    # Compute FLOP and param for full model. 
    full_flops = oto.compute_flops(in_million=True)['total']
    full_num_params = oto.compute_num_params(in_million=True)



    print(args.isc)
    # Create HESSO optimizer
    if args.hesso:
        optimizer = oto.hesso(variant='sgd', lr=args.lr, first_momentum=args.momentum, weight_decay=args.wd,
                        target_group_sparsity=0.7,args=args,start_pruning_step = 50*99, importance_score_criteria=args.isc)

    train_criterion = {}
    for index,t in enumerate(tasks):
        train_criterion[t] = get_loss_by_name(t, N_SAMPLES_PER_CLASS,args)
    
    mback = MBACK(optimizer,args,model.encoder)
    if args.cagrad:
        mback.init_cagrad(model.encoder.layer3,model.forward_again)
    # optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args,optimizer)
        
    teacher = None
    start_time = time.time()
     
    #--------------------------------------------train------------------------------------------------------------
    test_accs = []
    args.epochs = 30
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, scheduler, args)
        if args.cuda:
            if epoch % args.update_epoch == 0:
                curr_state, label = update_score_base(trainloader,testloader, model,optimizer, N_SAMPLES_PER_CLASS, posthoc_la = args.posthoc_la, num_test = args.num_test, accept_rate = args.accept_rate, tasks = tasks,args = args)

            if args.verbose:
                if epoch == 0:
                    maps = np.zeros((args.epochs,args.num_class))
                maps = plot_score_epoch(curr_state,label, epoch, maps, args.out)
        train_loss =train_sample(trainloader,model,optimizer)
        # train_sample(args, , , ,train_criterion, epoch, weighted_trainloader,mback, teacher, tasks) 
        

        test_loss, test_acc, test_cls = valid_normal(args, testloader, model, train_criterion, epoch, N_SAMPLES_PER_CLASS,  num_class=args.num_class, mode='test Valid', tasks=tasks)
        
        for t in test_acc.keys():
            if best_acc <= test_acc[t].avg:
                best_acc = test_acc[t].avg
                many_best = test_cls[t][0]
                med_best = test_cls[t][1]
                few_best = test_cls[t][2]
            # Save models
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'state_dict': model['model'].state_dict() if args.loss_fn == 'ncl' else model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            # }, epoch + 1, args.out)
        test_accs.append(test_acc)

        #-------------------------------------------------------logger--------------------------------------------
        args.logger(f'Epoch: [{epoch+1} | {args.epochs}]', level=1)
        if args.cuda:
            args.logger(f'Max_state: {int(torch.max(curr_state))}, min_state: {int(torch.min(curr_state))}', level=2)
        args.logger(f'[Train]\tLoss:\t{train_loss:.4f}', level=2)
        #args.logger('[weights]:\t {}'.format(mback.lmb.weights[-1]), level=2) 
        args.logger(f'[test]\tTask: ['+ ' , '.join(list(test_acc.keys())) + ']', level=2)
        test_info = [f'[Test]\tLoss: [', '] \tAcc:[ ' , ' ]']
        for t in test_acc.keys():
            for index,info_name in enumerate([test_loss,test_acc]):
                test_info[index] = test_info[index] + ' {:.4f},'.format(info_name[t].avg)
        
        args.logger(''.join(test_info), level=2)
        
        #args.logger(f'[Test ]\tLoss:\t{test_loss:.4f}\tAcc:\t{test_acc:.4f}', level=2)
        
        test_info = [f'[Stats]\tMany: [','] Medium: [ ','] Few: [',']']
        for t in test_acc.keys():
            for index in range(3):
                test_info[index] = test_info[index] + ' {:.4f},'.format(test_cls[t][index])

        args.logger(''.join(test_info), level=2)
        
        #args.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
        args.logger(f'[Best ]\tAcc:\t{best_acc:.4f}\tMany:\t{100*many_best:.4f}\tMedium:\t{100*med_best:.4f}\tFew:\t{100*few_best:.4f}', level=2)
        args.logger(f'[Param]\tLR:\t{lr:.8f}', level=2)
    
    end_time = time.time()

    # Print the final results
    args.logger(f'Final performance...', level=1)
    args.logger(f'best bAcc (test):\t{best_acc:.4f}', level=2)
    args.logger(f'best statistics:\tMany:\t{many_best}\tMed:\t{med_best}\tFew:\t{few_best}', level=2)
    args.logger(f'Training Time: {hms_string(end_time - start_time)}', level=1)
    
    torch.save(model.cpu().state_dict(),args.out + 'out_smaple.pt')
    oto.construct_subnet(out_dir=f'{args.out}cache_smaple')
    # Compute FLOP and param for pruned model after oto.construct_subnet()
    pruned_flops = oto.compute_flops(in_million=True)['total']
    pruned_num_params = oto.compute_num_params(in_million=True)
    args.logger(f"FLOP  reduction (%) :  {1.0 - pruned_flops / full_flops}  \t {pruned_flops}  \t {full_flops} ", level=1)
    args.logger(f"Param reduction (%) :  {1.0 - pruned_num_params / full_num_params} \t {pruned_num_params}\t {full_num_params}", level=1)




   

if __name__ == '__main__':
    main()