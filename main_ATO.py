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
from models.hypernet import HyperStructure
import losses

from datasets.cifar100 import *

from train.train_fn.base import train_base, update_score_base
from train.validate import *

from models.net import *
from losses.loss import *

from utils.config import *
from utils.plot import *
from utils.common import hms_string,adjust_learning_rate

from utils.logger import logger

from MultiBackward import MBACK
from alignment_functions import SelectionBasedRegularization,Flops_constraint_resnet_bb
import copy



args = parse_args()
args.ATO = True
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

##-----——————————————-------- hjx --————————————————————————————-------##
def one_step_net(inputs, targets, net, masks, args):

    targets = one_hot(targets, num_classes=1000, smoothing_eps=0.1)

    if args.mix_up:
        inputs, targets = mixup_func(inputs, targets)

    net.train()
    sel_loss = torch.tensor(0)
    outputs = net(inputs)

    # loss = cross_entropy_onehot_target(outputs, targets)


    # if hasattr(args, 'reg_align') and args.reg_align:
    if hasattr(net,'module'):
        weights = net.module.get_weights()
    else:
        weights = net.get_weights()

    ##### Group lasso remove --- Optional 
    with torch.no_grad():
        sel_loss = args.selection_reg(weights, masks)
    # loss = sel_loss

    # loss.backward()

    return sel_loss, outputs


def one_hot(y, num_classes, smoothing_eps=None):
    if smoothing_eps is None:
        one_hot_y = F.one_hot(y, num_classes).float()
        return one_hot_y
    else:
        one_hot_y = F.one_hot(y, num_classes).float()
        v1 = 1 - smoothing_eps + smoothing_eps / float(num_classes)
        v0 = smoothing_eps / float(num_classes)
        new_y = one_hot_y * (v1 - v0) + v0
        return new_y
    
def mixup_func(input, target, alpha=0.2):
    gamma = np.random.beta(alpha, alpha)
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    # return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)
    return input.mul_(gamma).add_(perm_input, alpha=1 - gamma), target.mul_(gamma).add_(perm_target, alpha=1 - gamma)

def cross_entropy_onehot_target(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss
##-----——————————————--------——————--————————————————————————————-------##


def train_base(args, trainloader, model, optimizer, criterion, epoch, weighted_trainloader,mback, teacher = None, tasks = None , 
               hyper_net = None, cur_maskVec = None,optimizer_hyper= None):
    
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {t:AverageMeter() for t in tasks}
    end = time.time()
    

    bar = Bar('Training', max=len(trainloader))
    if args.cmo and 3 < epoch < (args.epochs - 3):
        inverse_iter = iter(weighted_trainloader)
    
    output = {t:[] for t in tasks}
    targets = []

    batch_idx = 0

    ##-----——————————————-------- hjx加 --————————————————————————————-------##
    # alignments = AverageMeter('AlignmentLoss', ':.4e') # sel_loss
    if epoch < int((args.epochs - 5)/ 2) + 5:
        with torch.no_grad():
            hyper_net.eval()
            vector = hyper_net()  # a vector 
            return_vect = copy.deepcopy(vector)
            masks = hyper_net.vector2mask(vector)
            # print("masks",len(masks))
    else:
        print(">>>>> Using fixed mask")
        return_vect = copy.deepcopy(cur_maskVec)
        vector = cur_maskVec
        masks = hyper_net.vector2mask(vector)
        # print("masks",len(masks))
    ##-----——————————————--------——————--————————————————————————————-------##

    for batch_index,data_tuple in enumerate(trainloader):
        inputs_b = data_tuple[0]
        targets_b = data_tuple[1]
        indexs = data_tuple[2]
        targets += targets_b.tolist()

        saved_loss = {}

        # Measure data loading
        data_time.update(time.time() - end)
        
        if args.cmo and 3 < epoch < (args.epochs - 3):
            try:
                data_tuple_f = next(inverse_iter)
            except:
                inverse_iter = iter(weighted_trainloader)
                data_tuple_f = next(inverse_iter)

            inputs_f = data_tuple_f[0]
            targets_f = data_tuple_f[1]
            inputs_f = inputs_f[:len(inputs_b)]
            targets_f = targets_f[:len(targets_b)]
            inputs_f = inputs_f.cuda(non_blocking=True)
            targets_f = targets_f.cuda(non_blocking=True)
        
        targets_b = targets_b.cuda(non_blocking=True)
        
        r = np.random.rand(1)
        optimizer.zero_grad()
        
        outputs = model(inputs_b, None,(epoch >= criterion['shike'].cornerstone) if 'shike' in tasks else False)
        
        for index,t in enumerate(tasks):
            if t == 'bcl':
                centers, logits, features = outputs['bcl']
                loss = criterion['bcl'](centers, logits, features, targets_b)
            elif t== 'gml':
                query, key, k_labels, k_idx, logits = outputs[t]
                loss = criterion[t](query, targets_b, indexs.cuda(non_blocking=True), key, k_labels, k_idx, logits)
            else:
                logits = outputs[t]
                loss = criterion[t](logits, targets_b, epoch)
            if t == 'shike':
                logits = sum(logits)/3

            saved_loss[t] = loss
            output[t] += logits.max(dim=1)[1].tolist()
            losses[t].update(loss.item(), targets_b.size(0))
        
        if args.cagrad:
            mback.update_cagrad(epoch,model.shallow_outs[1],targets_b)
            
        # mback.backward(saved_loss)

        inputs_b = inputs_b.cuda(non_blocking=True)
        model.encoder.block_string = 'BasicBlock'
        sel_loss, outputs = one_step_net(inputs_b, targets_b, model.encoder, masks, args)

        mback.backward(saved_loss)
        
        ## project
        if epoch >= args.start_epoch_gl:
            if args.lmd > 0:
                lmdValue = args.lmd
            elif args.lmd == 0:
                if epoch < int((args.epochs - 5)/ 2):
                    lmdValue = 10
                else:
                    lmdValue = 1000 #10000000000

            with torch.no_grad():
                model.encoder.lr = 0
                model.encoder.block_string = 'BasicBlock'
                if args.project == 'gl':    #----------------------------#
                    if hasattr(model, 'module'):
                        model.encoder.module.project_wegit(hyper_net.transfrom_output(vector), lmdValue, model.encoder.lr)  #----------------------------#
                    else:
                        model.encoder.project_wegit(hyper_net.transfrom_output(vector), lmdValue, model.encoder.lr)
                elif args.project == 'oto':
                    model.oto(hyper_net.transfrom_output(vector))

        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot
        suffixet = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    ) + 'Loss:['
        
        for t in tasks:
            suffixet += '[{} : {:.4f}] ,'.format(t,losses[t].avg)
        
        bar.suffix  =  suffixet + ']'
        bar.next()
        batch_idx +=1
    
        # if epoch >= args.start_epoch_hyper:
        if epoch >= args.start_epoch_hyper and (epoch < int((args.epochs - 5)/ 2) + 5):
            if (batch_index + 1) % args.hyper_step == 0:
                optimizer_hyper.zero_grad()

                masks, h_loss, res_loss, hyper_outputs = one_step_hypernet(inputs_b, targets_b, model, hyper_net,
                                                                            args)
                optimizer_hyper.step()
                model.encoder.reset_gates()

    if epoch >= args.start_epoch:
        if epoch < int((args.epochs - 5)/ 2) + 5: 
            with torch.no_grad():
                    # resource_constraint.print_current_FLOPs(hyper_net.resource_output())
                hyper_net.eval()
                vector = hyper_net()
                display_structure(hyper_net.transfrom_output(vector))
        else:
            display_structure(hyper_net.transfrom_output(vector))

    bar.finish()
    
    #torch.tensor([(targets == i).sum() for i in range(args.num_class)]) 
    
    #acc_s = [get_section_acc(args.num_class,args.cls_num_list,np.array(targets),np.array(output[t])) for t in tasks]
    #mback.pla_update(acc_s)
    #mback.llm_update(np.array(targets),output,[losses[t].avg for t in tasks])
    return sum([losses[t].avg for t in tasks])/len(tasks), return_vect

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
    # model.load_state_dict(torch.load('./logs/results/cifar100/bs@N_500_ir_100/hesso_none_target_group_sparsity_0.7/cifar100/bs@N_500_ir_100/out.pt')) 
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd,
                     nesterov=args.nesterov)
    
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
    

    ##-----——————————————-------- hjx加 --————————————————————————————-------##
     # ---------------------------------- #
    width, structure = model.encoder.count_structure()
    cur_maskVec = None

    args.model_name = 'resnet'
    args.structure = structure
    args.gl_lam = 0.0001 # 补
    args.block_string = 'BasicBlock' # model.encoder.block_string
    args.start_epoch_gl = 100 #100
    args.start_epoch_hyper = 25 #20
    args.hyper_step = 1
    args.start_epoch = 0
    args.reg_w = 4.0
    args.lmd = 0
    args.p = 0.5
    args.project = 'gl'
    args.mix_up = False

    hyper_net = HyperStructure(structure=structure, T=0.4, base=3,args=args)
    hyper_net.cuda()
    
        # tmp = hyper_net()
        # print("Mask", tmp, tmp.size())
    params_group = group_weight(hyper_net)
        # print(len(list(hyper_net.parameters())))
    optimizer_hyper = torch.optim.AdamW(params_group, lr=1e-3, weight_decay=1e-2)
    scheduler_hyper = torch.optim.lr_scheduler.MultiStepLR(optimizer_hyper, milestones=[int(0.98 * ((args.epochs - 5) / 2) + 5)], gamma=0.1)

    sel_reg = SelectionBasedRegularization(args) 
    size_out, size_kernel, size_group, size_inchannel, size_outchannel,handles = get_middle_Fsize_resnetbb(model)
    args.resource_constraint = Flops_constraint_resnet_bb(args.p, size_kernel, size_out, size_group, size_inchannel,
                                                       size_outchannel, w=args.reg_w, HN=True,structure=structure)
    args.selection_reg = sel_reg
    ##-----——————————————--------——————--————————————————————————————-------##


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
                        target_group_sparsity=args.target_group_sparsity,args=args,start_pruning_step = 0, importance_score_criteria=args.isc)


    for epoch in range(args.epochs):
        optimizer.start_pruning_step = 99999 if epoch < 100 else 0
        lr = adjust_learning_rate(optimizer, epoch, scheduler, args)
        if args.cuda:
            if epoch % args.update_epoch == 0:
                curr_state, label = update_score_base(trainloader,testloader, model,optimizer, N_SAMPLES_PER_CLASS, posthoc_la = args.posthoc_la, num_test = args.num_test, accept_rate = args.accept_rate, tasks = tasks,args = args)

            if args.verbose:
                if epoch == 0:
                    maps = np.zeros((args.epochs,args.num_class))
                maps = plot_score_epoch(curr_state,label, epoch, maps, args.out)
        
        ##-----——————————————-------- hjx加 --————————————————————————————-------##
        train_loss, cur_maskVec = train_base(args, trainloader, model, optimizer,train_criterion, epoch, weighted_trainloader,mback, 
                                             teacher, tasks, hyper_net, cur_maskVec,optimizer_hyper) 
        
        scheduler_hyper.step()
        # test_loss, test_acc, test_cls = valid_normal(args, testloader, model, train_criterion, epoch, N_SAMPLES_PER_CLASS,  num_class=args.num_class, mode='test Valid', tasks=tasks)
        ##-----——————————————--------——————--————————————————————————————-------##

        # train_loss = train_base(args, trainloader, model, optimizer,train_criterion, epoch, weighted_trainloader,mback, teacher, tasks) 
        

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
        if args.save_all_epoch:
            import copy
            torch.save(copy.deepcopy(model).cpu().state_dict(),args.out + f'out{epoch}.pt')
    end_time = time.time()

    
    # Print the final results
    args.logger(f'Final performance...', level=1)
    args.logger(f'best bAcc (test):\t{best_acc:.4f}', level=2)
    args.logger(f'best statistics:\tMany:\t{many_best}\tMed:\t{med_best}\tFew:\t{few_best}', level=2)
    args.logger(f'Training Time: {hms_string(end_time - start_time)}', level=1)
    hyper_net.eval()
    total_flops =  args.resource_constraint.get_flops(hyper_net.resource_output())
    print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
    
    oto.construct_subnet(out_dir=f'{args.out}cache')
    # Compute FLOP and param for pruned model after oto.construct_subnet()
    pruned_flops = oto.compute_flops(in_million=True)['total']
    pruned_num_params = oto.compute_num_params(in_million=True)
    for handle in handles:
        handle.remove()

    torch.save(model.cpu().state_dict(),args.out + 'out.pt')
    np.save(args.out + 'w_acc.npy',model.W_acc.weigh_save_list)
    args.logger(f"FLOP  reduction (%) :  {1.0 - pruned_flops / full_flops}  \t {pruned_flops}  \t {full_flops} ", level=1)
    args.logger(f"Param reduction (%) :  {1.0 - pruned_num_params / full_num_params} \t {pruned_num_params}\t {full_num_params}", level=1)

##-----——————————————-------- hjx加 --————————————————————————————-------##
def group_weight(module, wegith_norm=True):
    group_decay = []
    group_no_decay = []
    #group_no_decay.append(module.inputs)
    if hasattr(module, 'inputs'):
        group_no_decay.append(module.inputs)
    count = 0
    #if isinstance(module, torch.nn.DataParallel):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

        elif isinstance(m, nn.Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GRU):

            for k in range(m.num_layers):
                # getattr(m,'weight_ih_l%d'%(k))
                # getattr(m, 'weight_hh_l%d' % (k))
                group_decay.append(getattr(m,'weight_ih_l%d'%(k)))
                group_decay.append(getattr(m, 'weight_hh_l%d' % (k)))

                if getattr(m, 'bias_hh_l%d' % (k)) is not None:
                    group_no_decay.append(getattr(m, 'bias_hh_l%d' % (k)))
                    group_no_decay.append(getattr(m, 'bias_ih_l%d' % (k)))

                if m.bidirectional:
                    group_decay.append(getattr(m, 'weight_ih_l%d_reverse' % (k)))
                    group_decay.append(getattr(m, 'weight_hh_l%d_reverse' % (k)))

                    if getattr(m, 'bias_hh_l%d_reverse' % (k)) is not None:
                        group_no_decay.append(getattr(m, 'bias_hh_l%d_reverse' % (k)))
                        group_no_decay.append(getattr(m, 'bias_ih_l%d_reverse' % (k)))

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    print(len(list(module.parameters())))
    print(len(group_decay))
    print(len(group_no_decay))
    print(count)
    #print(module)
    assert len(list(module.parameters()))-count == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def one_step_hypernet(inputs, targets, net, hyper_net, args):
    net.eval()
    hyper_net.train()

    vector = hyper_net() # 有分数
    net.encoder.set_vritual_gate(vector)
    outputs = net(inputs)[args.tasks[0]][0]

    res_loss = 2 * args.resource_constraint(hyper_net.resource_output())
    loss = nn.CrossEntropyLoss()(outputs, targets) + res_loss
    loss.backward()

    with torch.no_grad():
        hyper_net.eval()
        vector = hyper_net()
        masks = hyper_net.vector2mask(vector)

    return masks, loss, res_loss, outputs

def display_structure(all_parameters, p_flag=False):
    num_layers = len(all_parameters)
    layer_sparsity = []
    for i in range(num_layers):
        if i == 0 and p_flag is True:
            current_parameter = all_parameters[i].cpu().data
            print(current_parameter)
            #print(all_parameters[i].grad.cpu().data)
        current_parameter = all_parameters[i].cpu().data
        layer_sparsity.append((current_parameter>=0.5).sum().float().item()/current_parameter.size(0))

    print_string = ''
    for i in range(num_layers):
        print_string += 'l-%d s-%.3f '%(i+1, layer_sparsity[i])

    print(print_string)

from models.gate_function import virtual_gate

def get_middle_Fsize_resnetbb(model, input_res=32, num_gates=1):
    size_out = []
    size_kernel = []
    size_group = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        if hasattr(self, 'kernel_size'):
            batch_size, input_channels, input_height, input_width = input[0].size()
            output_channels, output_height, output_width = output[0].size()
            size_out.append(output_height * output_width)
            size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
            size_group.append(self.groups)
            size_inchannel.append(input_channels)
            size_outchannel.append(output_channels)

    def foo(net):
        handles = []
        modules = list(net.modules())
        #print(modules)
        soft_gate_count=0
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            #print(m)
            if layer_id + 3 >= len(modules):
                continue
            if isinstance(m, virtual_gate):
                if num_gates==2:
                    handles.append(modules[layer_id - 2].register_forward_hook(conv_hook))
                    if soft_gate_count%2 == 1:
                        handles.append(modules[layer_id + 1].register_forward_hook(conv_hook)) # ???
                    soft_gate_count += 1
                else:
                    handles.append(modules[layer_id - 4].register_forward_hook(conv_hook)) 
                    handles.append(modules[layer_id - 2].register_forward_hook(conv_hook))
                    handles.append(modules[layer_id + 1].register_forward_hook(conv_hook))
        return handles


    handles = foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(size_out))
    print(len(size_kernel))

    return size_out, size_kernel, size_group, size_inchannel, size_outchannel,handles
##-----——————————————--------——————--————————————————————————————-------##


if __name__ == '__main__':
    main()