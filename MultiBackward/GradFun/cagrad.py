from scipy.optimize import minimize, Bounds, minimize_scalar
import torch
import numpy as np
import copy
import torch.nn.functional as F

def cagrad(grads, per_weight, alpha=0.5, rescale=1,init_preference='ave'):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean()+1e-8).sqrt()  # norm of the average gradient
    _, num_class = grads.shape
    if init_preference == 'ave':
        x_start = np.ones(num_class) / num_class
    else:                                 # needed modify here
        x_start = per_weight
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1-sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha*g0_norm+1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_class).dot(A).dot(b.reshape(num_class, 1)) + c * np.sqrt(x.reshape(1, num_class).dot(A).dot(x.reshape(num_class, 1))+1e-8)).sum()
    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)

    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm+1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1+alpha**2), num_class
    else:
        return g / (1 + alpha), num_class

def overwrite_grad(m, newgrad, grad_dims, num_class):
    newgrad = newgrad * num_class # to match the sum loss
    cnt = 0
    for param in m.parameters():
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = newgrad[beg: en].contiguous().view(param.data.size())
        param.grad = this_grad.data.clone()
        cnt += 1

def grad_norm(param_groups, device):
    shared_device = device  # put everything on the same device, in case of model parallelism
    norm = torch.norm(
        torch.stack([
            p.grad.norm(p=2).to(shared_device) for p in param_groups
        ]
        ),
        p=2
    )
    return norm

def cagrad_backward(epoch,class_num_list,optimizer,feat,y,loss,pareto_module,forward_again,pareto_module_params,
                  grad_dims,pareto_start_epoch=10,rho=1.0e-4,bal_ratio=0.01,perturb_radius=1.0):
    if epoch < pareto_start_epoch: 
        loss.backward(retain_graph=True)
        y_in = y.detach().cpu().numpy()
   
        device = pareto_module_params[0].device
        # calculate gradient norm here
        grad_c_norm = grad_norm(pareto_module_params, device) + 1e-12

        # Apply SAM here
        origin_params = copy.deepcopy(pareto_module_params)
        for param in pareto_module_params:
            grad_c = param.grad
            denominator = grad_c / grad_c_norm
            noise = rho * 1.0 * denominator
            param.data = param.data + noise

        output = forward_again(feat)
        loss_sam = F.cross_entropy(output['ce'][0], y, reduction = 'none')

        var_cons_sam = 0
        count_class = len(np.unique(y_in))
        for y_tmp in np.unique(y_in):
            idx = np.where(y_in == y_tmp)
            if len(idx[0]) > 1:
                loss_c = loss_sam[idx]
                var_c = torch.std(loss_c)
                var_cons_sam += var_c
            else:
                count_class -= 1
        var_cons_sam = var_cons_sam / count_class
        loss_sam_mean = loss_sam.mean() + bal_ratio * var_cons_sam
            
        optimizer.zero_grad()
        loss_sam_mean.backward(retain_graph=True)

        grads = torch.Tensor(sum(grad_dims), len(np.unique(y_in))).cuda()
        cnt = 0
        for y_tmp in np.unique(y_in):
            grads[:, cnt].fill_(0.0)
            idx = np.where(y_in == y_tmp)
            f_param_grads = torch.autograd.grad(loss_sam[idx].sum() / y.shape[0], pareto_module_params, retain_graph=True) 
            for ii in range(len(grad_dims)):
                beg = 0 if ii == 0 else sum(grad_dims[:ii])
                en = sum(grad_dims[:(ii + 1)])
                grads[beg:en, cnt].copy_(f_param_grads[ii].data.view(-1))
            cnt += 1
            del f_param_grads

        pareto_search(pareto_module, grads, grad_dims, class_num_list, np.unique(y_in))
        for ii in range(len(pareto_module_params)):
            pareto_module_params[ii].data = perturb_radius * pareto_module_params[ii].data + (1 - perturb_radius) * origin_params[ii].data.clone()

    else:
        loss.backward()
    optimizer.step()


def pareto_search(pareto_module, grads, grad_dims, per_weight, y,init_preference='ave'):

    per_weight = per_weight[y] / np.sum(per_weight[y])
    g_w, num_class = cagrad(grads, per_weight,init_preference = init_preference)
    overwrite_grad(pareto_module, g_w, grad_dims, num_class)

