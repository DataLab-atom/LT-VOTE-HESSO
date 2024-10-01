import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from models.gate_function import custom_STE
from models.gate_function import soft_gate, virtual_gate
import torchvision
from math import sqrt, floor

class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        ctx.grad_w = grad_w
        input_clone = input.clone()
        return input_clone.float()
    @staticmethod
    def backward(ctx, grad_out):
        grad_input = ctx.grad_w * grad_out
        return grad_input, None

class Flops_constraint_resnet_bb(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3, w=10, HN=False, structure=None):
        super(Flops_constraint_resnet_bb, self).__init__()

        self.p = p
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel
        self.t_flops = self.init_total_flops() # total FLOPs
        self.inc_1st = in_channel
        self.num_gates = 2
        self.weight = w
        self.HN = HN   # True
        self.structure = structure
        print(self.p)

    def init_total_flops(self):
        total_flops = 0
        for i in range(len(self.k_size)):
            kernel_ops = self.k_size[i]*(self.in_csize[i]/self.g_size[i])
            total_flops+= kernel_ops*self.out_csize[i]*self.out_size[i] + 3*self.out_csize[i]*self.out_size[i]
            # if i == 0:
            #     print(total_flops)
        #print(i)
        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        return total_flops

    def get_flops(self, input):
        c_in = self.inc_1st
        sum_flops = 0
        #print(len(self.k_size))

        #print(len(input))
        if self.num_gates==2:
            i = 0
            if self.HN:
                # self.h = self.input.register_hook(lambda grad: grad)
                arch_vector = []
                start = 0
                for i in range(len(self.structure)):
                    end = start + self.structure[i]
                    input = input.squeeze()
                    #print(input[start:end].size())
                    arch_vector.append(input[start:end])
                    start = end
                    #print(len(input[start:end]))
                length = len(arch_vector)
                # print(length)
            else:
                length = len(input)
            i = 0
            for ind in range(0,length,2):

                if self.HN:
                    c_in = arch_vector[ind].sum() # calculate how many channels remaining
                    c_out = arch_vector[ind+1].sum()

                else:
                    current_tensor = custom_STE.apply(input[ind], False)

                    next_tensor = custom_STE.apply(input[ind+1], False)

                    c_in = current_tensor.sum()
                    c_out = next_tensor.sum()
                try:
                    sum_flops+= self.k_size[3*i]  *(self.in_csize[3*i]/self.g_size[3*i]) * c_in * self.out_size[3*i] + 3* c_in *self.out_size[3*i]
                except:
                    pass
                try:
                    sum_flops+= self.k_size[3*i+1]*(c_in/self.g_size[3*i+1])*c_out*self.out_size[3*i+1] + 3*c_out*self.out_size[3*i+1]
                except:
                    pass
                try:
                    sum_flops+= self.k_size[3*i+2]*(c_out/self.g_size[3*i+2])*self.out_csize[3*i+2]*self.out_size[3*i+2] + 3*self.out_csize[3*i+2]*self.out_size[3*i+2]
                except:
                    pass

                i+=1

        return sum_flops


    def forward(self, input):

        sum_flops = self.get_flops(input)

        resource_ratio = (sum_flops / self.t_flops)
        # abs_rv = torch.clamp(resource_ratio, min=self.p) ### 现在有大的变小，之后补充小的
        # loss = torch.log((abs_rv / (self.p))+1e-8)
        if resource_ratio > self.p:
            abs_rv = torch.clamp(resource_ratio, min=self.p + 0.005)
            loss = torch.log((abs_rv / (self.p)))
        else:
            abs_rv = torch.clamp(resource_ratio, max=self.p - 0.005)
            loss = torch.log(((self.p) / abs_rv))

        # loss = torch.log(torch.abs(sum_flops/self.t_flops - (self.p))+ 1)

        # print(sum_flops)
        #print(sum_flops/self.t_flops)
        #loss = -torch.log(1-torch.abs(sum_flops/self.t_flops - (self.p)) + 1e-8)
        #loss = torch.abs(sum_flops/self.t_flops - (self.p))**2
        return self.weight*loss

class SelectionBasedRegularization(nn.Module):
    def __init__(self, args):
        super().__init__()
        if hasattr(args, "grad_mul"):
            self.grad_mul = args.grad_mul
        else:
            self.grad_mul = 1.0
        self.structure = args.structure
        self.lam = args.gl_lam
        self.block_string = args.block_string

    def forward(self, weights, masks):
        if self.block_string == 'BasicBlock':
            return self.basic_forward(weights, masks)
        elif self.block_string == 'Bottleneck':
            return self.bb_forward(weights, masks)

    def basic_forward(self, weights, masks):
        gl_list = []
        for i in range(len(self.structure)):
            
            # print(f"weights length: {len(weights)}, masks length: {len(masks)}, structure length: {len(self.structure)}")

            w_up, w_low = weights[i]
            m_out, m_in = masks[i]
            gl_loss = (w_up*(1-m_out)).pow(2).sum((1,2,3)).add(1e-8).pow(1/2.).sum() \
                      + (w_low*(1-m_in)).pow(2).sum((0,2,3)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)
        sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list)/len(gl_list), self.grad_mul)
        return sum_loss
    # w_up, w_middle, w_low = weights[i]
    #             m_up, m_middle, m_low = masks[i]
    # gl_loss = (w_up * (1 - m_up)).pow(2).sum((1, 2, 3)).add(1e-8).pow(1 / 2.).sum() \
    #           + (w_middle * (1 - m_middle)).pow(2).sum((2, 3)).add(1e-8).pow(1 / 2.).sum() \
    #           + (w_low * (1 - m_low)).pow(2).sum((0, 2, 3)).add(1e-8).pow(1 / 2.).sum()

    def bb_forward(self, weights, masks):
        gl_list = []
        for i in range(len(weights)):
            w_up, w_middle, w_low = weights[i]
            m_out,mm_in, mm_out, m_in = masks[i]
            # mm_in, mm_out,
            # print(w_up.size())
            # print(m_out.size())
            #
            # print(mm_in.size())
            # print(w_middle.size())
            # print(mm_out.size())
            #
            # print(m_in.size())
            # print(w_low.size())

            gl_loss = (w_up     * (1 - m_out)).pow(2).sum((1,2,3)).add(1e-8).pow(1/2.).sum() 
            + (w_middle * (1 - mm_out)).pow(2).sum((1, 2, 3)).add(1e-8).pow(1 / 2.).sum() 
            # gl_loss = (w_up     * (1 - m_out)).pow(2).sum((1,2,3)).add(1e-8).pow(1/2.).sum() \
            #         # + (w_middle * (1 - mm_in)).pow(2).sum((0,2,3)).add(1e-8).pow(1/2.).sum() \
            #         + (w_middle * (1 - mm_out)).pow(2).sum((1, 2, 3)).add(1e-8).pow(1 / 2.).sum() \
            #         # + (w_low    * (1 - m_in)).pow(2).sum((0,2,3)).add(1e-8).pow(1/2.).sum()
            gl_list.append(gl_loss)
        sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list)/len(gl_list), self.grad_mul)
        return sum_loss

    # def proximal(self, model, masks):
    #     if self.block_string == 'BasicBlock':
    #         return self.basic_proximal(weights, masks)
    #     elif self.block_string == 'Bottleneck':
    #         return self.bb_proximal(weights, masks)

    # def basic_proximal(self, model, masks):
    #     return 0

    # def bb_proximal(self, model, masks):
    #     soft_gate_count = 0

    #     # orignal_weights_list = []
    #     # weights_list = []
    #     for layer_id, param in enumerate(model.modules()):
    #         if isinstance(param, virtual_gate):
    #             mask_id = layer_id // 2 

    #             model.modules()[layer_id - 2].copy_(groupproximal(model.modules()[layer_id - 2], masks[mask_id][]))

    #             if soft_gate_count % 2 == 1:
    #                 orignal_weights_list.append(modules[layer_id + 1].weight)
    #             soft_gate_count += 1

    # def groupproximal(self, model, masks):

    #     w_up_t = (w_up  * (1 - m_out)).pow(2).sum((1,2,3)).add(1e-8).pow(1/2.)

    #     w_up /  w_up_t * max(0, - self.lam * (1 - m_out).sum((1,2,3,4)) /self.N  + w_up_t)


    #     # numerator: tensor([2., 2., 0., 5., 7., 3., 4., 3., 6., 5.])
    #     a = torch.randint(0, 10, (10,), dtype=torch.float32)

    #     # denominator: tensor([3., 3., 0., 4., 5., 4., 7., 8., 0., 4.])
    #     b = torch.randint(0, 10, (10,), dtype=torch.float32)

    #     # initialize output tensor with desired value
    #     c = torch.full_like(a, fill_value=float('nan'))

    #     # zero mask
    #     mask = (b != 0)

    #     # finally perform division
    #     c[mask] = a[mask] / b[mask]


class SelectionBasedRegularization_MobileNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        if hasattr(args, "grad_mul"):
            self.grad_mul = args.grad_mul
        else:
            self.grad_mul = 1.0
        self.structure = args.structure
        self.lam = args.gl_lam

    def forward(self, weights, masks):
        gl_list = []
        for i in range(len(self.structure)):
            w_up, w_middle, w_low = weights[i]
            m_up, m_middle, m_low = masks[i]

            # print(w_up.size())
            # print(m_up.size())
            #
            # print(w_middle.size())
            # print(m_middle.size())
            #
            # print(w_low.size())
            # print(m_low.size())

            gl_loss = (w_up * (1 - m_up)).pow(2).sum((1, 2, 3)).add(1e-8).pow(1 / 2.).sum() \
                      + (w_middle * (1 - m_middle)).pow(2).sum((2, 3)).add(1e-8).pow(1 / 2.).sum() \
                      + (w_low * (1 - m_low)).pow(2).sum((0, 2, 3)).add(1e-8).pow(1 / 2.).sum()
            gl_list.append(gl_loss)
        sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list) / len(gl_list), self.grad_mul)
        return sum_loss

class SelectionBasedRegularization_MobileNetV3(nn.Module):
    def __init__(self, args):
        super().__init__()
        if hasattr(args, "grad_mul"):
            self.grad_mul = args.grad_mul
        else:
            self.grad_mul = 1.0
        self.structure = args.structure
        self.lam = args.gl_lam

    def forward(self, weights, masks):
        gl_list = []
        for i in range(len(self.structure)):
            assert (len(masks[i]) == len(weights[i])), \
                "Masks and Weights must have the same size, got {mask_len:n} and {w_len:n}"\
                    .format(mask_len=len(masks[i]), w_len=len(weights[i]))
            if len(masks[i])==3:
                w_up, w_middle, w_low = weights[i]
                m_up, m_middle, m_low = masks[i]

                gl_loss = (w_up * (1 - m_up)).pow(2).sum((1, 2, 3)).add(1e-8).pow(1 / 2.).sum() \
                          + (w_middle * (1 - m_middle)).pow(2).sum((2, 3)).add(1e-8).pow(1 / 2.).sum() \
                          + (w_low * (1 - m_low)).pow(2).sum((0, 2, 3)).add(1e-8).pow(1 / 2.).sum()
                gl_list.append(gl_loss)
            elif len(masks[i])==5:
                w_up, w_middle, w_low, se_up, se_low = weights[i]
                m_up, m_middle, m_low, mse_up, mse_low = masks[i]

                # print(se_up.size())
                # print(mse_up.size())
                # print(((se_up) * (1-mse_up)).pow(2).sum((0)).size())
                # print(se_low.size())
                # print(mse_low.size())
                # print(((se_low) * (1 - mse_low)).pow(2).sum((1)).size())
                # print(w_up.size())
                # print(m_up.size())
                #
                # print(w_middle.size())
                # print(m_middle.size())
                #
                # print(w_low.size())
                # print(m_low.size())
                #
                # print((w_up * (1 - m_up)).pow(2).sum((1, 2, 3)).size())
                # print((w_middle * (1 - m_middle)).pow(2).sum((2, 3)).size())
                # print((w_low * (1 - m_low)).pow(2).sum((0, 2, 3)).size())

                gl_loss = (w_up * (1 - m_up)).pow(2).sum((1, 2, 3)).add(1e-8).pow(1 / 2.).sum() \
                          + (w_middle * (1 - m_middle)).pow(2).sum((2, 3)).add(1e-8).pow(1 / 2.).sum() \
                          + (w_low * (1 - m_low)).pow(2).sum((0, 2, 3)).add(1e-8).pow(1 / 2.).sum()
                se_loss = ((se_up) * (1-mse_up)).pow(2).sum((0)).add(1e-8).pow(1 / 2.).sum() \
                    + ((se_low) * (1-mse_low)).pow(2).sum((1)).add(1e-8).pow(1 / 2.).sum()
                gl_list.append(gl_loss)
                gl_list.append(se_loss)
        sum_loss = self.lam * custom_grad_weight.apply(sum(gl_list) / len(gl_list), self.grad_mul)
        return sum_loss