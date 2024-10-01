import torch
import torch.nn as nn
from .gate_function import soft_gate, custom_STE
from .gate_function import virtual_gate
import copy

__all__ = ['ResNet', 'resnet18', 'resnet34', 'my_resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cfg=None, num_gate=1):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if num_gate==1:
            self.gate = virtual_gate(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.num_gate = num_gate

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.num_gate == 1:
            out = self.gate(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cfg=None, num_gate=2):
        super(Bottleneck, self).__init__()
        if cfg is None:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            if num_gate>1:
                self.gate1 = virtual_gate(width)

            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)

            if num_gate>=1:
                self.gate2 = virtual_gate(width)

            #self.gate2 = self.gate1

            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            #width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, cfg[0])
            self.bn1 = norm_layer(cfg[0])
            #self.gate1 = soft_gate(cfg[0])
            if num_gate>1:
                self.gate1 = virtual_gate(cfg[0])
            else:
                self.gate1 = None
            self.conv2 = conv3x3(cfg[0], cfg[1], stride, groups, dilation)
            self.bn2 = norm_layer(cfg[1])
            if num_gate>=1:

                self.gate2 = virtual_gate(cfg[1])

            #self.gate2 = self.gate1


            self.conv3 = conv1x1(cfg[1], planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)

            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.gate1(out)
        if self.gate1 is not None:
            out = self.gate1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.gate2 is not None:
            out = self.gate2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cfg=None, num_gate=1):
        super(ResNet, self).__init__()
        self.lmd = 0
        self.lr = 0 


        if block is Bottleneck:
            print('Bottleneck')
            self.factor = 2
            self.block_string = 'Bottleneck'
        elif block is BasicBlock:
            print('BasicBlock')
            self.factor = 1
            self.block_string = 'BasicBlock'

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.safe_guard = 1e-8

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_gate = num_gate

        print(self.factor)
        if cfg == None:

            self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 16, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 32, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 64, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        else:

            start = 0
            end = int(self.factor*layers[0])
            self.layer1 = self._make_layer(block, 16, layers[0], cfg=cfg[start:end])
            start = end
            end = end+int(self.factor*layers[1])
            self.layer2 = self._make_layer(block, 16, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0], cfg=cfg[start:end])
            start = end
            end = end+int(self.factor*layers[2])
            self.layer3 = self._make_layer(block, 32, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1], cfg=cfg[start:end])
            start = end
            end = end+int(self.factor*layers[3])
            self.layer4 = self._make_layer(block, 64, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2], cfg=cfg[start:end])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cfg=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        if cfg is None:
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, num_gate=self.num_gate))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, num_gate=self.num_gate))
        else:

            index = 0
            layers = []

            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, cfg=cfg[int(self.factor*index):int(self.factor*index+self.factor)], num_gate=self.num_gate))
            index+=1
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, cfg=cfg[int(self.factor*index):int(self.factor*index+self.factor)], num_gate=self.num_gate))
                index+=1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width) # width
        self.structure = structure
        print("structure", structure, len(structure)) # structure [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512] 32

        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.set_structure_value(arch_vector.squeeze()[start:end])
                start = end

                i+=1

    def reset_gates(self):
        for m in self.modules():
            if isinstance(m, virtual_gate):
                m.reset_value()
# ResNet50
# layer1.0.conv1
# layer1.0.bn1
# layer1.0.conv2
# layer1.0.bn2
# layer1.0.conv3
# layer1.0.bn3
# layer1.0.relu
# layer1.0.downsample
# layer1.0.downsample.0
# layer1.0.downsample.1

# ResNet18
# layer1.0.conv1
# layer1.0.bn1
# layer1.0.relu
# layer1.0.conv2
# layer1.0.bn2
    
    
    # ---------------------------------剪枝（修改模型的参数-也就是权重）----------------------------------------------  
    def project_wegit(self, masks, lmd, lr):
        self.lmd, self.lr = lmd, lr
        # print("self.lam * ratio * self.lr", self.lmd, self.lr)

        N_t = 0
        for itm in masks:
            N_t += (1 - itm).sum()
        gap = 2 if self.block_string == 'Bottleneck' else 3 # 
        modules = list(self.modules())
        weights_list = []
        vg_idx = 0

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if isinstance(m, virtual_gate):
                # print("Qian", masks[vg_idx])
                # masks[vg_idx][0] = 0
                # print("Qian", masks[vg_idx])

                ratio = (1 - masks[vg_idx]).sum() / N_t
                if ratio == 0:
                    vg_idx += 1
                    continue

                # print("Starting project")
                m_out = (masks[vg_idx] == 0)
                vg_idx += 1
                ## calculate group norm
                w_norm = (modules[layer_id - gap].weight.data[m_out]).pow(2).sum((1,2,3))
                # print("w_norm shape", w_norm.size(), modules[layer_id - gap].weight.data[m_out].size())
                w_norm += (modules[layer_id - gap + 1].weight.data[m_out]).pow(2) #.sum((1,2,3))
                # print("w_norm shape", w_norm.size(), modules[layer_id - gap + 1].weight.data[m_out].size())
                w_norm += (modules[layer_id - gap + 1].bias.data[m_out]).pow(2)
                # print("w_norm shape", w_norm.size(), modules[layer_id - gap + 1].bias.data[m_out].size())
                w_norm = w_norm.add(1e-8).pow(1/2.)
                # print("w_norm shape", w_norm.size())

                modules[layer_id - gap].weight.copy_(self.groupproximal(modules[layer_id - gap].weight.data, m_out, ratio, w_norm))
                modules[layer_id - gap + 1].weight.copy_(self.groupproximal(modules[layer_id - gap + 1].weight.data, m_out, ratio, w_norm))
                modules[layer_id - gap + 1].bias.copy_(self.groupproximal(modules[layer_id - gap + 1].bias.data, m_out, ratio, w_norm))

    def groupproximal(self, weight, m_out, ratio, w_norm):
        # #######  Test ######
        # weight[m_out] = 0
        # return weight
        ####################

        # print(weight.size(), weight[m_out].size())
        dimlen = len(weight.size())
        while dimlen > 1:
            w_norm = w_norm.unsqueeze(1)
            dimlen -= 1

        weight[m_out] = weight[m_out] / w_norm 
        tmp = - self.lmd * ratio * self.lr + w_norm
        tmp[tmp < 0] = 0 # tmp = max(0, - self.lmd * ratio * self.lr + w_norm)

        weight[m_out] = weight[m_out] * tmp
        return weight

    def getxs(self):
        self.xs = []
        gap = 2 if self.block_string == 'Bottleneck' else 3
        modules = list(self.modules())

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if isinstance(m, virtual_gate):
                channel_num = len(modules[layer_id - gap].weight.data)
                self.xs.append(copy.deepcopy(modules[layer_id - gap].weight.data.view(channel_num, -1)))
                self.xs.append(copy.deepcopy(modules[layer_id - gap + 1].weight.data.view(channel_num, -1)))
                self.xs.append(copy.deepcopy(modules[layer_id - gap + 1].bias.data.view(channel_num, -1)))


    def half_space_project(self, hat_x, x, epsilon, upper_group_sparsity = 1):
        num_groups = x.shape[0]
        x_norm = torch.norm(x, p=2, dim=1)
        before_group_sparsity = torch.sum(x_norm == 0) / float(num_groups)
        if before_group_sparsity < upper_group_sparsity:
            proj_idx = (torch.bmm(hat_x.view(hat_x.shape[0], 1, -1), x.view(x.shape[0], -1, 1)).squeeze() \
                < epsilon * x_norm ** 2)    
            
            trial_group_sparsity = torch.sum(torch.logical_or(proj_idx, x_norm == 0)) / float(num_groups) # element-wise logical OR 
            # if trial group sparsity larger than upper group sparsity, then control the size of half-space projection
            if trial_group_sparsity > upper_group_sparsity:
                max_num_proj_groups = int(num_groups * (trial_group_sparsity - upper_group_sparsity))
                max_num_proj_groups = min(max(0, max_num_proj_groups), num_groups - 1)
                proj_group_idxes = torch.arange(num_groups)[proj_idx == True] #
                refined_proj_idxes = torch.randperm(torch.sum(proj_idx))[:max_num_proj_groups].sort()[0]
                hat_x[proj_group_idxes[refined_proj_idxes], ...] = 0.0
            else:
                hat_x[proj_idx, ...] = 0.0
        return hat_x

    def get_weights(self):
        if self.block_string == 'BasicBlock':
            return self.get_weights_basicblock()
        elif self.block_string == 'Bottleneck':
            return self.get_weights_bottleneck()

    def get_weights_basicblock(self):
        modules = list(self.modules())
        weights_list = []

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            current_list = []
            if isinstance(m, virtual_gate):
                up_weight = modules[layer_id - 3].weight
                low_weight = modules[layer_id + 1].weight
                current_list.append(up_weight), current_list.append(low_weight)
                weights_list.append(current_list)

        return weights_list

    def get_weights_bottleneck(self):
        modules = list(self.modules())
        orignal_weights_list = []
        weights_list = []
        soft_gate_count = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            # print(m)
            # if layer_id + 3 <= len(modules):
            if isinstance(m, virtual_gate):
                # print(m)
                # modules[layer_id - 4].register_forward_hook(conv_hook)
                # modules[layer_id - 2].register_forward_hook(conv_hook)
                # modules[layer_id + 1].register_forward_hook(conv_hook)

                orignal_weights_list.append(modules[layer_id - 2].weight)
                if soft_gate_count % 2 == 1:
                    orignal_weights_list.append(modules[layer_id + 1].weight)
                soft_gate_count += 1

                # up_weight = modules[layer_id - 4].weight
                # middle_weight = modules[layer_id - 2].weight
                # low_weight = modules[layer_id + 1].weight
        length = len(orignal_weights_list)
        for i in range(0,length,3):
            current_list = []

            current_list.append(orignal_weights_list[i])
            current_list.append(orignal_weights_list[i+1])
            current_list.append(orignal_weights_list[i+2])

            weights_list.append(current_list)

        return weights_list

class ResNet_rep(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cfg=None, num_gate=1):
        super(ResNet_rep, self).__init__()
        self.lmd = 0
        self.lr = 0 

        if block is Bottleneck:
            print('Bottleneck')
            self.factor = 2
            self.block_string = 'Bottleneck'
        elif block is BasicBlock:
            print('BasicBlock')
            self.factor = 1
            self.block_string = 'BasicBlock'

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.safe_guard = 1e-8

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_gate = num_gate

        print(self.factor)
        if cfg == None:

            self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 32, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 64, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        else:

            start = 0
            end = int(self.factor*layers[0])
            self.layer1 = self._make_layer(block, 16, layers[0], cfg=cfg[start:end])
            start = end
            end = end+int(self.factor*layers[1])
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0], cfg=cfg[start:end])
            start = end
            end = end+int(self.factor*layers[2])
            self.layer3 = self._make_layer(block, 32, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1], cfg=cfg[start:end])
            start = end
            end = end+int(self.factor*layers[3])
            self.layer4 = self._make_layer(block, 64, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2], cfg=cfg[start:end])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cfg=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        if cfg is None:
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, num_gate=self.num_gate))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, num_gate=self.num_gate))
        else:

            index = 0
            layers = []

            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, cfg=cfg[int(self.factor*index):int(self.factor*index+self.factor)], num_gate=self.num_gate))
            index+=1
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, cfg=cfg[int(self.factor*index):int(self.factor*index+self.factor)], num_gate=self.num_gate))
                index+=1

        return nn.Sequential(*layers)

    def forward(self, x):
        shallow_outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        shallow_outs.append(x)
        x = self.layer1(x)
        shallow_outs.append(x)
        x = self.layer2(x)
        shallow_outs.append(x)
        x = self.layer3(x)
        shallow_outs.append(x)
        x = self.layer4(x)
        shallow_outs.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x,shallow_outs

    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width) # width
        self.structure = structure
        print("structure", structure, len(structure)) # structure [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512] 32

        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.set_structure_value(arch_vector.squeeze()[start:end])
                start = end

                i+=1

    def reset_gates(self):
        for m in self.modules():
            if isinstance(m, virtual_gate):
                m.reset_value()

    # ---------------------------------剪枝（修改模型的参数-也就是权重）----------------------------------------------  
    def project_wegit(self, masks, lmd, lr):
        self.lmd, self.lr = lmd, lr
        # print("self.lam * ratio * self.lr", self.lmd, self.lr)

        N_t = 0
        for itm in masks:
            N_t += (1 - itm).sum()
        gap = 2 if self.block_string == 'Bottleneck' else 3 # 
        modules = list(self.modules())
        weights_list = []
        vg_idx = 0

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if isinstance(m, virtual_gate):
                # print("Qian", masks[vg_idx])
                # masks[vg_idx][0] = 0
                # print("Qian", masks[vg_idx])

                ratio = (1 - masks[vg_idx]).sum() / N_t
                if ratio == 0:
                    vg_idx += 1
                    continue

                # print("Starting project")
                m_out = (masks[vg_idx] == 0)
                vg_idx += 1
                ## calculate group norm
                w_norm = (modules[layer_id - gap].weight.data[m_out]).pow(2).sum((1,2,3))
                # print("w_norm shape", w_norm.size(), modules[layer_id - gap].weight.data[m_out].size())
                w_norm += (modules[layer_id - gap + 1].weight.data[m_out]).pow(2) #.sum((1,2,3))
                # print("w_norm shape", w_norm.size(), modules[layer_id - gap + 1].weight.data[m_out].size())
                w_norm += (modules[layer_id - gap + 1].bias.data[m_out]).pow(2)
                # print("w_norm shape", w_norm.size(), modules[layer_id - gap + 1].bias.data[m_out].size())
                w_norm = w_norm.add(1e-8).pow(1/2.)
                # print("w_norm shape", w_norm.size())

                modules[layer_id - gap].weight.copy_(self.groupproximal(modules[layer_id - gap].weight.data, m_out, ratio, w_norm))
                modules[layer_id - gap + 1].weight.copy_(self.groupproximal(modules[layer_id - gap + 1].weight.data, m_out, ratio, w_norm))
                modules[layer_id - gap + 1].bias.copy_(self.groupproximal(modules[layer_id - gap + 1].bias.data, m_out, ratio, w_norm))

    def groupproximal(self, weight, m_out, ratio, w_norm):
        # #######  Test ######
        # weight[m_out] = 0
        # return weight
        ####################

        # print(weight.size(), weight[m_out].size())
        dimlen = len(weight.size())
        while dimlen > 1:
            w_norm = w_norm.unsqueeze(1)
            dimlen -= 1

        weight[m_out] = weight[m_out] / w_norm 
        tmp = - self.lmd * ratio * self.lr + w_norm
        tmp[tmp < 0] = 0 # tmp = max(0, - self.lmd * ratio * self.lr + w_norm)

        weight[m_out] = weight[m_out] * tmp
        return weight

    def getxs(self):
        self.xs = []
        gap = 2 if self.block_string == 'Bottleneck' else 3
        modules = list(self.modules())

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if isinstance(m, virtual_gate):
                channel_num = len(modules[layer_id - gap].weight.data)
                self.xs.append(copy.deepcopy(modules[layer_id - gap].weight.data.view(channel_num, -1)))
                self.xs.append(copy.deepcopy(modules[layer_id - gap + 1].weight.data.view(channel_num, -1)))
                self.xs.append(copy.deepcopy(modules[layer_id - gap + 1].bias.data.view(channel_num, -1)))


    def half_space_project(self, hat_x, x, epsilon, upper_group_sparsity = 1):
        num_groups = x.shape[0]
        x_norm = torch.norm(x, p=2, dim=1)
        before_group_sparsity = torch.sum(x_norm == 0) / float(num_groups)
        if before_group_sparsity < upper_group_sparsity:
            proj_idx = (torch.bmm(hat_x.view(hat_x.shape[0], 1, -1), x.view(x.shape[0], -1, 1)).squeeze() \
                < epsilon * x_norm ** 2)    
            
            trial_group_sparsity = torch.sum(torch.logical_or(proj_idx, x_norm == 0)) / float(num_groups) # element-wise logical OR 
            # if trial group sparsity larger than upper group sparsity, then control the size of half-space projection
            if trial_group_sparsity > upper_group_sparsity:
                max_num_proj_groups = int(num_groups * (trial_group_sparsity - upper_group_sparsity))
                max_num_proj_groups = min(max(0, max_num_proj_groups), num_groups - 1)
                proj_group_idxes = torch.arange(num_groups)[proj_idx == True] #
                refined_proj_idxes = torch.randperm(torch.sum(proj_idx))[:max_num_proj_groups].sort()[0]
                hat_x[proj_group_idxes[refined_proj_idxes], ...] = 0.0
            else:
                hat_x[proj_idx, ...] = 0.0
        return hat_x

    def get_weights(self):
        if self.block_string == 'BasicBlock':
            return self.get_weights_basicblock()
        elif self.block_string == 'Bottleneck':
            return self.get_weights_bottleneck()

    def get_weights_basicblock(self):
        modules = list(self.modules())
        weights_list = []

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            current_list = []
            if isinstance(m, virtual_gate):
                up_weight = modules[layer_id - 3].weight
                low_weight = modules[layer_id + 1].weight
                current_list.append(up_weight), current_list.append(low_weight)
                weights_list.append(current_list)

        return weights_list

    def get_weights_bottleneck(self):
        modules = list(self.modules())
        orignal_weights_list = []
        weights_list = []
        soft_gate_count = 0
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            # print(m)
            # if layer_id + 3 <= len(modules):
            if isinstance(m, virtual_gate):
                # print(m)
                # modules[layer_id - 4].register_forward_hook(conv_hook)
                # modules[layer_id - 2].register_forward_hook(conv_hook)
                # modules[layer_id + 1].register_forward_hook(conv_hook)

                orignal_weights_list.append(modules[layer_id - 2].weight)
                if soft_gate_count % 2 == 1:
                    orignal_weights_list.append(modules[layer_id + 1].weight)
                soft_gate_count += 1

                # up_weight = modules[layer_id - 4].weight
                # middle_weight = modules[layer_id - 2].weight
                # low_weight = modules[layer_id + 1].weight
        length = len(orignal_weights_list)
        for i in range(0,length,3):
            current_list = []

            current_list.append(orignal_weights_list[i])
            current_list.append(orignal_weights_list[i+1])
            current_list.append(orignal_weights_list[i+2])

            weights_list.append(current_list)

        return weights_list

def resnet32_rep_gete(num_class, use_norm):
    return ResNet_rep(BasicBlock, [3, 4, 6, 3], num_class)

def resnet32_liner(num_class, use_norm):
    from .resnet import ResNet_clsf
    return ResNet_clsf(num_class, use_norm)