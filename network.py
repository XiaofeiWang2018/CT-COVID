import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial



"""STEM part"""

class stem(nn.Module):

    def __init__(self,inplanes):
        super(stem, self).__init__()
        self.inplanes=inplanes
        self.branch1=nn.Conv3d(1, self.inplanes//4, kernel_size=1, stride=(2, 2, 2),
                                bias=False)



        self.branch21 = nn.Conv3d(1, self.inplanes // 4, kernel_size=1, stride=(1, 1, 1),
                                  bias=False)
        self.branch22 = nn.Conv3d(self.inplanes // 4, self.inplanes // 4, kernel_size=3, stride=(2, 2, 2),
                                  padding=(1, 1, 1), bias=False)


        self.branch31 = nn.Conv3d(1, self.inplanes // 4, kernel_size=1, stride=(1, 1, 1),
                                  bias=False)
        self.branch32 = nn.Conv3d(self.inplanes // 4, int(self.inplanes // 8*3), kernel_size=3, stride=(2, 2, 2),
                                  padding=(1, 1, 1), bias=False)
        self.branch33 = nn.Conv3d(self.inplanes // 8*3, int(self.inplanes //8*3), kernel_size=3, stride=(1, 1, 1),
                                  padding=(1, 1, 1), bias=False)



        self.branch41 = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=1, padding=1)
        self.branch42 = nn.Conv3d(1, self.inplanes  // 8, kernel_size=3, stride=(2, 2, 2),
                                  padding=(1, 1, 1), bias=False)
        self.bn1 = nn.GroupNorm(num_groups=2, num_channels=self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

    def forward(self, x):
        out1=self.branch1(x)
        out2 = self.branch22(self.branch21(x))
        out3 = self.branch33(self.branch32(self.branch31(x)))
        out4 = self.branch42(self.branch41(x))

        out=torch.cat((out1,out2),1)
        out=torch.cat((out,out3),1)
        out = torch.cat((out, out4), 1)
        out=self.relu(self.bn1(out))
        # out=self.maxpool(skip_out)
        return out




"""3D segmentation part"""

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)
        # self.bn1 = nn.GroupNorm(num_groups=2, num_channels=out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)


    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 16 * (2 ** (depth+1)),act)
        layer2 = LUConv(16 * (2 ** (depth+1)), 16 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 16*(2**depth),act)
        layer2 = LUConv(16*(2**depth), 16*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)



class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        if self.depth==0:
            self.ops = _make_nConv(inChans, depth, act, double_chnnel=True)
        else:
            self.ops = _make_nConv(inChans + outChans // 2, depth, act, double_chnnel=True)


    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)

        if self.depth==0:
            out = self.ops(out_up_conv)
        else:
            concat = torch.cat((out_up_conv, skip_x), 1)
            out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out = self.sigmoid(x)
        out=x
        return out





class Seg(nn.Module):

    def __init__(self, n_class=2, act='relu'):
        super(Seg, self).__init__()

        # self.first_layer=stem(inplanes=32)
        # self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(32,1,act)
        self.down_tr256 = DownTransition(64,2,act)
        self.down_tr512 = DownTransition(128,3,act)

        self.up_tr256 = UpTransition(256, 256,2,act)
        self.up_tr128 = UpTransition(128,128, 1,act)
        self.up_tr64 = UpTransition(64,64,0,act)
        self.out_tr = OutputTransition(32, n_class)

        self.poly1=nn.Conv3d(256, 2, kernel_size=1)
        self.poly2 = nn.Conv3d(128, 2, kernel_size=1)
        self.poly3 = nn.Conv3d(64, 2, kernel_size=1)
        self.poly4 = nn.Conv3d(32, 2, kernel_size=1)


    def forward(self, x,eta1,eta2,eta3):
        # self.out64 = self.first_layer(x)
        self.out128,self.skip_out128 = self.down_tr128(x)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512,self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, None)

        self.out1_ori = self.poly1(self.out512)
        self.out2_ori = self.poly2(self.out_up_256)
        self.out3_ori = self.poly3(self.out_up_128)
        self.out4_ori = self.poly4(self.out_up_64)

        self.out1 = F.interpolate(self.out1_ori, size=(self.out_up_64.shape[2], self.out_up_64.shape[3], self.out_up_64.shape[4]))
        self.out2 = F.interpolate(self.out2_ori, size=(self.out_up_64.shape[2], self.out_up_64.shape[3], self.out_up_64.shape[4]))
        self.out3 = F.interpolate(self.out3_ori, size=(self.out_up_64.shape[2], self.out_up_64.shape[3], self.out_up_64.shape[4]))
        self.out4 = F.interpolate(self.out4_ori, size=(self.out_up_64.shape[2], self.out_up_64.shape[3], self.out_up_64.shape[4]))

        self.out = self.out_tr(eta1*self.out1+eta2*self.out2+eta3*self.out3+self.out4)

        return self.out,self.out1_ori,self.out2_ori,self.out3_ori


"""3D classification part"""


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=2, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=2, num_channels=out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3,
                     stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=2, num_channels=out_planes)

    def forward(self, x):

        out1 = self.conv1(x)
        out = self.bn1(out1)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += out1
        out = self.relu(out)
        return out
import weakref
import inspect
import builtins
import torch.nn.functional as F
import torch.nn.init as init
# Tracks standalone weak script functions
_compiled_weak_fns = weakref.WeakKeyDictionary()

# Tracks which methods should be converted to strong methods
_weak_script_methods = weakref.WeakKeyDictionary()

# Converted modules and their corresponding WeakScriptModuleProxy objects
_weak_modules = weakref.WeakKeyDictionary()

# Types that have been declared as weak modules
_weak_types = weakref.WeakKeyDictionary()

def createResolutionCallback(frames_up=0):
    frame = inspect.currentframe()
    i = 0
    while i < frames_up + 1:
        frame = frame.f_back
        i += 1

    f_locals = frame.f_locals
    f_globals = frame.f_globals

    def env(key):
        if key in f_locals:
            return f_locals[key]
        elif key in f_globals:
            return f_globals[key]
        elif hasattr(builtins, key):
            return getattr(builtins, key)
        else:
            return None

    return env


    return fn
def weak_module(cls):
    _weak_types[cls] = {
        "method_stubs": None
    }
    return cls


def weak_script_method(fn):
    _weak_script_methods[fn] = {
        "rcb": createResolutionCallback(frames_up=2),
        "original_method": fn
    }
    return fn

@weak_module
class Linear(nn.Module):

    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    @weak_script_method
    def forward(self, concat_gap,out1,out2,out3):
        fc_out=torch.addmm(torch.jit._unwrap_optional(self.bias), concat_gap, self.weight.t())
        vis1_covid=0
        for i in range(64):
            vis1_covid+=out1[:,i,...]*self.weight[0,i]
        vis1_cap = 0
        for i in range(64):
            vis1_cap += out1[:, i, ...] * self.weight[1, i]
        vis1=(vis1_covid+vis1_cap)/2
        vis1=torch.unsqueeze(vis1, 1)

        vis2_covid = 0
        for i in range(128):
            vis2_covid += out2[:, i, ...] * self.weight[0, i+64]
        vis2_cap = 0
        for i in range(128):
            vis2_cap += out2[:, i, ...] * self.weight[1, i+64]
        vis2 = (vis2_covid + vis2_cap) / 2
        vis2 = torch.unsqueeze(vis2, 1)

        vis3_covid = 0
        for i in range(256):
            vis3_covid += out3[:, i, ...] * self.weight[0, i+192]
        vis3_cap = 0
        for i in range(256):
            vis3_cap += out3[:, i, ...] * self.weight[1, i+192]
        vis3 = (vis3_covid + vis3_cap) / 2
        vis3 = torch.unsqueeze(vis3, 1)

        return fc_out, vis1, vis2, vis3


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



class Cls(nn.Module):

    def __init__(self, block, layers,inplanes=32):
        self.inplanes = inplanes
        super(Cls, self).__init__()

        self.encoder1 = BasicBlock(32,64)
        self.encoder2 = BasicBlock(64, 128)
        self.encoder3 = BasicBlock(128, 256)
        self.p=torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.GAP=nn.AdaptiveAvgPool3d((1, 1,1))
        self.fc = Linear(448, 3)
        self.layer1_semi = nn.Conv3d(1, 16,kernel_size=1, stride=(1,2,2), bias=False)
        self.layer1_1_semi = nn.Conv3d(32, 16, kernel_size=1, stride=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_cls,seg_out,flag_segment): #[N,Z,W,H]
        """

        :param input_cls:  [BS,32,30,128,128]
        :param seg_out: [BS,2,32,160,160]
        :param ct_crop:  [BS,1,32,160,160]
        :param flag_segment:
        :return:
        """

        seg_out = torch.unsqueeze(seg_out[:, 1, ...], 1)
        seg_out = self.p*self.layer1_semi(seg_out)  # [BS,16 ,32,160,160]
        input_cls = self.layer1_1_semi(input_cls)  # [BS,16 ,32,160,160]
        concat = torch.cat((seg_out, input_cls), 1)  # [BS,32 ,32,160,160]
        out1=self.encoder1(concat)
        out2 = self.encoder2(out1)
        out3 = self.encoder3(out2)

        out1_gap=self.GAP(out1)
        out2_gap = self.GAP(out2)
        out3_gap = self.GAP(out3)

        concat_gap=torch.cat((out1_gap, out2_gap,out3_gap), 1)[:,:,0,0,0]

        fc_out, vis1, vis2, vis3=self.fc(concat_gap,out1,out2,out3)

        return fc_out,vis1,vis2,vis3


def cls(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = Cls(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


import numpy as np
@weak_module
class task_inter(nn.Module):
    def __init__(self, device_ids):
        super(task_inter, self).__init__()
        self.device_ids = device_ids
        self.alpha_1 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.beta_1 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.alpha_2 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.beta_2 = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)


    def forward(self, output_cls_stem,output_seg_stem,crop_center,crop_size,split_num_cls,phase='Train'):

        Size_cls_stem = output_cls_stem.detach().cpu().numpy().shape
        input_cls_part2 = torch.zeros(Size_cls_stem[0], Size_cls_stem[1], int(crop_center[3] / 2), 256,
                                      256)  # [BS,32,T/2,256,256]
        # print(current[0],current[1],current[2])

        if int(crop_center[2] / 2 + crop_size[2] / 4)>=int(crop_center[3] / 2):
            input_cls_part2[:, :,
            int(crop_center[2] / 2 - crop_size[2] / 4)-((int(crop_center[2] / 2 + crop_size[2] / 4))-(int(crop_center[3] / 2)-1)):int(crop_center[3] / 2)-1,
            int(crop_center[0] / 2 - crop_size[0] / 4):int(crop_center[0] / 2 + crop_size[0] / 4),
            int(crop_center[1] / 2 - crop_size[1] / 4):int(crop_center[1] / 2 + crop_size[1] / 4)
            ] = output_seg_stem.detach().cpu()
        else:
            input_cls_part2[:, :,
            int(crop_center[2] / 2 - crop_size[2] / 4):int(crop_center[2] / 2 + crop_size[2] / 4),
            int(crop_center[0] / 2 - crop_size[0] / 4):int(crop_center[0] / 2 + crop_size[0] / 4),
            int(crop_center[1] / 2 - crop_size[1] / 4):int(crop_center[1] / 2 + crop_size[1] / 4)
            ] = output_seg_stem.detach().cpu()

        slice_width = ((crop_center[3] - 1) / split_num_cls).detach().cpu().numpy()# 2.8
        crop_num =[np.int16(slice_width * i) for i in range(int(split_num_cls / 2))]
        crop_num = np.array(crop_num)[:, 0]
        input_cls_part2 = input_cls_part2[:, :, crop_num, :, :]  # [BS,32,30,256,256]
        input_cls_part2 = F.interpolate(input_cls_part2, size=(
            int(input_cls_part2.size(2)), int(input_cls_part2.size(3) / 2),
            int(input_cls_part2.size(4) / 2)))  # [BS,32,30,128,128]
        input_cls = self.alpha_1 * output_cls_stem + self.beta_1* input_cls_part2.cuda(self.device_ids[0])

        num_slices_part2 = int(split_num_cls * crop_size[2] / crop_center[3])
        ratio_z = crop_center[3] / Size_cls_stem[2]
        ratio_x = 512 / Size_cls_stem[3]
        ratio_y = 512 / Size_cls_stem[4]

        if phase=='Test':
            k1 = np.ceil((crop_center[2] / ratio_z.detach().cpu().numpy()))[0]
        elif phase=='Train':
            k1=np.ceil((crop_center[2].detach().cpu().numpy() / ratio_z.detach().cpu().numpy()))[0]
        k2=np.ceil(np.ceil(crop_size[2] / ratio_z.detach().cpu().numpy()[0]) / 2)
        delta_k1_subtraction=int(k1 - k2)
        delta_k1_plus=int(k1 + k2)
        if k2==0:
            delta_k1_plus=delta_k1_plus+1

        if delta_k1_subtraction<0:
            delta_k1_subtraction=0
        if delta_k1_subtraction == 0:
            if k1==0:
                delta_k1_subtraction=0
                delta_k1_plus=1
        # print(output_cls_stem.shape[2],delta_k1_subtraction,delta_k1_plus)
        if delta_k1_plus>output_cls_stem.shape[2]-1:
            delta_k1_subtraction = delta_k1_subtraction - (delta_k1_plus - output_cls_stem.shape[2] + 1)
            delta_k1_plus=output_cls_stem.shape[2]-1

        # print(output_cls_stem.shape[2],delta_k1_subtraction,delta_k1_plus)
        input_seg_part2 = output_cls_stem[:, :,delta_k1_subtraction:delta_k1_plus,
                          int(crop_center[0] / ratio_x - (crop_size[0] / ratio_x) / 2):int(
                              crop_center[0] / ratio_x + (crop_size[0] / ratio_x) / 2),
                          int(crop_center[1] / ratio_y - (crop_size[1] / ratio_y) / 2):int(
                              crop_center[1] / ratio_y + (crop_size[1] / ratio_y) / 2),
                          ]

        input_seg_part2 = F.interpolate(input_seg_part2, size=(
            int(crop_size[2] / 2), int(crop_size[0] / 2), int(crop_size[1] / 2)))  # [BS,1,30,128,128]
        input_seg = self.alpha_2 * output_seg_stem + self.beta_2 * input_seg_part2.cuda(self.device_ids[0])

        return input_cls,input_seg






""" load pretrained models"""


def load_model(inplanes=32,device_ids=[0],pretrain=True,crop_size=[160,160,32]):
    model_stem_cls = stem(inplanes=inplanes)
    model_stem_cls = nn.DataParallel(model_stem_cls, device_ids=device_ids)
    model_stem_seg = stem(inplanes=inplanes)
    model_stem_seg = nn.DataParallel(model_stem_seg, device_ids=device_ids)
    model_cls = cls(inplanes=inplanes).cuda(device_ids[0])
    model_cls = nn.DataParallel(model_cls, device_ids=device_ids)
    model_seg = Seg().cuda(device_ids[0])
    model_seg = nn.DataParallel(model_seg, device_ids=device_ids)
    cross_stitch=task_inter(device_ids).cuda(device_ids[0])
    cross_stitch = nn.DataParallel(cross_stitch, device_ids=device_ids)
    print('# network parameters:', sum(param.numel() for param in model_stem_cls.parameters()) / 1e6, 'M')
    print('# network parameters:', sum(param.numel() for param in model_stem_seg.parameters()) / 1e6, 'M')
    print('# network parameters:', sum(param.numel() for param in model_cls.parameters()) / 1e6, 'M')
    print('# network parameters:', sum(param.numel() for param in model_seg.parameters()) / 1e6, 'M')

    if pretrain:
        model_stem_cls.load_state_dict(torch.load('pretrained_model/best_stem_cls.pth' ))
        model_stem_seg.load_state_dict(torch.load('pretrained_model/best_stem_seg.pth'))
        model_cls.load_state_dict(torch.load('pretrained_model/best_cls.pth'))
        model_seg.load_state_dict(torch.load('pretrained_model/best_seg.pth'))
        cross_stitch.load_state_dict(torch.load('pretrained_model/best_cross_stitch.pth'))

    return model_stem_cls,model_stem_seg,model_cls,model_seg,cross_stitch










