from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import *
from tensorboardX import SummaryWriter
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import *
from network import load_model
from data_process_all import get_patch_seg
from loss import FocalLoss,SoftDiceLoss
from utils import *
import nibabel as nib
from metrics_seg import *

if 1:
    device_ids = [2]
    crop_size = [256, 256, 24]
    epoch=301


inplanes=32
train_BATCH_SIZE=1
test_BATCH_SIZE=1
num_thread = 8
crop_stride=crop_size
num_lesion = 2
num_class=num_lesion
eta1=0.25
eta2=0.25
eta3=0.50
seg_threshold=0.5

model_name='stage1'+'_crop_size_'+str(crop_size[0])+'_'+str(crop_size[2])
model_stem_cls,model_stem_seg,model_cls,model_seg,cross_stitch=load_model(inplanes=inplanes,device_ids=device_ids,pretrain='stage1',crop_size=crop_size,model_name=model_name,test_epoch=epoch)
#### pretrain model


def main():
    val_set = get_patch_seg(num_split=48,phase='Test',crop_size=crop_size,dataset='ours')
    test_data_loader = DataLoader(dataset=val_set, batch_size=test_BATCH_SIZE, num_workers=num_thread,
                                     shuffle=False)

    count = 0


    print("Waiting Test!")
    with torch.no_grad():
        count_test = 0
        val_bar = tqdm(test_data_loader)
        valing_results = {'sen': [], 'spe': [], 'softdice': [], 'harddice': [], 'nsd': [], 'batch_sizes': 0}
        for packs in val_bar:
            ct_scan=packs[0].cuda(device_ids[0])
            ct_crops = packs[1].cuda(device_ids[0])  # [BS,1,32,160,160]
            seg_gt = packs[2].cuda(device_ids[0])  # [BS,1,32,160,160]
            lung_mask = packs[3].float()
            ct_dim = packs[4]

            batch_size = ct_crops.size(0)
            valing_results['batch_sizes'] += batch_size
            model_stem_cls.eval()
            model_stem_seg.eval()
            model_cls.eval()
            model_seg.eval()
            cross_stitch.eval()
            ct_shape = seg_gt.detach().cpu().numpy()[0][0].shape
            ct_crops = ct_crops.cuda(device_ids[0])  # [N,150,128,128,64]
            seg_gt = seg_gt.cuda(device_ids[0])  # [N,2,512,512,200]
            h_crops_num = int(np.ceil((ct_shape[2] - crop_size[2]) / crop_stride[2]) + 1)
            j_crops_num = int((ct_shape[1] - crop_size[1]) / crop_stride[1]) + 1  # 8
            i_crops_num = int((ct_shape[0] - crop_size[0]) / crop_stride[0]) + 1  # 8
            count_val = 0
            predict_whole = np.zeros(
                shape=(num_class, ct_shape[0], ct_shape[1], int(crop_size[2] + crop_stride[2] * np.ceil(
                    (ct_shape[2] - crop_size[2]) / crop_stride[2]))))  # [C,512,512,203]
            for h in range(h_crops_num):
                for j in range(j_crops_num):  # 8
                    for i in range(i_crops_num):  # 8
                        crop_center_x = crop_size[0] / 2 + crop_stride[0] * i
                        crop_center_y = crop_size[1] / 2 + crop_stride[1] * j
                        crop_center_z = crop_size[2] / 2 + crop_stride[2] * h
                        ################# stem ####################
                        output_cls_stem = model_stem_cls(ct_scan)
                        output_seg_stem = model_stem_seg( ct_crops[:, count_val, :, :, :].unsqueeze(1))  # [BS,32,12,128,128]
                        input_cls, input_seg = cross_stitch(output_cls_stem, output_seg_stem, crop_center=[crop_center_x,crop_center_y,crop_center_z,ct_dim[2]], crop_size=crop_size,
                                                            split_num_cls=48)
                        ################# main model of seg and cls ####################
                        masks_pred, seg_out1, seg_out2, seg_out3 = model_seg(output_seg_stem, 0.25, 0.25,0.50)  # [BS,2,32,160,160]
                        cls_out, vis3, vis2, vis1 = model_cls(input_cls, masks_pred, flag_segment=True)
                        _, predicted = torch.max(cls_out.data, 1)

                        masks_pred = masks_pred.detach().cpu().numpy()
                        masks_pred = masks_pred.transpose((0, 1, 3, 4, 2))
                        for c in range(num_class):
                            predict_whole[c][
                            int(crop_center_x - crop_size[0] / 2):int(
                                crop_center_x + crop_size[0] / 2),
                            int(crop_center_y - crop_size[1] / 2):int(
                                crop_center_y + crop_size[1] / 2),
                            int(crop_center_z - crop_size[2] / 2):int(
                                crop_center_z + crop_size[2] / 2)
                            ] = masks_pred[0][c]

                        count_val += 1
                print(h)
            predict_whole_realsize = predict_whole[:, :, :, 0:ct_shape[2]].astype(np.float32)  # [2,512,512,200]
            predict_softmax = softmax_helper(torch.from_numpy(predict_whole_realsize).unsqueeze(0)).detach().numpy()[0]  # [2,512,512,200]
            predict_softmax = predict_softmax * lung_mask.detach().cpu().numpy()[0]  # [2,512,512,200]
            save_nii=np.array(predict_softmax[1] >seg_threshold).astype(np.int16)# (512,512,200)
            save_nii = save_nii[:,::-1,:]  #### (512,512,200)
            new_image = nib.Nifti1Image(save_nii, np.eye(4))
            nib.save(new_image, 'seg_result/'+ model_name+'/'+str(count_test)+'.nii.gz')




def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)
if __name__ == '__main__':
    init_seed = 103
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    import platform

    sysstr = platform.system()
    if (sysstr == "Linux"):
        if not os.path.isdir('model/' + model_name ):
            os.makedirs('model/' + model_name )
        else:
            remove_all_file('model/' + model_name )
        if os.path.isdir('runs/runs_' + model_name ):
            remove_all_file('runs/runs_' + model_name )
    import warnings

    warnings.filterwarnings('ignore')
    main()