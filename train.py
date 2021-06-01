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
from data_process import get_patch_seg
from loss import FocalLoss,SoftDiceLoss
from utils import *

if 1:
    device_ids = [0]
    split_num_cls = 48
    slice_resolu_cls=256
    lr={'lr_cls':2 * 1e-3,'lr_seg':5 * 1e-3,'lr_stem':2 * 1e-3}
    crop_size = [256, 256, 24]
inplanes=32
train_BATCH_SIZE=1
test_BATCH_SIZE=1
num_thread = 4
Epoch = 300
num_epochs_decay = 100
num_lesion = 2
eta1=0.25
eta2=0.25
eta3=0.5
model_name='model'+'_crop_size'+str(crop_size[2])
model_stem_cls,model_stem_seg,model_cls,model_seg,cross_stitch=load_model(inplanes=inplanes,device_ids=device_ids,pretrain=False,crop_size=crop_size)
#### pretrain model

def main():
    train_set = get_patch_seg(num_split=split_num_cls,phase='Train',crop_size=crop_size)
    val_set = get_patch_seg(num_split=split_num_cls,phase='Val',crop_size=crop_size)
    training_data_loader = DataLoader(dataset=train_set, batch_size=train_BATCH_SIZE, num_workers=num_thread,
                                      shuffle=True)
    test_data_loader = DataLoader(dataset=val_set, batch_size=test_BATCH_SIZE, num_workers=num_thread,
                                     shuffle=False)
    optimizer_stem_cls = optim.Adam(model_stem_cls.parameters(), lr=lr['lr_stem'], betas=(0.9, 0.999))
    optimizer_stem_seg = optim.Adam(model_stem_seg.parameters(), lr=lr['lr_stem'], betas=(0.9, 0.999))
    optimizer_cls = optim.Adam(model_cls.parameters(), lr=lr['lr_cls'], betas=(0.9, 0.999))
    optimizer_seg = optim.Adam(model_seg.parameters(), lr=lr['lr_seg'], betas=(0.9, 0.999))
    optimizer_cs = optim.Adam(cross_stitch.parameters(), lr=lr['lr_seg'], betas=(0.9, 0.999))
    CE_criterion_cls = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.5,1,1])).float()).cuda(device_ids[0])
    Focal_criterion = FocalLoss(weight=torch.from_numpy(np.array([0.5,1,1])).float().cuda(device_ids[0])).cuda(device_ids[0])
    CE_criterion_seg = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1, 1.0])).float()).cuda(device_ids[0])
    Dice_criterion_1 = SoftDiceLoss(weight=[0.5, 0.5]).cuda(device_ids[0])
    Dice_criterion_2 = SoftDiceLoss(weight=[0.5,0.5]).cuda(device_ids[0])
    TA_criterion=nn.MSELoss().cuda(device_ids[0])
    writer = SummaryWriter(logdir='runs/runs_' + model_name)
    new_lr = lr
    count = 0
    with open('metrics/metrics_' + model_name  + '.txt', "w+") as f:
        for epoch in range(0, Epoch):
            train_bar = tqdm(training_data_loader)
            running_results = {'batch_sizes': 0,  'dice_loss': 0,  'acc': 0,'acc_loss': 0}
            if (epoch + 1) % 20 ==0:
                new_lr['lr_stem'] = new_lr['lr_stem']/2
                for param_group in optimizer_stem_cls.param_groups:
                    param_group['lr'] = new_lr['lr_stem']

                for param_group in optimizer_stem_seg.param_groups:
                    param_group['lr'] = new_lr['lr_stem']

                new_lr['lr_cls'] = new_lr['lr_cls'] / 2
                for param_group in optimizer_cls.param_groups:
                    param_group['lr'] = new_lr['lr_cls']

                new_lr['lr_seg'] = new_lr['lr_seg'] / 2
                for param_group in optimizer_seg.param_groups:
                    param_group['lr'] = new_lr['lr_seg']
            count_dice=0
            for packs in train_bar:
                flag_segment=packs[0].numpy()[0]
                ct_scans = packs[1].cuda(device_ids[0])  # [BS,1,60,512,512]
                labels = packs[2].cuda(device_ids[0])
                ct_crop = packs[3].cuda(device_ids[0])  # [BS,1,32,160,160]
                if flag_segment:
                    gt_crop= packs[4].cuda(device_ids[0])#[BS,1,32,160,160]
                    lung_crop= packs[5].cuda(device_ids[0])#[BS,1,32,160,160]
                    crop_center = packs[6]
                else:
                    lung_crop = packs[4].cuda(device_ids[0])  # [BS,1,32,160,160]
                    crop_center= packs[5]
                running_results['batch_sizes'] += 1
                count += 1
                model_stem_cls.train()
                model_stem_seg.train()
                model_cls.train()
                model_seg.train()
                cross_stitch.train()

                ct_scans=F.interpolate(ct_scans,size=(ct_scans.size(2),slice_resolu_cls,slice_resolu_cls))# [BS,1,60,256,256]

                ################# stem ####################
                output_cls_stem= model_stem_cls(ct_scans) # [BS,1,30,128,128]
                output_seg_stem = model_stem_seg(ct_crop)  # [BS,1,16,80,80]
                input_cls,input_seg=cross_stitch(output_cls_stem,output_seg_stem,crop_center,crop_size,split_num_cls)


                ################# main model of seg and cls ####################
                seg_out,seg_out1,seg_out2,seg_out3 = model_seg(input_seg,eta1,eta2,eta3)  # [BS,2,32,160,160]
                cls_out,vis3,vis2,vis1=model_cls(input_cls,seg_out,flag_segment)   # gap_weight:[N,256]

                Size_vis1 = vis1.detach().cpu().numpy().shape
                vis1_temp = F.interpolate(vis1, size=( int(Size_vis1[2] *8), int(Size_vis1[3] *8), int(Size_vis1[4] *8)))  # [BS,1,30,128,128]
                Size_vis1_temp= vis1_temp.detach().cpu().numpy().shape
                ratio_z = crop_center[3] / Size_vis1_temp[2]
                ratio_x = 512 / Size_vis1_temp[3]
                ratio_y = 512 / Size_vis1_temp[4]
                vis1_temp2 = vis1_temp[:, :,
                                  int(crop_center[2] / ratio_z - crop_size[2] / ratio_z / 2):int(
                                      crop_center[2] / ratio_z + crop_size[2] / ratio_z / 2),
                                  int(crop_center[0] / ratio_x - crop_size[0] / ratio_x / 2):int(
                                      crop_center[0] / ratio_x + crop_size[0] / ratio_x / 2),
                                  int(crop_center[1] / ratio_y - crop_size[1] / ratio_y / 2):int(
                                      crop_center[1] / ratio_y + crop_size[1] / ratio_y / 2),
                                  ]
                Size_seg_out1=seg_out1.detach().cpu().numpy().shape
                vis1 = F.interpolate(vis1_temp2, size=(int(Size_seg_out1[2]), int(Size_seg_out1[3]), int(Size_seg_out1[4] )))  # [BS,1,30,128,128]

                Size_vis2 = vis2.detach().cpu().numpy().shape
                vis2_temp = F.interpolate(vis2, size=(
                int(Size_vis2[2] * 4), int(Size_vis2[3] * 4), int(Size_vis2[4] * 4)))  # [BS,1,30,128,128]
                Size_vis2_temp = vis2_temp.detach().cpu().numpy().shape
                ratio_z = crop_center[3] / Size_vis2_temp[2]
                ratio_x = 512 / Size_vis2_temp[3]
                ratio_y = 512 / Size_vis2_temp[4]
                vis2_temp2 = vis2_temp[:, :,
                             int(crop_center[2] / ratio_z - crop_size[2] / ratio_z / 2):int(
                                 crop_center[2] / ratio_z + crop_size[2] / ratio_z / 2),
                             int(crop_center[0] / ratio_x - crop_size[0] / ratio_x / 2):int(
                                 crop_center[0] / ratio_x + crop_size[0] / ratio_x / 2),
                             int(crop_center[1] / ratio_y - crop_size[1] / ratio_y / 2):int(
                                 crop_center[1] / ratio_y + crop_size[1] / ratio_y / 2),
                             ]
                Size_seg_out2 = seg_out2.detach().cpu().numpy().shape
                vis2 = F.interpolate(vis2_temp2, size=(
                int(Size_seg_out2[2]), int(Size_seg_out2[3]), int(Size_seg_out2[4])))  # [BS,1,30,128,128]

                Size_vis3 = vis3.detach().cpu().numpy().shape
                vis3_temp = F.interpolate(vis3, size=(
                int(Size_vis3[2] * 8), int(Size_vis3[3] * 8), int(Size_vis3[4] * 8)))  # [BS,1,30,128,128]
                Size_vis3_temp = vis3_temp.detach().cpu().numpy().shape
                ratio_z = crop_center[3] / Size_vis3_temp[2]
                ratio_x = 512 / Size_vis3_temp[3]
                ratio_y = 512 / Size_vis3_temp[4]
                vis3_temp2 = vis3_temp[:, :,
                             int(crop_center[2] / ratio_z - crop_size[2] / ratio_z / 2):int(
                                 crop_center[2] / ratio_z + crop_size[2] / ratio_z / 2),
                             int(crop_center[0] / ratio_x - crop_size[0] / ratio_x / 2):int(
                                 crop_center[0] / ratio_x + crop_size[0] / ratio_x / 2),
                             int(crop_center[1] / ratio_y - crop_size[1] / ratio_y / 2):int(
                                 crop_center[1] / ratio_y + crop_size[1] / ratio_y / 2),
                             ]
                Size_seg_out3 = seg_out3.detach().cpu().numpy().shape
                vis3 = F.interpolate(vis3_temp2, size=(
                int(Size_seg_out3[2]), int(Size_seg_out3[3]), int(Size_seg_out3[4])))  # [BS,1,30,128,128]

                Loss_cls = CE_criterion_cls(cls_out, labels)
                Loss_cls_1 = Focal_criterion(cls_out, labels)
                Loss_cls_2 = CE_criterion_cls(cls_out, labels)
                Loss_cls = Loss_cls_2 + Loss_cls_1 * 2


                if flag_segment:
                    masks_pred_transpose = seg_out.permute(0, 2, 3, 4, 1)  # [N,32,160,160,2]
                    masks_pred_flat = masks_pred_transpose.reshape(-1,
                                                                   masks_pred_transpose.shape[-1])  # [N*160*160*32,2]
                    true_masks_indices = torch.argmax(gt_crop, 1).reshape(-1)  # [N*160*160*32]
                    Dice_Loss = Dice_criterion_1(x=seg_out, y=gt_crop, lung_mask=lung_crop)
                    Focal_Loss = CE_criterion_seg(masks_pred_flat, true_masks_indices)
                    Loss_seg = Dice_Loss  + Focal_Loss* 2
                    optimizer_stem_seg.zero_grad()
                    optimizer_seg.zero_grad()
                    optimizer_cs.zero_grad()
                    Loss_seg.backward(retain_graph=True)
                    optimizer_stem_seg.step()
                    optimizer_seg.step()
                    optimizer_cs.step()
                    running_results['dice_loss'] += Dice_Loss.item()
                    count_dice+=1
                else:
                    Loss_seg = 0
                TA_loss = eta1 * TA_criterion(seg_out1, vis1) + eta2 * TA_criterion(seg_out2,vis2) + eta3 * TA_criterion(
                    seg_out3, vis3)
                Loss_seg = 5 * TA_loss + Loss_seg

                optimizer_stem_cls.zero_grad()
                optimizer_cls.zero_grad()
                Loss_cls.backward()
                optimizer_stem_cls.step()
                optimizer_cls.step()

                _, predicted = torch.max(cls_out.data, 1)
                total = labels.size(0)
                correct = predicted.eq(labels.data).cpu().sum()
                acc = 100. * correct / total

                running_results['acc'] += acc.detach().cpu().numpy()
                running_results['acc_loss']+= Loss_cls.item()
                train_bar.set_description(
                    desc=model_name + ' [%d/%d] acc_loss: %.4f | dice_loss: %.4f | Acc: %.4f' % (
                        epoch, Epoch,
                        running_results['acc_loss'] / running_results['batch_sizes'],
                        (0 if count_dice==0 else running_results['dice_loss'] / count_dice),
                        running_results['acc'] / running_results['batch_sizes']
                    ))


                a=1

            ############ Val####################
            print("Waiting Test!")
            if epoch%2==0:
                with torch.no_grad():
                    correct_all = 0
                    total_all = 0
                    correct_perclass = [0, 0, 0, 0, 0]
                    total_perclass = [0, 0, 0, 0, 0]
                    test_bar = tqdm(test_data_loader)
                    valing_results = {'dice': 0,
                                      'batch_sizes': 0}
                    for packs in test_bar:
                        begin_time = time()
                        flag_segment = packs[0].numpy()[0]
                        ct_scans = packs[1].cuda(device_ids[0])  # [BS,1,60,512,512]
                        labels = packs[2].cuda(device_ids[0])
                        ct_crop = packs[3].cuda(device_ids[0])  # [BS,1,32,160,160]
                        if flag_segment:
                            gt_crop = packs[4].cuda(device_ids[0])  # [BS,1,32,160,160]
                            lung_crop = packs[5].cuda(device_ids[0])  # [BS,1,32,160,160]
                            crop_center = packs[6]
                        else:
                            lung_crop = packs[4].cuda(device_ids[0])  # [BS,1,32,160,160]
                            crop_center = packs[5]
                        model_stem_cls.eval()
                        model_stem_seg.eval()
                        model_cls.eval()
                        model_seg.eval()
                        cross_stitch.eval()
                        ct_scans=F.interpolate(ct_scans,size=(ct_scans.size(2),slice_resolu_cls,slice_resolu_cls))# [BS,1,60,256,256]
                        ################# stem ####################
                        output_cls_stem = model_stem_cls(ct_scans)  # [BS,32,30,128,128]
                        output_seg_stem = model_stem_seg(ct_crop)  # [BS,32,16,80,80]

                        alpha_1 = torch.nn.Parameter(torch.FloatTensor([0.8]),requires_grad=False).cuda(device_ids[0])
                        beta_1 = torch.nn.Parameter(torch.FloatTensor([0.2]),requires_grad=False).cuda(device_ids[0])
                        alpha_2 = torch.nn.Parameter(torch.FloatTensor([0.8]),requires_grad=False).cuda(device_ids[0])
                        beta_2 = torch.nn.Parameter(torch.FloatTensor([0.2]),requires_grad=False).cuda(device_ids[0])
                        start = time()
                        Size_cls_stem = output_cls_stem.detach().cpu().numpy().shape
                        input_cls_part2 = torch.zeros(Size_cls_stem[0], Size_cls_stem[1], int(crop_center[3] / 2), 256,
                                                      256)  # [BS,32,T/2,256,256]
                        input_cls_part2[:, :,
                        int(crop_center[2] / 2 - crop_size[2] / 4):int(crop_center[2] / 2 + crop_size[2] / 4),
                        int(crop_center[0] / 2 - crop_size[0] / 4):int(crop_center[0] / 2 + crop_size[0] / 4),
                        int(crop_center[1] / 2 - crop_size[1] / 4):int(crop_center[1] / 2 + crop_size[1] / 4)
                        ] = output_seg_stem.detach().cpu()  # [BS,32,T/2,256,256]
                        slice_width = (crop_center[3] - 1) / split_num_cls  # 2.8
                        crop_num = np.array([np.int16(slice_width * i) for i in range(int(split_num_cls / 2))])[:, 0]
                        input_cls_part2 = input_cls_part2[:, :, crop_num, :, :]  # [BS,32,30,256,256]
                        input_cls_part2 = F.interpolate(input_cls_part2, size=(
                            int(input_cls_part2.size(2)), int(input_cls_part2.size(3) / 2),
                            int(input_cls_part2.size(4) / 2)))  # [BS,1,30,128,128]
                        input_cls = alpha_1[0] * output_cls_stem + beta_1[0] * input_cls_part2.cuda(device_ids[0])

                        ratio_z = crop_center[3] / Size_cls_stem[2]
                        ratio_x = 512 / Size_cls_stem[3]
                        ratio_y = 512 / Size_cls_stem[4]
                        input_seg_part2 = output_cls_stem[:, :,
                                          int(crop_center[2] / ratio_z - crop_size[2] / ratio_z / 2):int(
                                              crop_center[2] / ratio_z + crop_size[2] / ratio_z / 2),
                                          int(crop_center[0] / ratio_x - crop_size[0] / ratio_x / 2):int(
                                              crop_center[0] / ratio_x + crop_size[0] / ratio_x / 2),
                                          int(crop_center[1] / ratio_y - crop_size[1] / ratio_y / 2):int(
                                              crop_center[1] / ratio_y + crop_size[1] / ratio_y / 2),
                                          ]
                        input_seg_part2 = F.interpolate(input_seg_part2, size=(
                            int(crop_size[2] / 2), int(crop_size[0] / 2), int(crop_size[1] / 2)))  # [BS,1,30,128,128]
                        input_seg = alpha_2[0] * output_seg_stem + beta_2[0] * input_seg_part2.cuda(device_ids[0])

                        ################# main model of seg and cls ####################
                        seg_out = model_seg(output_seg_stem)  # [BS,2,32,160,160]
                        cls_out, gt_crop_semi = model_cls(output_cls_stem, seg_out, ct_crop,
                                                          flag_segment)  # gap_weight:[N,256]

                        end_time = time()
                        print('time:',end_time-begin_time)
                        if valing_results['batch_sizes']==0:
                            dice_val=0
                        if flag_segment:
                            valing_results['batch_sizes'] += test_BATCH_SIZE

                            from utils import DSC
                            Dice = DSC(x=seg_out, y=gt_crop, lung_mask=lung_crop)
                            valing_results['dice'] += Dice.detach().cpu().numpy()
                            dice_val= valing_results['dice'] / valing_results['batch_sizes']

                        _, predicted = torch.max(cls_out.data, 1)
                        total_all += labels.size(0)
                        correct_all += (predicted == labels).sum()
                        acc = 100. * correct_all / total_all
                        labels = labels.cpu().numpy()
                        if labels[0]==2:
                            a=1
                        predicted = predicted.cpu().numpy()
                        for i_test in range(test_BATCH_SIZE):
                            total_perclass[labels[i_test]] += 1
                            if predicted[i_test] == labels[i_test]:
                                correct_perclass[labels[i_test]] += 1
                        test_bar.set_description(
                            desc=model_name + ' [%d/%d]  dice: %.4f | acc: %.4f' % (
                                epoch, Epoch,
                                dice_val,
                                acc.detach().cpu().numpy()
                            ))

                    correct_all = correct_all.cpu().numpy()
                    OA = 100. * correct_all / total_all
                    Acc_0 = 100. * correct_perclass[0] / total_perclass[0]
                    Acc_1 = 100. * correct_perclass[1] / total_perclass[1]
                    Acc_2 = 100. * correct_perclass[2] / total_perclass[2]
                    print(
                        'Testset OA=：%.3f%%   Acc_0=：%.3f%% | Acc_1=：%.3f%% | Acc_2=：%.3f%%  |dice: %.4f'
                        % (OA,  Acc_0, Acc_1, Acc_2,dice_val))
                    writer.add_scalar('scalar/test_OA', OA, epoch)
                    writer.add_scalar('scalar/Dice_Val', dice_val, epoch)



            """------------------Test--------------"""
            if epoch % 4 == 0 :

                torch.save(model_stem_cls.state_dict(),'model/' + model_name + '/model_' + str(epoch + 1) + '_stem_cls'+'.pth')
                torch.save(model_stem_seg.state_dict(),
                           'model/' + model_name + '/model_' + str(epoch + 1) + '_stem_seg' + '.pth')
                torch.save(model_cls.state_dict(),
                           'model/' + model_name + '/model_' + str(epoch + 1) + '_cls' + '.pth')
                torch.save(model_seg.state_dict(),
                           'model/' + model_name + '/model_' + str(epoch + 1) + '_seg' + '.pth')
                torch.save(cross_stitch.state_dict(),
                           'model/' + model_name + '/model_' + str(epoch + 1) + '_cross_stitch' + '.pth')




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