
import torch.utils.data as data
import os
import torch
import platform
import numpy as np
sysstr = platform.system()
import nibabel as nib
from matplotlib import pyplot as plt
from skimage import transform
from PIL import Image
import random
import cv2




class Dataset_covid(data.Dataset):
    def __init__(self, ct_paths,lung_path, mask_paths,phase, num_split,crop_size,dataset):
        super(Dataset_covid, self).__init__()
        self.lung_path=lung_path
        self.ct_paths = ct_paths
        self.mask_paths = mask_paths
        self.phase=phase
        self.num_split=num_split
        self.crop_size = crop_size
        self.crop_stride =crop_size
        self.dataset = dataset


    def __len__(self):
        return len(self.ct_paths)
    def __getitem__(self, index):

        ct_nii = nib.load(self.ct_paths[index])
        ct_nii = np.int16(ct_nii.dataobj)  # (512,512,200)
        ct_nii[(ct_nii > 200)] = 200
        ct_nii[(ct_nii < -1400)] = -1400
        ct_max = np.max(ct_nii, axis=(0, 1))
        ct_min = np.min(ct_nii, axis=(0, 1))
        scans = ((ct_nii - (ct_min)) / (ct_max - ct_min) * 1.0)  # (512,512,200)
        scans = scans[..., ::-1]
        if self.ct_paths[index][11:14]=='cap' and int(self.ct_paths[index][24:-7])>=214:
            scans = np.rot90(scans,k=-1)
        if self.ct_paths[index][11:14] == 'nor' and int(self.ct_paths[index][27:-7]) >= 235:
            scans = np.rot90(scans,k=-1)
        ct_dim = scans.shape  # [512,512,200]
        scan_start = 0
        scan_end =ct_dim[2]-1

        gt_nii = nib.load(self.mask_paths[index])

        seg_gt = np.uint8(gt_nii.dataobj)  # (512,512,200)
        assert seg_gt.shape == scans.shape
        seg_gt = np.fliplr(seg_gt)
        seg_gt[seg_gt > 0] = 1


        lung_mask=np.load(self.lung_path[index])# (512,512,200)
        lung_mask[lung_mask > 0] = 1
        lung_mask = lung_mask[:, :, ::-1]
        lung_mask = np.rot90(lung_mask)
        lung_mask = lung_mask[ :,::-1, :]
        lung_mask =lung_mask[::-1,:,:]

        scans=scans*lung_mask
        slice_width = (scan_end - scan_start) / self.num_split  # 2.8
        crop_num = [np.int16(scan_start + slice_width * i) for i in range(self.num_split)]
        ct_scans = scans[..., crop_num]
        ct_scans = ct_scans.transpose((2, 0, 1))

        ct_scans =np.array(ct_scans)
        ct_scans =np.expand_dims(ct_scans, axis=0)
        ct_scans = np.float32(ct_scans / 1.0)#  [1,num_split,512,512]

        if self.phase=='Train' or self.phase=='Val' :
            gt_crop_count = 0
            count = 1
            Random = random.uniform(0, 1)
            if Random >= 0.4:
                while gt_crop_count < np.floor(
                        self.crop_size[0] * self.crop_size[1] * self.crop_size[2] * 0.01) and (count < 50):
                    crop_center_x = int(random.randint(100 + self.crop_size[0] / 2, 412 - self.crop_size[0] / 2))
                    crop_center_y = int(random.randint(100 + self.crop_size[1] / 2, 412 - self.crop_size[1] / 2))
                    if ct_dim[2] < 100:
                        crop_center_z = int(
                            random.randint(self.crop_size[2] / 2, ct_dim[2] - self.crop_size[2] / 2))
                    else:
                        crop_center_z = int(
                            random.randint(30 + self.crop_size[2] / 2, ct_dim[2] - 30 - self.crop_size[2] / 2))

                    gt_crop = seg_gt[
                              int(crop_center_x - self.crop_size[0] / 2):int(crop_center_x + self.crop_size[0] / 2),
                              int(crop_center_y - self.crop_size[1] / 2):int(crop_center_y + self.crop_size[1] / 2),
                              int(crop_center_z - self.crop_size[2] / 2):int(crop_center_z + self.crop_size[2] / 2)
                              ]
                    count += 1
                    gt_crop_count = np.sum(gt_crop)

            else:
                crop_center_x = int(random.randint(0 + self.crop_size[0] / 2, 512 - self.crop_size[0] / 2))
                crop_center_y = int(random.randint(0 + self.crop_size[1] / 2, 512 - self.crop_size[1] / 2))
                if ct_dim[2] < 100:
                    crop_center_z = int(
                        random.randint(self.crop_size[2] / 2, ct_dim[2] - self.crop_size[2] / 2))
                else:
                    crop_center_z = int(
                        random.randint(0 + self.crop_size[2] / 2, ct_dim[2] - 0 - self.crop_size[2] / 2))

                gt_crop = seg_gt[
                          int(crop_center_x - self.crop_size[0] / 2):int(crop_center_x + self.crop_size[0] / 2),
                          int(crop_center_y - self.crop_size[1] / 2):int(crop_center_y + self.crop_size[1] / 2),
                          int(crop_center_z - self.crop_size[2] / 2):int(crop_center_z + self.crop_size[2] / 2)
                          ]
            ct_crop = scans[
                      int(crop_center_x - self.crop_size[0] / 2):int(crop_center_x + self.crop_size[0] / 2),
                      int(crop_center_y - self.crop_size[1] / 2):int(crop_center_y + self.crop_size[1] / 2),
                      int(crop_center_z - self.crop_size[2] / 2):int(crop_center_z + self.crop_size[2] / 2)
                      ]
            lung_crop = lung_mask[
                        int(crop_center_x - self.crop_size[0] / 2):int(crop_center_x + self.crop_size[0] / 2),
                        int(crop_center_y - self.crop_size[1] / 2):int(crop_center_y + self.crop_size[1] / 2),
                        int(crop_center_z - self.crop_size[2] / 2):int(crop_center_z + self.crop_size[2] / 2)
                        ]
            ct_crop = np.expand_dims(ct_crop, axis=0)
            ct_crop = np.float32(ct_crop / 1.0)  # (1,128,128,64)
            gt_crop = np.expand_dims(gt_crop, axis=0)
            gt_crop = np.concatenate(((1 - gt_crop), gt_crop), axis=0)
            gt_crop = np.float32(gt_crop)  # (1,128,128,64)
            lung_crop = np.expand_dims(lung_crop, axis=0)
            lung_crop = np.concatenate(((1 - lung_crop), lung_crop), axis=0)
            lung_crop = np.float32(lung_crop / 1.0)  # (1,128,128,64)
            ct_crop = ct_crop.transpose((0, 3, 1, 2))
            gt_crop = gt_crop.transpose((0, 3, 1, 2))
            lung_crop = lung_crop.transpose((0, 3, 1, 2))
            return  torch.from_numpy(ct_scans),torch.from_numpy(ct_crop),torch.from_numpy(gt_crop),torch.from_numpy(lung_crop),[crop_center_x,crop_center_y,crop_center_z,ct_dim[2]]

        elif self.phase=='Test':
            ct_crops = []

            scans_pad = np.zeros(shape=(scans.shape[0], scans.shape[1], int(self.crop_size[2] + self.crop_stride[2] * np.ceil((ct_dim[2] - self.crop_size[2]) / self.crop_stride[2]) - ct_dim[2])), dtype=np.uint8)
            gt_pad = np.zeros(shape=(scans.shape[0], scans.shape[1], int(
                self.crop_size[2] + self.crop_stride[2] * np.ceil(
                    (ct_dim[2] - self.crop_size[2]) / self.crop_stride[2]) - ct_dim[2])), dtype=np.uint8)
            scans = np.concatenate((scans, scans_pad), axis=2)
            seg_gt = np.concatenate((seg_gt, gt_pad), axis=2)

            for h in range(int(np.ceil((ct_dim[2] - self.crop_size[2]) / self.crop_stride[2]) + 1)):
                for j in range(int((ct_dim[1] - self.crop_size[1]) / self.crop_stride[1]) + 1):
                    for i in range(int((ct_dim[0] - self.crop_size[0]) / self.crop_stride[0]) + 1):
                        crop_center_x = self.crop_size[0] / 2 + self.crop_stride[0] * i
                        crop_center_y = self.crop_size[1] / 2 + self.crop_stride[1] * j
                        crop_center_z = self.crop_size[2] / 2 + self.crop_stride[2] * h

                        ct_crops.append(scans[
                                        int(crop_center_x - self.crop_size[0] / 2):int(
                                            crop_center_x + self.crop_size[0] / 2),
                                        int(crop_center_y - self.crop_size[1] / 2):int(
                                            crop_center_y + self.crop_size[1] / 2),
                                        int(crop_center_z - self.crop_size[2] / 2):int(
                                            crop_center_z + self.crop_size[2] / 2)
                                        ])

            ct_crops = np.array(ct_crops)  # (150,128,128,64)
            ct_crops = np.float32(ct_crops / 1.0)
            seg_gt_realsize = np.expand_dims(seg_gt[..., 0:ct_dim[2]], axis=0)  # (1,512,512,200)
            seg_gt_realsize = np.concatenate(((1 - seg_gt_realsize), seg_gt_realsize), axis=0)  # (2,512,512,200)
            lung_mask_realsize = np.expand_dims(lung_mask, axis=0)  # (1,512,512,200)
            lung_mask_realsize = np.concatenate(((1 - lung_mask_realsize), lung_mask_realsize),
                                                axis=0)  # (2,512,512,200)
            lung_mask_realsize = np.float32(lung_mask_realsize / 1.0)

            ct_crops = ct_crops.transpose((0, 3, 1, 2))  # (150,64,128,128)
            return torch.from_numpy(ct_scans), torch.from_numpy(ct_crops), torch.from_numpy(seg_gt_realsize), torch.from_numpy(lung_mask_realsize),ct_dim

def get_patch_seg(num_split=6,phase='Train',crop_size=[160,160,64],dataset='ours'):
    root_path='../dataset/'
    ######## segment_label ########
    mask_paths_covid = os.listdir(root_path + 'covid/segment_label')
    mask_paths_covid.sort()
    train_end_covid_segment_new= np.floor(len(mask_paths_covid) )
    train_end_covid_segment = np.floor(len(mask_paths_covid) * 0.75)

    mask_paths_cap = os.listdir(root_path + 'cap/segment_label')
    mask_paths_cap.sort()
    train_end_cap_segment_new = np.floor(len(mask_paths_cap))
    train_end_cap_segment = np.floor(len(mask_paths_cap) * 0.75)
    ct_paths=[]
    lung_path=[]
    mask_paths=[]
    if phase=='Test':
        for i in range(int(train_end_covid_segment),len(mask_paths_covid)):
            mask_paths.append(root_path + 'covid/segment_label/' + mask_paths_covid[i])
            ct_paths.append(root_path + 'covid/data-nii/' + mask_paths_covid[i][:-4]+'.nii.gz')
            lung_path.append(root_path + 'covid/lung-npy/' + mask_paths_covid[i][:-4] + '.npy')
        for i in range(int(train_end_cap_segment),len(mask_paths_cap)):
            mask_paths.append(root_path + 'cap/segment_label/' + mask_paths_cap[i])
            ct_paths.append(root_path + 'cap/data-nii/' + mask_paths_cap[i][:-4]+'.nii.gz')
            lung_path.append(root_path + 'cap/lung-npy/' + mask_paths_cap[i][:-4] + '.npy')
    return Dataset_covid(ct_paths,lung_path,mask_paths, phase, num_split,crop_size,dataset)



def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)

if __name__ == '__main__':
    remove_all_file('temp')

























