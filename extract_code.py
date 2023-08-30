import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import CodeRow
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from vqvae import VQVAE
from torch.utils.data import Dataset
import torch
import os
import SimpleITK as sitk
import numpy as np


def extract_label_from_filename(filename):
    # 假设文件名的格式是 "Brats2021_00000_t1.nii.gz"
    parts = filename.split('_')
    if len(parts) >= 3 and parts[-1].endswith('.nii.gz'):
        label = parts[-2]  # 文件名的倒数第二部分即为标签信息
        return label
    return None

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

class NiFTIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        nii_path = os.path.join(self.root_dir, self.file_list[idx])
        # print(nii_path)

        # data = load_nifti_file(nii_path)
        nii_img = sitk.ReadImage(nii_path)
        # 获取数据和元数据
        nii_data = sitk.GetArrayFromImage(nii_img)
        nrrd_img = sitk.GetImageFromArray(nii_data)
        nrrd_img = resize_image_itk(nrrd_img, (256, 256, 256),
                                    resamplemethod=sitk.sitkLinear)
        nii_data = sitk.GetArrayFromImage(nrrd_img)

        #         nii_data = nii_img.get_fdata()

        #         # 调整体积尺寸
        #         new_size = (256, 256, 256)
        #         current_size = nii_data.shape
        #         scale_factors = (
        #             new_size[0] / current_size[0],
        #             new_size[1] / current_size[1],
        #             new_size[2] / current_size[2]
        #         )
        #         adjusted_volume = ndimage.zoom(nii_data, scale_factors, order=1, mode='constant')
        #         nii_data = adjusted_volume

        #         if self.transform:
        #             nii_data = self.transform(nii_data)

        nii_data = torch.tensor(nii_data, dtype=torch.float32)
        nii_data = nii_data.unsqueeze(0)  # 添加批次维度
        # print(nii_data)
        label = extract_label_from_filename(self.file_list[idx])
        # print(nii_data.shape)
        # print(nii_data)
        # print(label)
        return nii_data, label , self.file_list[idx]

def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)
        #print(pbar)

        for img, _, filename in pbar:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            #print(id_t.shape)
            id_b = id_b.detach().cpu().numpy()
            #print(id_b.shape)

            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    device = 'cuda'

    dataset = NiFTIDataset(root_dir=args.path)
    # dataset = datasets.ImageFolder(args.path, transform=transform)
    loader = DataLoader(
        dataset, batch_size=1,  num_workers=12
    )

    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)
