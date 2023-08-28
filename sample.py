import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist
import nibabel as nib
import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from scipy import ndimage
import SimpleITK as sitk
import os
import nibabel as nib
import nrrd
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk
from glob import glob

def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint)

    model = VQVAE()

    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model
def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

device = "cuda"
print("Loading model")
model = load_model('./checkpoint/vqvae_557.pt',device)

model.eval()
print("Loading data")
nii_img = sitk.ReadImage('./test/BraTS2021_00058_t2.nii.gz')
                # 获取数据和元数据
nii_data = sitk.GetArrayFromImage(nii_img)
nrrd_img = sitk.GetImageFromArray(nii_data)
nrrd_img = resize_image_itk(nrrd_img, (256, 256, 256),
                                            resamplemethod=sitk.sitkLinear)
sample = sitk.GetArrayFromImage(nrrd_img)
nii_direction = nii_img.GetDirection()
nii_origin = nii_img.GetOrigin()
nrrd_file = './sample/BraTS2021_00058_t2.nii.gz'
sitk.WriteImage(nrrd_img, nrrd_file)
                # 扩展维度并复制通道
sample = torch.tensor(sample, dtype=torch.float32)
sample = sample.unsqueeze(0)
sample = sample.unsqueeze(0)

                #data1,label1 = NiFTIDataset('./test').__getitem__(0)
print("Process")
with torch.no_grad():
    out, _ = model(sample.to(device))
    out=out.cpu().numpy()
    #print(out.shape)
    out=out.squeeze(0)
    out=out.squeeze(0)
    #print(out.shape)
out_file=f"sample/{str(1).zfill(5)}_{str(1).zfill(5)}.nii.gz"

out_img=sitk.GetImageFromArray(out)
out_img.SetDirection(nii_direction)
out_img.SetOrigin(nii_origin)

                # 使用 SimpleITK 保存为 nrrd 文件
sitk.WriteImage(out_img, out_file)
print("Complete")
# import argparse
# import sys
# import os
#
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader
#
# from torchvision import datasets, transforms, utils
#
# from tqdm import tqdm
#
# from vqvae import VQVAE
# from scheduler import CycleScheduler
# import distributed as dist
# import nibabel as nib
# import os
# import nibabel as nib
# import numpy as np
# from torch.utils.data import Dataset
# import numpy as np
# import nibabel as nib
# from scipy import ndimage
# import SimpleITK as sitk
# import os
# import nibabel as nib
# import nrrd
# from pathlib import Path
# import SimpleITK as sitk
# import numpy as np
# import SimpleITK as sitk
# from glob import glob
#
# def load_model(checkpoint, device):
#     ckpt = torch.load(os.path.join('checkpoint', checkpoint))
#
#     model = VQVAE()
#
#     if 'model' in ckpt:
#         ckpt = ckpt['model']
#
#     model.load_state_dict(ckpt)
#     model = model.to(device)
#     model.eval()
#
#     return model
# def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
#     resampler = sitk.ResampleImageFilter()
#     originSize = itkimage.GetSize()
#     originSpacing = itkimage.GetSpacing()
#     newSize = np.array(newSize,float)
#     factor = originSize / newSize
#     newSpacing = originSpacing * factor
#     newSize = newSize.astype(np.int)
#     resampler.SetReferenceImage(itkimage)
#     resampler.SetSize(newSize.tolist())
#     resampler.SetOutputSpacing(newSpacing.tolist())
#     resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
#     resampler.SetInterpolator(resamplemethod)
#     itkimgResampled = resampler.Execute(itkimage)
#     return itkimgResampled
#
# device = "cuda"
#
# model = load_model('./checkpoint/vqvae_560.pt',device)
#
# model.eval()
# nii_img = sitk.ReadImage('./test/BraTS2021_00495_t2.nii.gz')
#                 # 获取数据和元数据
# nii_data = sitk.GetArrayFromImage(nii_img)
# nrrd_img = sitk.GetImageFromArray(nii_data)
# nrrd_img = resize_image_itk(nrrd_img, (256, 256, 256),
#                                             resamplemethod=sitk.sitkLinear)
# sample = sitk.GetArrayFromImage(nrrd_img)
# nii_direction = nii_img.GetDirection()
# nii_origin = nii_img.GetOrigin()
# nrrd_file = './sample/BraTS2021_00495_t2.nii.gz'
# sitk.WriteImage(nrrd_img, nrrd_file)
#                 # 扩展维度并复制通道
# sample = torch.tensor(sample, dtype=torch.float32)
# sample = sample.unsqueeze(0)
# sample = sample.unsqueeze(0)
#
#                 #data1,label1 = NiFTIDataset('./test').__getitem__(0)
# with torch.no_grad():
#     out, _ = model(sample.to(device))
#     out=out.cpu().numpy()
#     print(out.shape)
#     out=out.squeeze(0)
#     out=out.squeeze(0)
#     print(out.shape)
# out_file=f"sample/{str(1).zfill(5)}_{str(1).zfill(5)}.nii.gz"
#
# out_img=sitk.GetImageFromArray(out)
# out_img.SetDirection(nii_direction)
# out_img.SetOrigin(nii_origin)
#
#                 # 使用 SimpleITK 保存为 nrrd 文件
# sitk.WriteImage(out_img, out_file)
# import argparse
# import os
#
# import torch
# from torchvision.utils import save_image
# from tqdm import tqdm
#
# from vqvae import VQVAE
# from pixelsnail import PixelSNAIL
#
#
# @torch.no_grad()
# def sample_model(model, device, batch, size, temperature, condition=None):
#     row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
#     cache = {}
#
#     for i in tqdm(range(size[0])):
#         for j in range(size[1]):
#             out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
#             prob = torch.softmax(out[:, :, i, j] / temperature, 1)
#             sample = torch.multinomial(prob, 1).squeeze(-1)
#             row[:, i, j] = sample
#
#     return row
#
#
# def load_model(model, checkpoint, device):
#     ckpt = torch.load(os.path.join('checkpoint', checkpoint))
#
#
#     if 'args' in ckpt:
#         args = ckpt['args']
#
#     if model == 'vqvae':
#         model = VQVAE()
#
#     elif model == 'pixelsnail_top':
#         model = PixelSNAIL(
#             [32, 32],
#             512,
#             args.channel,
#             5,
#             4,
#             args.n_res_block,
#             args.n_res_channel,
#             dropout=args.dropout,
#             n_out_res_block=args.n_out_res_block,
#         )
#
#     elif model == 'pixelsnail_bottom':
#         model = PixelSNAIL(
#             [64, 64],
#             512,
#             args.channel,
#             5,
#             4,
#             args.n_res_block,
#             args.n_res_channel,
#             attention=False,
#             dropout=args.dropout,
#             n_cond_res_block=args.n_cond_res_block,
#             cond_res_channel=args.n_res_channel,
#         )
#
#     if 'model' in ckpt:
#         ckpt = ckpt['model']
#
#     model.load_state_dict(ckpt)
#     model = model.to(device)
#     model.eval()
#
#     return model
#
#
# if __name__ == '__main__':
#     device = 'cuda'
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch', type=int, default=8)
#     parser.add_argument('--vqvae', type=str)
#     parser.add_argument('--top', type=str)
#     parser.add_argument('--bottom', type=str)
#     parser.add_argument('--temp', type=float, default=1.0)
#     parser.add_argument('filename', type=str)
#
#     args = parser.parse_args()
#
#     model_vqvae = load_model('vqvae', args.vqvae, device)
#     model_top = load_model('pixelsnail_top', args.top, device)
#     model_bottom = load_model('pixelsnail_bottom', args.bottom, device)
#
#     top_sample = sample_model(model_top, device, args.batch, [32, 32], args.temp)
#     bottom_sample = sample_model(
#         model_bottom, device, args.batch, [64, 64], args.temp, condition=top_sample
#     )
#
#     decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
#     decoded_sample = decoded_sample.clamp(-1, 1)
#
#     save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))