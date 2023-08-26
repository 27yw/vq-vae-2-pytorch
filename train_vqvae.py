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
import torch
import numpy as np
import nibabel as nib
from torchvision import utils
import os
import nibabel as nib
import nrrd
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk
from glob import glob


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
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


def extract_label_from_filename(filename):
    # 假设文件名的格式是 "Brats2021_00000_t1.nii.gz"
    parts = filename.split('_')
    if len(parts) >= 3 and parts[-1].endswith('.nii.gz'):
        label = parts[-2]  # 文件名的倒数第二部分即为标签信息
        return label
    return None


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
        return nii_data, label


#     def __getitem__(self, idx):
#         nii_path = os.path.join(self.root_dir, self.file_list[idx])
#         nii_img = nib.load(nii_path)
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
#         # 扩展维度并复制通道
#         nii_data = torch.tensor(nii_data, dtype=torch.float32)
#         nii_data = nii_data.unsqueeze(0).repeat(3, 1, 1, 1)
#         label = extract_label_from_filename(self.file_list[idx])
#         print(nii_data.shape)
#         return nii_data, label

def load_nifti_file(file_path):
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()  # Get the voxel data as a numpy array
    # print(data)
    return torch.tensor(data, dtype=torch.float32)


def train(epoch, loader, model, optimizer, scheduler, device):
    torch.cuda.empty_cache()
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)
        print(img.shape)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()
        #     for i, (img, label) in enumerate(loader):
        #         model.zero_grad()

        #         img = img.to(device)

        #         out, latent_loss = model(img)
        #         recon_loss = criterion(out, img)
        #         latent_loss = latent_loss.mean()
        #         loss = recon_loss + latent_loss_weight * latent_loss
        #         loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )
            if i % 100 == 0:
                model.eval()
                image_path = "./test/BraTS2021_00621_t2.nii.gz"
                image_path_resize = '/root/autodl-tmp/data/sample/BraTS2021_00621_t2.nii.gz'

                sitk_image = sitk.ReadImage(image_path)
                sitk_image = resize_image_itk(sitk_image, (256, 256, 256), resamplemethod=sitk.sitkLinear)
                sitk.WriteImage(sitk_image, image_path_resize)

                # Convert the SimpleITK image to a numpy array
                # numpy_image = sitk.GetArrayFromImage(sitk_image)
                dataset_sample = NiFTIDataset("./test")
                loader_sample = DataLoader(dataset_sample, batch_size=1, num_workers=12)
                for i, (img_sample, label) in enumerate(loader_sample):

                    print(i)
                    if i == 1:
                        # Modify this part to use your fixed sample
                        sample = img_sample.to(device)  # Load your fixed sample tensor here

                        with torch.no_grad():
                            out, _ = model(sample)

                        # Convert the PyTorch tensor to a SimpleITK image
                        out_image = sitk.GetImageFromArray(
                            out.cpu().numpy())  # Move the tensor to CPU before conversion
                        output_filename = f"/root/autodl-tmp/data/sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.nii.gz"
                        sitk.WriteImage(out_image, output_filename)

                        model.train()



                # 可以跑的一个版本
                # model.eval()
                #
                # nii_img = sitk.ReadImage('./test/BraTS2021_00621_t2.nii.gz')
                # # 获取数据和元数据
                # nii_data = sitk.GetArrayFromImage(nii_img)
                # nrrd_img = sitk.GetImageFromArray(nii_data)
                # nrrd_img = resize_image_itk(nrrd_img, (256, 256, 256),
                #                             resamplemethod=sitk.sitkLinear)
                # sample = sitk.GetArrayFromImage(nrrd_img)
                # nii_direction = nii_img.GetDirection()
                # nii_origin = nii_img.GetOrigin()
                # nrrd_file = '/root/autodl-tmp/sample/BraTS2021_00621_t2.nii.gz'
                # sitk.WriteImage(nrrd_img, nrrd_file)
                # # 扩展维度并复制通道
                # sample = torch.tensor(sample, dtype=torch.float32)
                # sample = sample.unsqueeze(0)  # 有这一行，我只加了一个维度，那conv3d怎么传进去4d的？
                # sample = sample.unsqueeze(0)  # 下午那个效果还可以的没有这一行
                # # data1,label1 = NiFTIDataset('./test').__getitem__(0)
                # with torch.no_grad():
                #     out, _ = model(sample.to(device))
                #     out = out.cpu().numpy()
                #     # print(out.shape)
                #     out = out.squeeze(0)  # 下午那个效果还可以的没有这一行
                #     out = out.squeeze(0)  # 下午那个效果还可以的没有这一行
                #     # print(out.shape)
                # out_file = f"/root/autodl-tmp/sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.nii.gz"
                #
                # # img_file=f"/root/autodl-tmp/sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_img.nii.gz"
                # # img=img.cpu().numpy()
                # # img=img.squeeze(0)
                # # img=img.squeeze(0)
                # # img=sitk.GetImageFromArray(img)
                # # img.SetDirection(nii_direction)
                # # img.SetOrigin(nii_origin)
                # # sitk.WriteImage(img, img_file)
                #
                # out_img = sitk.GetImageFromArray(out)
                # out_img.SetDirection(nii_direction)
                # out_img.SetOrigin(nii_origin)
                #
                # # 使用 SimpleITK 保存为 nrrd 文件
                # sitk.WriteImage(out_img, out_file)
                # model.train()


#                 model.eval()
#                 nii_img = sitk.ReadImage('./test/BraTS2021_00495_t2.nii.gz')
#                 # 获取数据和元数据
#                 nii_data = sitk.GetArrayFromImage(nii_img)
#                 nrrd_img = sitk.GetImageFromArray(nii_data)
#                 nrrd_img = resize_image_itk(nrrd_img, (256, 256, 256),
#                                             resamplemethod=sitk.sitkLinear)
#                 sample = sitk.GetArrayFromImage(nrrd_img)
#                 nii_direction = nii_img.GetDirection()
#                 nii_origin = nii_img.GetOrigin()
#                 nrrd_file = './sample/BraTS2021_00495_t2.nii.gz'
#                 sitk.WriteImage(nrrd_img, nrrd_file)
#                 sample = torch.tensor(sample, dtype=torch.float32)
#                 sample = sample.unsqueeze(0)  # 添加批次维度
#                 sample = sample.unsqueeze(0)
#                 with torch.no_grad():
#                     out, _ = model(sample.to(device))
#                 out_file=f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.nii.gz"
#                 out_img=sitk.GetImageFromArray(out)
#                 out_img.SetDirection(nii_direction)
#                 out_img.SetOrigin(nii_origin)

#                 # 使用 SimpleITK 保存为 nrrd 文件
#                 sitk.WriteImage(out_img, out_file)

#                 sample = img[:sample_size]
#                 print(sample)
#                 with torch.no_grad():
#                     out, _ = model(sample)

#                 # Convert PyTorch tensors to NumPy arrays
#                 sample_np = sample.cpu().numpy()
#                 out_np = out.cpu().numpy()

#                 # Create Niigz images from NumPy arrays
#                 sample_img = nib.Nifti1Image(sample_np, affine=None)  # Use the inverse affine matrix
#                 out_img = nib.Nifti1Image(out_np, affine=None)  # Use the inverse affine matrix

#                 # Save the Niigz images
#                 nib.save(sample_img, f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_sample.nii.gz")
#                 nib.save(out_img, f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_output.nii.gz")

#             if i % 100 == 0:
#                 model.eval()

#                 sample = img[:sample_size]
#                 #sample = sample.repeat(1, 3, 1, 1, 1)  # 重复扩展通道数
#                 with torch.no_grad():
#                     out, _ = model(sample)


#                 utils.save_image(
#                     torch.cat([sample, out], 0),
#                     f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
#                     nrow=sample_size,
#                     normalize=True,
#                     range=(-1, 1),
#                 )
#                 # sample = img[:sample_size]
#                 #
#                 nii_img = sitk.ReadImage('./test_resize/BraTS2021_00495_t2.nii.gz')
#     # # 获取数据和元数据
#                 nii_data = sitk.GetArrayFromImage(nii_img)
#                 nii_direction = nii_img.GetDirection()
#                 nii_origin = nii_img.GetOrigin()
#     #             nrrd_file = './sample/BraTS2021_00495_t2.nii.gz'
#     #             #sitk.WriteImage(nrrd_img, nrrd_file)
#                 nii_data = torch.tensor(nii_data, dtype=torch.float)
#                 nii_data = nii_data.unsqueeze(0)
#                 with torch.no_grad():
#                     out, _ = model(nii_data.to(device))
#                 out_file=f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.nii.gz"
#                 out_img=sitk.GetImageFromArray(out.cpu().numpy())
#                 out_img.SetDirection(nii_direction)
#                 out_img.SetOrigin(nii_origin)

#                 # 使用 SimpleITK 保存为 nrrd 文件
#                 sitk.WriteImage(out_img, out_file)
#
# checkpoint_path = f"sample/volume_checkpoint_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.pt"
# torch.save({
#     'sample_volume': sample,
#     'output_volume': out,
# }, checkpoint_path)


# def adjust_volume_size(volume, new_size):
#     current_size = volume.shape
#     scale_factors = (
#         new_size[0] / current_size[0],
#         new_size[1] / current_size[1],
#         new_size[2] / current_size[2]
#     )
#
#     # 使用线性插值调整体积尺寸
#     adjusted_volume = ndimage.zoom(volume, scale_factors, order=1, mode='constant')
#
#     return adjusted_volume

def main(args):
    device = "cuda"
    torch.cuda.empty_cache()
    args.distributed = dist.get_world_size() > 1

    # transform = transforms.Compose([
    #     transforms.Lambda(lambda x: adjust_volume_size(x, (256, 256, 256))),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5]),
    # ])
    dataset = NiFTIDataset(root_dir=args.path)
    # dataset = datasets.ImageFolder(args.path, transform=transform)
    # sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=1 // args.n_gpu,  num_workers=12
    )

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
