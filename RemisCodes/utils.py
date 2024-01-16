import torch
import numpy as np 
from torch import Tensor
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
import torch.nn.functional as F
from torch import nn

from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset
import os
from tqdm import tqdm

from numba import jit


from skimage.util import view_as_windows

# Run-Length Encode and Decode

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.9):
        """
        alpha = 0.5 -> balanced importance
        alpha < 0.5 -> more importance to background
        alpha > 0.5 -> more importance to foreground
        """
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        ### Code taken from chatGPT, not approved
        epsilon=1e-8
        y_pred = torch.clamp(y_pred, epsilon, 1-epsilon)

        p_t = y_true * y_pred + (1-y_true)*(1-y_pred)
        alpha_factor = 1.0
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        focal_loss = -self.alpha * alpha_factor * modulating_factor * torch.log(torch.clamp(p_t, epsilon, 1.0-epsilon))
        return torch.mean(focal_loss)


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def make_grids(inputs, masks, preds, n_rows=3):
    inputs_grid = make_grid(inputs.to('cpu'), nrow=n_rows, padding=5, normalize=True)
    masks_grid  = make_grid(masks.to('cpu'), nrow=n_rows, padding=5)
    preds_grid  = make_grid((preds>0.5).float().to('cpu'), nrow=n_rows, padding=5, normalize=True)
    return inputs_grid, masks_grid, preds_grid

def plot_im_pred_gt(inputs, masks, preds):

    fig, axs = plt.subplots(1,3, figsize=(10,4))
    for ax, title, grid in zip(axs.flat, ["Inputs", "Ground truth", "Predictions"], make_grids(inputs, masks,preds)):
        ax.imshow(grid.numpy().transpose((1,2,0)))
        ax.set_title(title)
        ax.set_axis_off()

    return fig, axs


def log_gradient_histograms(writer, model, step):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and 'bias' not in name:
            try:
                writer.add_histogram(f'grad/{name}', param.grad.detach(), global_step=step)
            except ValueError:
                print(name, param.grad.detach())
                raise ValueError("The histogram is empty.")

def _PatchImage(img, size):  
    ## Non overlapping patches with stride == size
    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(img)
    if len(img.shape)==3: # 2D image
        padH, padW = [size - dim%size for dim in img.shape[1:]]
        paddedImg = F.pad(img, (0, 0,
                                int(padH/2), padH - int(padH/2), 
                                int(padW/2), padW - int(padW/2)))
        paddedImg = paddedImg.unfold(1, size, size).unfold(2, size, size)

    elif len(img.shape)==4: # 2D image
        padH, padW, padD = [size - dim%size for dim in img.shape[1:]]
        paddedImg = F.pad(img, (
            0, 0,
            int(padH/2), padH - int(padH/2), 
            int(padW/2), padW - int(padW/2),
            int(padD/2), padD - int(padD/2)
        ))
        paddedImg = paddedImg.unfold(1, size, size).unfold(2, size, size).unfold(3, size, size)

    return paddedImg.reshape([-1] + [size for _ in img.shape[1:]])
            
def CreatePatchesDataset(input_data_dir:str, output_data_dir:str, size:int, stride=tuple()):
    assert isinstance(size, int), f"'size' must be an integer, got {type(size)} instead."
    assert isinstance(stride, (int, tuple)), f"'size' must be an integer or tuple[int], got {type(stride)} instead."
    assert input_data_dir!=output_data_dir, "Input and output data directories are the same. Stopping to avoid deleting data."

    imgs_dir = os.path.join(input_data_dir, "images")
    labs_dir = os.path.join(input_data_dir, "labels")
    out_imgs_dir = os.path.join(output_data_dir, "images")
    out_labs_dir = os.path.join(output_data_dir, "labels")

    depth = len(os.listdir(imgs_dir)) # The depth dimension
    w,h = np.array(Image.open(next(f.path for f in os.scandir(imgs_dir)))).shape # width and height

    nSlices = depth//size # Number of slices to use per patch

    for inFolder, outFolder in zip((imgs_dir, labs_dir), (out_imgs_dir, out_labs_dir)):
                

        patchNum = 0
        for i in tqdm(range(nSlices), desc=f"Scanning '{inFolder}'"):
            files = [f.path for f in os.scandir(inFolder)][i*size:(i+1)*size]
            im = torch.Tensor(np.stack(list(map(lambda x: np.array(Image.open(x)), files)), axis=0).astype(int)[np.newaxis,...])
            im = _PatchImage(im, size).to('cpu', torch.int8)

            for p in map(lambda x: x.squeeze().clone(), im):
                torch.save(p, os.path.join(outFolder, f"{patchNum:010d}.tens"))
                patchNum+=1
            del im

def split_image_into_patches(image, patch_size, overlap):
    patchs = view_as_windows(image, (patch_size, patch_size),                            
                             step=(patch_size-overlap, patch_size-overlap))
    return patchs.reshape(-1, patch_size, patch_size), patchs.shape

def split_batch_into_patches(imgs:torch.Tensor, patch_size:int, overlap:int):
    patchs = view_as_windows(imgs.numpy(), (1, patch_size, patch_size),                            
                             step=(1, patch_size-overlap, patch_size-overlap))
    return patchs.reshape(patchs.shape[0], -1, patch_size, patch_size)[0], patchs.shape


def reconstruct_image_from_patches(patches, outimg_shape, patch_size, overlap):
    full_image = np.zeros(outimg_shape, dtype=float)
    patch_count = np.zeros(outimg_shape, dtype=float)
    for i in range(patches.shape[0]):
        start_row = (i * (patch_size-overlap))
        end_row = start_row + patch_size
        for j in range(patches.shape[1]):
            start_col = j * (patch_size-overlap)
            end_col = start_col + patch_size
            full_image[start_row:end_row, start_col:end_col, ...] += patches[i,j,...]
            patch_count[start_row:end_row, start_col:end_col, ...] += 1
    return full_image/np.where(patch_count!=0, patch_count, 1)


class HiPCTDataset2D(Dataset):
    
    def __init__(self, data_dir, applyTransform:bool=True, reshapeSize:tuple=(512,512)):
        
        self.labels_dir = os.path.abspath(os.path.join(data_dir, 'labels'))
        self.img_dir = os.path.abspath(os.path.join(data_dir, 'images'))
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(size=reshapeSize, antialias=True)
        self.applyTransform = applyTransform

    def transform(self, image, labels):
       
        image, labels = image.astype(float), labels.astype(float)

        image  = self.to_tensor(image)
        labels = (self.to_tensor(labels)>0.0).to(dtype=torch.float16)

        image  = self.resize(image)
        labels = self.resize(labels)
        image  = ((image-image.min())/(image.max()-image.min()))#.astype(float) # Normalize to 0-1
        # labels = labels.astype(float) #((labels-labels.min())/(labels.max()-labels.min())).astype(float)

        if not self.applyTransform:
            return image, labels
        
        if np.random.random() > 0.5:
            image  = VF.hflip(image)
            labels = VF.hflip(labels)

        if np.random.random() > 0.5:
            image  = VF.vflip(image)
            labels = VF.vflip(labels)

        # if np.random.random() > 0.5:
        #     angle  = np.random.random()*45
        #     image  = VF.rotate(image, angle=angle)
        #     labels = VF.rotate(labels, angle=angle)

        return image, labels
                
    def __len__(self):
        return len([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])

    def __getitem__(self, idx:int):

        img_path = os.path.join(self.img_dir, f"{idx:04d}.tif")
        labels_path = os.path.join(self.labels_dir, f"{idx:04d}.tif")                                                                                                                            
        image, labels = self.transform(plt.imread(img_path), plt.imread(labels_path))
        return image, labels
    

class HiPCTDataset2D_patches(Dataset):
    
    def __init__(self, data_dir, applyTransform:bool=True, 
                 patch_size:int=128,
                 stride:int=64,
                 reshapeSize:tuple=(512,512)):
        
        self.labels_dir = os.path.abspath(os.path.join(data_dir, 'labels'))
        self.img_dir = os.path.abspath(os.path.join(data_dir, 'images'))
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(size=reshapeSize, antialias=True)
        self.applyTransform = applyTransform
        self.ps = patch_size
        self.st = stride
        self.reshapeSize = reshapeSize
        self.numPatches = int(np.prod([(s - (self.ps - 1))/self.st + 1 for s in self.reshapeSize])) 
        
    def createPatches(self, x):
        if len(x.shape)==2:
            x = x.unsqueeze(0)
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        patches = x.unfold(2, self.ps, self.st).unfold(3, self.ps, self.st)
        patches = patches.contiguous().view(-1,1, self.ps, self.ps)
        return patches

    def recreateOriginal(self, patches):
        unfold_shape = patches.size()
        patches = patches.view(-1, self.ps, self.ps)
        x = patches.view(unfold_shape)
        #x = x.permute(0,2,1,3).contiguous()
        x = x.view(1, *self.reshapeSize)
        return x


    def transform(self, image, labels):
       
        image, labels = image.astype(float), labels.astype(float)

        image  = self.to_tensor(image)
        labels = (self.to_tensor(labels)>0.0).to(dtype=torch.float16)

        image  = self.resize(image)
        labels = self.resize(labels)
        image  = ((image-image.min())/(image.max()-image.min()))#.astype(float) # Normalize to 0-1
        # labels = labels.astype(float) #((labels-labels.min())/(labels.max()-labels.min())).astype(float)

        image  = self.createPatches(image)
        labels = self.createPatches(labels)

        if not self.applyTransform:
            return image, labels
        
        if np.random.random() > 0.5:
            image  = VF.hflip(image)
            labels = VF.hflip(labels)

        if np.random.random() > 0.5:
            image  = VF.vflip(image)
            labels = VF.vflip(labels)

        # if np.random.random() > 0.5:
        #     angle  = np.random.random()*45
        #     image  = VF.rotate(image, angle=angle)
        #     labels = VF.rotate(labels, angle=angle)

        return image, labels
                
    def __len__(self):
        return len([f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))])

    def __getitem__(self, idx:int):

        img_path = os.path.join(self.img_dir, f"{idx:04d}.tif")
        labels_path = os.path.join(self.labels_dir, f"{idx:04d}.tif")                                                                                                                            
        image, labels = self.transform(plt.imread(img_path), plt.imread(labels_path))
        return image, labels

def GetBatchOfPatches(imgs, labels):
    return imgs.view(-1,1,*imgs.shape[-2:]), labels.view(-1,1,*labels.shape[-2:])
