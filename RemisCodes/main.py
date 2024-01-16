import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary
from torchvision.utils import make_grid
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import os

from evaluate import evaluate
from unet_model import UNet
from utils import dice_loss, rle_encode

# dirs_train = [f.path for f in os.scandir("/media/Storage3.6TB/SenNet-Segmentation/train")]
# dirs_test  = [f.path for f in os.scandir("/media/Storage3.6TB/SenNet-Segmentation/test")]
dirs_train =  [f.path for f in os.scandir("/media/Storage3.6TB/SenNet-Segmentation/train")][2:]
dirs_test = [[f.path for f in os.scandir("/media/Storage3.6TB/SenNet-Segmentation/train")][0]]
    

batch_size = 6
epochs = 500

amp = True

if __name__=='__main__':
    
    writer = SummaryWriter(log_dir="runs/Jan9-Resize1024-Stride256-ps256/")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_train = ConcatDataset(
        [HiPCTDataset2D_patches(data_dir, reshapeSize=(1024,1024), patch_size=256, stride=256) for data_dir in dirs_train]
    )
    dataset_test = ConcatDataset(
        [HiPCTDataset2D_patches(data_dir, reshapeSize=(1024,1024), patch_size=256, stride=256) for data_dir in dirs_test]
    )
    
    n_train, n_test = len(dataset_train), len(dataset_test)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True,)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True,)

    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.load_state_dict(torch.load("runs/Jan9-Resize1024-Stride256-ps256/checkpoint_model_199.pth"))
    print(GetBatchOfPatches(*next(iter(train_loader)))[0].shape)

    print(summary(model, GetBatchOfPatches(*next(iter(train_loader)))[0].shape))

    model.to(device = device, memory_format=torch.channels_last)
    
    optimizer = optim.RMSprop(model.parameters(), lr=1e-7,
                              weight_decay=1e-9, momentum=0.99, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    criterion = BinaryFocalLoss(gamma=2.0, alpha=0.9) #nn.BCEWithLogitsLoss()

    epoch = 200
    with tqdm(total=epochs,initial=epoch, unit='epoch') as pbar:
        for epoch in range(epoch+1, epochs+1):
            pbar.update(1)
            model.train()
            epoch_loss = 0
            batch_step = 0
        #with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                
                images, masks = GetBatchOfPatches(*batch)
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks = masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    preds = model(images)
                    loss = criterion(preds.squeeze(1), masks.float().squeeze(1))
                    dice = dice_loss(nn.Sigmoid()(preds.squeeze(1)), masks.float().squeeze(1), multiclass=False)

                    writer.add_scalar("Loss/train/"+str(criterion), loss.item(), epoch)
                    writer.add_scalar("Loss/train/DICE loss", dice.item(), epoch)
                    loss += dice
                    
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                writer.add_scalar("Loss/train/"+str(criterion)+"Dice", loss.item(), epoch)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                epoch_loss += loss.item()

                batch_step +=1
                division_step = ((batch_step*batch_size) // n_train)
                if division_step > 0:
                    if epoch % division_step == 0:

                        if True: #epoch % 5 == 4:#max(10, epochs//10) == max(10, epochs//10)-1:

                            n = min(images.shape[0],8)
                            idx = torch.randperm(images.shape[0])[:n]
                            grid = make_grid([*(images[idx].to('cpu')), 
                                              *(masks[idx].to('cpu')), 
                                              *((preds[idx].to('cpu')>0.5).float())], 
                                              nrow=n,
                                             pad_value=1) 
                            writer.add_image("Examples: original (top), gt (middle), preds (bottom)", grid, epoch)
                            #grids = make_grids(images[:n], masks[:n], preds[:n], n_rows=2)
                            # writer.add_image('Examples/original', grids[0], epoch)
                            # writer.add_image('Examples/ground truth', grids[1], epoch)
                            # writer.add_image('Examples/predictions', grids[2], epoch)
                            torch.save(model.state_dict(), os.path.join(writer.log_dir, f'checkpoint_model_{epoch}.pth'))
                    
                        for tag, value in model.named_parameters():
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                writer.add_histogram(f'Weights/{tag}', value.data.cpu(), epoch)
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                writer.add_histogram(f"Gradients/{tag}", value.grad.data.cpu(), epoch)
                        

                        val_score = evaluate(model, test_loader, device, amp)
                        scheduler.step(val_score)
                        writer.add_scalar("Loss/test/dice", val_score, epoch)
                        
                        
    # batch_size=9
    # dataset    = OCTAVesselSegmentationDataset("/home/remi/Desktop/OCTA-Net-OCTA-Vessel-Segmentation-Network/data/ROSE-1/SVC/test/")
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # inputs, masks = next(iter(dataloader))
    # preds = model(inputs.to(device))
    # fig, axs = plot_im_pred_gt(inputs, masks, preds)
    # fig.suptitle(f"Loss {criterion(preds.to(device), masks.to(device))/batch_size}")
    # plt.show()

