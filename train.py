"""
@author: Originally by Francesco La Rosa
         Adapted by Vatsal Raina, Nataliia Molchanova
"""

import argparse
import os
import torch
from torch import nn
from monai.data import decollate_batch
from monai.transforms import Compose, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
import numpy as np
import random
from metrics import dice_metric
from data_load import get_train_dataloader, get_val_dataloader
import wandb
from os.path import join as pjoin
from metrics import *
from model import *
import time
from config import setup_config

setup_config()

parser = argparse.ArgumentParser(description='Get all command line arguments.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# trainining
parser.add_argument('--frozen_learning_rate', type=float, default=-1, help='Specify the initial learning rate')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify the initial learning rate')
parser.add_argument('--seg_loss_weight', type=float, default=1, help='Specify the weight of the segmentation loss')
parser.add_argument('--heatmap_loss_weight', type=float, default=100, help='Specify the weight of the heatmap loss')
parser.add_argument('--offsets_loss_weight', type=float, default=10, help='Specify the weight of the offsets loss')
parser.add_argument('--n_epochs', type=int, default=300, help='Specify the number of epochs to train for')
parser.add_argument('--path_model', type=str, default=None, help='Path to pretrained model')
# initialisation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
# data
parser.add_argument('--data_dir', type=str, default=os.environ["DATA_ROOT_DIR"], help='Specify the path to the data files directory')

parser.add_argument('--I', nargs='+', default=['FLAIR'], choices=['FLAIR', 'phase_T2starw', 'mag_T2starw',\
                                                                  'MPRAGE_reg-T2starw_T1w', 'T1map', 'UNIT1'])

parser.add_argument('--apply_mask', type=str, default=None, 
        help='The name of the folder containing the masks you want to the images to be applied to.')
parser.add_argument('--cp_factor', type=int, default=0, 
			help="number of times each object has been copied for the copy paste data augmentation strategy")

parser.add_argument('--save_path', type=str, default=os.environ["MODELS_ROOT_DIR"],
                    help='Specify the path to the save directory')

parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--cache_rate', default=1.0, type=float)
# logging
parser.add_argument('--val_interval', type=int, default=5, help='Validation every n-th epochs')
parser.add_argument('--threshold', type=float, default=0.4, help='Probability threshold')

parser.add_argument('--wandb_project', type=str, default='WMLIS', help='wandb project name')
parser.add_argument('--name', default="idiot without a name", help='Wandb run name')
parser.add_argument('--force_restart', default=False, action='store_true', help="force the training to restart at 0 even if a checkpoint was found")

VAL_AMP = True
roi_size = (96, 96, 96)

def load_checkpoint(model, optimizer, filename):
    start_epoch = 0
    print(f"=> Loading checkpoint '{filename}'")
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    wandb_run_id = checkpoint['wandb_run_id']
    print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{filename}'\n")

    return model, optimizer, start_epoch, wandb_run_id

def check_paths(args):
    from os.path import exists as pexists
    assert pexists(args.data_dir), f"Directory not found {args.data_dir}"
    #assert pexists(args.bm_path), f"Directory not found {args.bm_path}"
    assert pexists(args.save_path), f"Directory not found {args.save_path}"
    if args.path_model:
        assert pexists(args.path_model), f"Warning: File not found {args.path_model}"


def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi_size,
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        raise Exception("CUDA WAS NOT DETECTED !!!")
        return torch.device('cpu')


post_trans = Compose(
    [AsDiscrete(argmax=True, to_onehot=2)]
)


def min_max_grad(model):

    full_grads = [x.grad for x in model.parameters()]
    maxi = 0
    max_grad = 0
    mini = 1e20
    min_grad = 1e20
    for grad in full_grads:
        if(not(grad is None)):
            maxi = torch.max(torch.abs(grad))
            max_grad = max(max_grad,maxi)    
            mini = torch.min(torch.abs(grad))
            min_grad = min(min_grad,mini) 
    return min_grad, max_grad


def main(args):
    check_paths(args)
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    set_determinism(seed=seed_val)

    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    save_dir = f'{args.save_path}/{args.name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
    
    flr = args.learning_rate if args.frozen_learning_rate < 0 else args.frozen_learning_rate

    # Initialize model
    checkpoint_filename = os.path.join(save_dir,'checkpoint.pth.tar')
    if os.path.exists(checkpoint_filename) and not args.force_restart:
        model = PanopticDeepLab3D(in_channels=len(args.I), num_classes=2).to(device)
        
        checkpoint = torch.load(checkpoint_filename)
        
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        first_layer_params = model.a_block1.conv1.parameters()
        rest_of_model_params = [p for p in model.parameters() if p not in first_layer_params]
        
        optimizer = torch.optim.Adam([{'params': first_layer_params, 'lr': args.learning_rate},
            {'params': rest_of_model_params, 'lr': flr}], weight_decay=0.0005) #momentum=0.9,
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        wandb_run_id = checkpoint['wandb_run_id']
        wandb.init(project=args.wandb_project, mode="online", name=args.name, resume="must", id=wandb_run_id)
        
        # Initialize scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        lr_scheduler.load_state_dict(checkpoint["scheduler"])

        print(f"\nResuming training: (epoch {checkpoint['epoch']})\nLoaded checkpoint '{checkpoint_filename}'\n")
        
    else:
        if args.path_model is not None:
            print(f"Retrieving pretrained model from {args.path_model}")
            model = get_pretrained_model(args.path_model, len(args.I))
        else:
            print(f"Initializing new model with {len(args.I)} input channels")
            model = PanopticDeepLab3D(in_channels=len(args.I), num_classes=2)
            #torch.manual_seed(1)
            #for layer in model.children():
            #    if isinstance(layer, nn.Linear):
            #        nn.init.xavier_uniform_(layer.weight)
            #        nn.init.zeros_(layer.bias)

            #torch.manual_seed(seed_val)
        model.to(device)
        first_layer_params = model.a_block1.conv1.parameters()
        rest_of_model_params = [p for p in model.parameters() if p not in first_layer_params]
        #optimizer = torch.optim.Adam([{'params': first_layer_params, 'lr': args.learning_rate},
        #     {'params': rest_of_model_params, 'lr': flr}], weight_decay=0.0005) #momentum=0.9,
        optimizer = torch.optim.SGD([
                {'params': first_layer_params, 'lr': args.learning_rate},
                {'params': rest_of_model_params, 'lr': flr}
            ], lr=args.learning_rate, weight_decay=0.0005)
        
        start_epoch = 0
        wandb.login()
        wandb.init(project=args.wandb_project, mode="online", name=args.name)
        wandb_run_id = wandb.run.id
        
        # Initialize scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # Initialize dataloaders
    train_loader = get_train_dataloader(data_dir=args.data_dir, 
                                        num_workers=args.num_workers, 
                                        I=args.I, 
                                        cache_rate=args.cache_rate,
                                        apply_mask=args.apply_mask,
                                        cp_factor=args.cp_factor)
    val_loader = get_val_dataloader(data_dir=args.data_dir, 
                                    num_workers=args.num_workers,
                                    I=args.I, 
                                    cache_rate=args.cache_rate,
                                    apply_mask=args.apply_mask)
    
    # Initialize losses
    loss_function_dice = DiceLoss(to_onehot_y=True,
                                  softmax=True, sigmoid=False,
                                  include_background=False)
    loss_function_mse = nn.MSELoss()
    loss_function_l1 = nn.SmoothL1Loss()
    
    
    # Initialize other variables and metrics
    act = nn.Softmax(dim=1)
    epoch_num = args.n_epochs
    val_interval = args.val_interval
    threshold = args.threshold
    gamma_focal = 2.0
    dice_weight = 0.5
    focal_weight = 1.0
    seg_loss_weight = args.seg_loss_weight
    heatmap_loss_weight = args.heatmap_loss_weight
    offsets_loss_weight = args.offsets_loss_weight
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    best_metric_nDSC, best_metric_epoch_nDSC = -1, -1
    best_metric_DSC, best_metric_epoch_DSC = -1, -1
    epoch_loss_values, metric_values_nDSC, metric_values_DSC = [], [], []
    
    # Initialize scaler
    scaler = torch.cuda.amp.GradScaler()
    step_print = 1

    import nibabel as nib
    
    ''' Training loop '''
    for epoch in range(start_epoch, epoch_num):
        start_epoch_time = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        epoch_loss_ce = 0
        epoch_loss_dice = 0
        epoch_loss_seg = 0
        epoch_loss_mse = 0
        epoch_loss_l1 = 0
        step = 0
        
        ex = epoch
        done = True

        for batch_data in train_loader:
            n_samples = batch_data["image"].size(0)
            for m in range(0, batch_data["image"].size(0), args.batch_size):
                step += args.batch_size
                inputs, labels, center_heatmap, offsets = (
                    batch_data["image"][m:(m + 2)].to(device),
                    batch_data["label"][m:(m + 2)].type(torch.LongTensor).to(device),
                    batch_data["center_heatmap"][m:(m + 2)].to(device),
                    batch_data["offsets"][m:(m + 2)].to(device))

                with torch.cuda.amp.autocast():
                    semantic_pred, center_pred, offsets_pred = model(inputs)
                    
                    if not done:
                        nib.save(nib.Nifti1Image(inputs[0].detach().cpu().numpy().squeeze().astype(np.float32), affine=np.eye(4)), f"image_{ex}.nii.gz")
                        nib.save(nib.Nifti1Image(labels[0].detach().cpu().numpy().squeeze().astype(np.float32), affine=np.eye(4)), f"labels_{ex}.nii.gz")
                        nib.save(nib.Nifti1Image(center_heatmap[0].detach().cpu().numpy().squeeze().astype(np.float32), affine=np.eye(4)), f"center_{ex}.nii.gz")
                        nib.save(nib.Nifti1Image(offsets[0].detach().cpu().numpy().squeeze().astype(np.float32), affine=np.eye(4)), f"offsets_{ex}.nii.gz")
                        nib.save(nib.Nifti1Image(semantic_pred[0].detach().cpu().numpy().squeeze().astype(np.float32), affine=np.eye(4)), f"pred-labels_{ex}.nii.gz")
                        nib.save(nib.Nifti1Image(offsets_pred[0].detach().cpu().numpy().squeeze().astype(np.float32), affine=np.eye(4)), f"pred-offsets_{ex}.nii.gz")

                    done = True

                    ### SEGMENTATION LOSS ###
                    # Dice loss
                    dice_loss = loss_function_dice(semantic_pred, labels)
                    # Focal loss
                    ce_loss = nn.CrossEntropyLoss(reduction='none')
                    ce = ce_loss(semantic_pred, torch.squeeze(labels, dim=1))
                    pt = torch.exp(-ce)
                    loss2 = (1 - pt) ** gamma_focal * ce
                    focal_loss = torch.mean(loss2)
                    segmentation_loss = dice_weight * dice_loss + focal_weight * focal_loss
                    
                    ### COM PREDICTION LOSS ###
                    mse_loss = loss_function_mse(center_pred, center_heatmap)

                    ### COM REGRESSION LOSS ###
                    # Disregard voxels outside of the GT segmentation
                    
                    offset_loss_weights_matrix = labels.expand_as(offsets_pred)
                    offset_loss = loss_function_l1(offsets_pred, offsets) * offset_loss_weights_matrix
                    if offset_loss_weights_matrix.sum() > 0:
                        l1_loss = offset_loss.sum() / offset_loss_weights_matrix.sum()
                    else:
                        l1_loss = offset_loss.sum() * 0

                    ### TOTAL LOSS ###
                    loss = (seg_loss_weight * segmentation_loss) + (heatmap_loss_weight * mse_loss) + (offsets_loss_weight * l1_loss)

                epoch_loss += loss.item()
                epoch_loss_ce += focal_loss.item()
                epoch_loss_dice += dice_loss.item()
                epoch_loss_seg += segmentation_loss.item()
                epoch_loss_mse += mse_loss.item()
                epoch_loss_l1 += l1_loss.item()

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
          
                if step % 100 == 0:
                    elapsed_time = time.time() - start_epoch_time
                    step_print = int(step / args.batch_size)
                    print(
                        f"{step_print}/{(len(train_loader) * n_samples) // (train_loader.batch_size * args.batch_size)}, train_loss: {loss.item():.4f}" + \
                        f"(elapsed time: {int(elapsed_time // 60)}min {int(elapsed_time % 60)}s)") 

        epoch_loss /= step_print
        epoch_loss_dice /= step_print
        epoch_loss_ce /= step_print
        epoch_loss_seg /= step_print
        epoch_loss_mse /= step_print
        epoch_loss_l1 /= step_print
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()

        min_grad, max_grad = min_max_grad(model)

        wandb.log(
            {'Training Loss/Total Loss': epoch_loss, 'Training Segmentation Loss/Dice Loss': epoch_loss_dice, 
                'Training Segmentation Loss/Focal Loss': epoch_loss_ce,
                'Training Loss/Segmentation Loss': epoch_loss_seg, 'Training Loss/Center Prediction Loss': epoch_loss_mse,
                'Training Loss/Offsets Loss': epoch_loss_l1, 'Learning rate': current_lr, 
				'Gradients/Min Gradient': min_grad, 'Gradients/Max Gradient': max_grad}, 
            step=epoch)
        
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'wandb_run_id': wandb_run_id,
            'scheduler': lr_scheduler.state_dict()
        }, checkpoint_filename)

        ##### Validation #####


        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                nDSC_list = []
                for val_data in val_loader:
                    val_inputs, val_labels, val_heatmaps, val_offsets_pred, val_bms = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                        val_data["center_heatmap"].to(device),
                        val_data["offsets"].to(device),
                        val_data["brain_mask"].squeeze().cpu().numpy()
                    )

                    val_semantic_pred, val_center_pred, val_offsets = inference(val_inputs, model)

                    for_dice_outputs = [post_trans(i) for i in decollate_batch(val_semantic_pred)]

                    dice_metric(y_pred=for_dice_outputs, y=val_labels)

                    val_semantic_pred = act(val_semantic_pred)[:, 1]
                    val_semantic_pred = torch.where(val_semantic_pred >= threshold, torch.tensor(1.0).to(device),
                                                    torch.tensor(0.0).to(device))
                    val_semantic_pred = val_semantic_pred.squeeze().cpu().numpy()
                    nDSC = dice_norm_metric(val_labels.squeeze().cpu().numpy()[val_bms == 1], val_semantic_pred[val_bms == 1])
                    nDSC_list.append(nDSC)

                torch.cuda.empty_cache()
                del val_inputs, val_labels, val_semantic_pred, val_heatmaps, val_offsets_pred, for_dice_outputs, val_bms # , thresholded_output, curr_preds, gts , val_bms
                metric_nDSC = np.mean(nDSC_list)
                metric_DSC = dice_metric.aggregate().item()
                wandb.log({'nDSC Metric/val': metric_nDSC, 'DSC Metric/val': metric_DSC}, step=epoch)
                metric_values_nDSC.append(metric_nDSC)
                metric_values_DSC.append(metric_DSC)

                if metric_nDSC > best_metric_nDSC:# and epoch > 5:
                    best_metric_nDSC = metric_nDSC
                    best_metric_epoch_nDSC = epoch + 1
                    save_path = os.path.join(save_dir, f"best_nDSC_{args.name}_seed{args.seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model for nDSC")

                if metric_DSC > best_metric_DSC:# and epoch > 5:
                    best_metric_DSC = metric_DSC
                    best_metric_epoch_DSC = epoch + 1
                    save_path = os.path.join(save_dir, f"best_DSC_{args.name}_seed{args.seed}.pth")
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model for DSC")

                print(f"current epoch: {epoch + 1} current mean normalized dice: {metric_nDSC:.4f}"
                      f"\nbest mean normalized dice: {best_metric_nDSC:.4f} at epoch: {best_metric_epoch_nDSC}"
                      f"\nbest mean dice: {best_metric_DSC:.4f} at epoch: {best_metric_epoch_DSC}"
                      )

                dice_metric.reset()


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
