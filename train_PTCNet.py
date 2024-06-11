import argparse
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from monai.data import decollate_batch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset.brats import get_datasets
from loss.dice import EDiceLoss, EDiceLoss_Val
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch, save_checkpoint_final
from ptcnet.PTCNet import PTCNet as PTCNet_seg

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='PTCNet BRATS 2023/2021 Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float, metavar='W', help='weight decay (default: 0)', dest='weight_decay')
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--val', default=1, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int, default=3, help='output channel of network')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="PTCNet/configs/ptcnet_base.yaml", metavar="FILE", help='path to config file')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')


def check_nan_in_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    else:
        print(f"No NaN in {name}")

def forward_hook(module, input, output):
    if isinstance(output, tuple):
        for idx, out in enumerate(output):
            check_nan_in_tensor(out, f"{module.__class__.__name__} output {idx}")
    else:
        check_nan_in_tensor(output, f"{module.__class__.__name__} output")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = torch.cuda.device_count()
    print(f"Working with {world_size} GPUs")
    args.exp_name = "logs_base2023_PTCNet_5_31"
    args.save_folder_1 = pathlib.Path(f"./PTCNet/runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    print("args.save_folder_1:", str(args.save_folder_1))
    args.checkpoint_folder = pathlib.Path(f"./PTCNet/runs/{args.exp_name}/model_1")
    print("args.checkpoint_folder:", str(args.checkpoint_folder))

    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    config = get_config(args)
    model_1 = PTCNet_seg(in_channels=4, out_channels=args.num_classes, embed_dim=96, num_heads=4, mlp_ratio=4.0).to(device)
    #model_1.load_from(config)
    
    if args.resume:
        args.checkpoint = args.checkpoint_folder / "model_best.pth.tar"
        reload_ckpt_bis(args.checkpoint, model_1)
    
    print(f"total number of trainable parameters {count_parameters(model_1)}")
    
    model_file = args.save_folder_1 / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)
    
    criterion = EDiceLoss().to(device)
    criterian_val = EDiceLoss_Val().to(device)
    metric = criterian_val.metric
    print(metric)
    params = model_1.parameters()
    
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    full_train_dataset, l_val_dataset, bench_dataset = get_datasets(args.seed, fold_number=args.fold)
    
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)
    
    print("Train dataset number of batch:", len(train_loader))
    print("Val dataset number of batch:", len(val_loader))
    print("Bench Test dataset number of batch:", len(bench_loader))
    
    best_1 = 0.0
    patients_perf = []
    print("start training now!")
    # for name, module in model_1.named_modules():
    #     module.register_forward_hook(forward_hook)
    
    for epoch in range(args.epochs):
        try:
            ts = time.perf_counter()
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')
            mode = "train" if model_1.training else "val"
            batch_per_epoch = len(train_loader)
            progress = ProgressMeter(batch_per_epoch, [batch_time, data_time, losses_], prefix=f"{mode} Epoch: [{epoch}]")
            end = time.perf_counter()
            metrics = []
            
            for i, batch in enumerate(zip(train_loader)):
                torch.cuda.empty_cache()
                data_time.update(time.perf_counter() - end)
                
                inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["label"].float()
                inputs_S1, labels_S1 = inputs_S1.to(device), labels_S1.to(device)
                
                optimizer.zero_grad()
                
                segs_S1 = model_1(inputs_S1)

                loss_ = criterion(segs_S1, labels_S1)
                
                loss_.backward()
                
                gradient_norm = 0.0
                for param in model_1.parameters():
                    if param.grad is not None:
                        gradient_norm += torch.norm(param.grad.detach()) ** 2
                
                gradient_norm = gradient_norm.sqrt()
                torch.nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=1.0)
                optimizer.step()
                t_writer_1.add_scalar(f"gradients/{mode}{''}",
                                      gradient_norm.item(),
                                      global_step=batch_per_epoch * epoch + i)
                t_writer_1.add_scalar(f"Loss/{mode}{''}", loss_.item(), global_step=batch_per_epoch * epoch + i)
                
                print("train_loss:", loss_.item())
                
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                else:
                    print("NaN in model loss!!")
                
                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)
                if scheduler is not None:
                    scheduler.step()
                
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                progress.display(i)
            
            t_writer_1.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)
            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")
            torch.cuda.empty_cache()
            
            if (epoch + 1) % args.val == 0:
                validation_loss_1, validation_dice, validation_dice_2 = step(val_loader, model_1, criterian_val, metric, epoch, t_writer_1, save_folder=args.save_folder_1, patients_perf=patients_perf,device=device)
                
                t_writer_1.add_scalar(f"SummaryLoss", validation_loss_1, epoch)
                t_writer_1.add_scalar(f"SummaryDice", validation_dice_2, epoch)
                print("validation_loss_1:", validation_loss_1)
                print("validation_dice:", validation_dice)
                print("validation_dice_2:", validation_dice_2)
                print("best_1:", best_1)
                if validation_dice_2 > best_1:
                    print(f"Saving the model with DSC {validation_dice_2}")
                    best_1 = validation_dice_2
                    model_dict = model_1.state_dict()
                    save_checkpoint(dict(epoch=epoch, state_dict=model_dict, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), save_folder=args.save_folder_1)
                
                model_dict_final = model_1.state_dict()
                save_checkpoint_final(dict(epoch=epoch, state_dict=model_dict_final, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), save_folder=args.save_folder_1)
                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")
                torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            break

            
def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    else:
        print(f"No NaN in {name}")            

def step(data_loader, model, criterion: EDiceLoss_Val, metric, epoch, writer, save_folder=None, patients_perf=None,device=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mode = "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(batch_per_epoch, [batch_time, data_time, losses], prefix=f"{mode} Epoch: [{epoch}]")
    end = time.perf_counter()
    metrics = []
    
    for i, val_data in enumerate(data_loader):
        data_time.update(time.perf_counter() - end)
        patient_id = val_data["patient_id"]
        model.eval()
        with torch.no_grad():
            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            print("11111111")
            check_nan(val_inputs, "val_inputs")
            val_outputs = inference(val_inputs, model)
            val_outputs = val_outputs.float()
            val_labels = val_labels.float()
            val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs)]
            segs = val_outputs
            targets = val_labels
            # print(f"val_outputs: {val_outputs}")
            # print(f"val_labels: {val_labels}")
            loss_ = criterion(segs, targets)
            dice_metric(y_pred=val_outputs_1, y=val_labels)
            print("val_labels:", val_labels.shape) #1,3，xxx
            print("val_outputs:", val_outputs.shape) #1,3,xxx
            print("val_outputs max:", val_outputs.max())
            print("val_outputs min:", val_outputs.min())
            print("val_labels max:", val_labels.max())
            print("val_labels min:", val_labels.min())

            # 验证是否存在全零或全负输出
            print("val_outputs mean:", val_outputs.mean())
            print("val_outputs unique values:", val_outputs.unique())
        
        if patients_perf is not None:
            patients_perf.append(dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item()))
        
        writer.add_scalar(f"Loss/{mode}{''}", loss_.item(), global_step=batch_per_epoch * epoch + i)
        print("loss_.item():", loss_.item())
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")
        
        metric_ = metric(segs, targets)
        metrics.extend(metric_)
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        progress.display(i)
    
    save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)
    dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()
    metrics_list = [torch.tensor(dice) for dice in metrics]
    mean_ET = torch.mean(metrics_list[0])
    mean_TC = torch.mean(metrics_list[1])
    mean_WT = torch.mean(metrics_list[2])
    mean_dice = (mean_ET + mean_TC + mean_TC) / 3.0
    mean_dice = mean_dice.item()
    print("mean_dice:", mean_dice)
    
    return losses.avg, dice_values, mean_dice

if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)

