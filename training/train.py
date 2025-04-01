import warnings
import os
import random
import sys
import torch

import numpy as np
import pyxis.torch as pxt

from torch.utils.data import DataLoader, random_split
from time import time
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter

base_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
sys.path.insert(0, base_dir)

from dataset import MSADataset
from network.RNAformer import DistPredictor

from utils_training import *

parser = ArgumentParser(description='Training of trRosettaRNA', add_help=False,
                        formatter_class=RawTextHelpFormatter)
group_help = parser.add_argument_group('help information')
group_help.add_argument("-h", "--help", action="help", help="show this help message and exit")

# common arguments
group = parser.add_argument_group('arguments for training script')
group.add_argument('npz_dir',
                   type=str,
                   help=f'{bcolors.BOLD}(required) Location of folder storing the NPZ files of training set{bcolors.RESET}')
group.add_argument('out_dir',
                   type=str,
                   help=f"{bcolors.BOLD}(required) Location of folder to save the trained model to{bcolors.RESET}")

group.add_argument('-lst', '--lst',
                   type=str, default=None,
                   help='Text file storing the list of training samples (default:use all samples in `npz_dir`)')
group.add_argument('-init_lr', '--init_lr',
                   type=float, default=5e-4,
                   help='Initial learning rate (default:0.0005)')
group.add_argument('-batch_size', '--batch_size',
                   type=int, default=1,
                   help='Batch size for training (default:1)')
group.add_argument('-early_stopping', '--early_stopping',
                   action='store_true', default=True,
                   help='Whether to stop early if val loss cannot drop for several epochs (default:True)')
group.add_argument('-max_epochs', '--max_epochs',
                   type=int, default=30,
                   help='Maximum number of epochs to train (default:30)')
group.add_argument('-crop_size', '--crop_size',
                   type=int, default=200,
                   help='Sequence with residues more than this number will be randomly cropped for training (default:300)')
group.add_argument('-gpu', '--gpu',
                   type=int, default=0,
                   help='Use which gpu. cpu only if set to -1 (default:0)')
group.add_argument('-cpu', '--cpu',
                   type=int, default=2,
                   help='Number of cpus to use (default:2)')
group.add_argument('-warning', '--warning',
                   action='store_true',
                   help='Whether to print warnings (default: False)')


def train(dataloader, is_training=False):
    model.train(is_training)
    for param in model.parameters():
        param.requires_grad = is_training

    metrics = {'loss': [], 'dist_corr': []}
    n_OOM = 0

    for i, data in enumerate(dataloader):
        if not data: continue
        # if i==5:break
        msa = data['aln'].long().to(device)
        ss = data['ss'].float().to(device)
        res_id = data['idx'].long().to(device)
        conf_mask1d = data['conf_mask'].long().to(device)
        conf_mask2d = data['conf_mask_2d'].long().to(device)
        native_geom = data['labels']

        try:
            ## predict
            num_recycle = random.choice(range(4)) if is_training else 3
            msa_cutoff = 300 if is_training else 800
            pred_geoms = model(msa, ss, res_id=res_id, msa_cutoff=msa_cutoff, num_recycle=num_recycle,
                               is_training=is_training)['geoms']

            ## losses
            loss = geometry_loss(pred_geoms, native_geom, device, conf_mask1d, conf_mask2d)

            loss_np = float(loss.detach())
            if np.isnan(loss_np):
                warn = f'{bcolors.RED}loss is nan!{bcolors.RESET}'
                warnings.warn(warn)
                continue

            if is_training:
                loss.backward()
                if i % args.batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            metrics['loss'].append(loss_np)
            if not is_training:
                pred_c3 = pred_geoms['distance']["C3'"].detach().cpu().numpy()
                native_c3 = data['distance']["C3'"].detach().cpu().numpy()
                dcorr = dist_corr(pred_c3, native_c3)
                metrics['dist_corr'].append(dcorr)


        except RuntimeError as exception:
            if "out of memory" in str(exception):
                n_OOM += 1
                if args.warning:
                    warn = f'{bcolors.RED}OOM for MSA with shape {msa.size()}!{bcolors.RESET}'
                    warnings.warn(warn)
                try:
                    del pred_geoms, loss
                except UnboundLocalError:
                    pass
                empty_cache()
                continue
            else:
                raise exception

        if args.warning:
            sample_name = 'train' if is_training else 'val'
            total = num_samples if is_training else len(val_set)
            if is_training:
                print(
                    f"\rEpoch: {epoch + 1}, {sample_name}: {i + 1}/{total}, loss: {loss_np:.2f}   ",
                    end='')
            else:
                print(
                    f"\rEpoch: {epoch + 1}, {sample_name}: {i + 1}/{total}, loss: {loss_np:.2f}, distance accuracy: {dcorr:.3f}  ",
                    end='')
    if args.warning:
        print('\r' + ' ' * 80)
    return metrics, n_OOM


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    torch.set_num_threads(args.cpu)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_path = f'{args.out_dir}/state_dict'
    config_path = f'{args.out_dir}/config'
    log_file = f'{args.out_dir}/training.log'
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(config_path, exist_ok=True)

    config = {'lr': args.init_lr, 'n_blocks': 18, 'channels': 48}
    save_to_json(config, f'{config_path}/model_1.json')

    if args.lst is None:
        full_lst = [f[:-4] for f in os.listdir(args.npz_dir) if f.endswith('.npz')]
    else:
        full_lst = open(args.lst).read().splitlines()

    val_lst = random.sample(full_lst, 100)
    train_lst = list(set(full_lst) - set(val_lst))

    train_set = MSADataset(train_lst, npz_dir=args.npz_dir, lengthmax=args.crop_size, warning=args.warning)
    val_set = MSADataset(val_lst, npz_dir=args.npz_dir, lengthmax=args.crop_size, random_=False, warning=args.warning)

    num_samples = len(train_set)

    sampler = None
    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=sampler is None, sampler=sampler, num_workers=0)
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    if args.warning:
        print(f'\n{bcolors.BLUE}train:{len(train_set)}{bcolors.RESET}')
        print(f'{bcolors.BLUE}val:{len(val_set)}{bcolors.RESET}')
        print(f'\n{bcolors.BOLD}total_epochs:{args.max_epochs}, learning_rate:{args.init_lr}\n')

    # Define network models
    ## initialize RNAformer
    if args.warning:
        print(f'{bcolors.GREEN}initialize an RNAformer model{bcolors.RESET}')

    model = DistPredictor(dim_2d=config['channels'], layers_2d=config['n_blocks']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 18, 20, 25], gamma=.5,
                                                     last_epoch=-1)

    max_corr = -1
    steps_without_enhancing = 0
    endurance = 5

    # training
    print(f'\n{bcolors.BOLD}{bcolors.BLUE}--------------------train start-------------------------{bcolors.RESET}',
          end='\n')
    for epoch in range(0, args.max_epochs):

        train_metrics, n_OOM_train = train(train_dataloader, is_training=True)
        val_metrics, n_OOM_val = train(val_dataloader, is_training=False)
        scheduler.step()

        ## output log
        log_info = f'\r{bcolors.BOLD}{bcolors.HEADER}---------------------Epoch: {epoch + 1}----------------------{bcolors.RESET}\n' + \
                   f'learning rate: {optimizer.param_groups[0]["lr"]}\n' \
                   f'train loss: {np.mean(train_metrics["loss"]):.2f}\n' \
                   f'val loss: {np.mean(val_metrics["loss"]):.2f}, distance accuracy: {np.mean(val_metrics["dist_corr"]):.2f}{bcolors.RESET}\n'

        if args.warning:
            print(log_info)
        with open(log_file, 'a') as f:
            f.write(log_info.replace(bcolors.CYAN, '').replace(bcolors.BOLD, '').replace(bcolors.RESET, '').replace(
                bcolors.HEADER, ''))

        ## save model if achieves higher distance accuracy on validation set
        if np.mean(val_metrics['dist_corr']) > max_corr:
            steps_without_enhancing = 0
            max_corr = np.mean(val_metrics["dist_corr"])
            ckpt_dict = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(ckpt_dict, f'{checkpoint_path}/model_1.pth.tar')
            device_dict = dict((ind, int(str(optimizer.state_dict()['state'][ind]['exp_avg'].device)[-1])) for ind in
                               optimizer.state_dict()['state'])
            save_to_json(device_dict, f'{checkpoint_path}/opt_devices.json')
        elif epoch > 10:
            steps_without_enhancing += 1
            if steps_without_enhancing == endurance:
                log_info = f'{bcolors.RED}-------------------early stopped!----------------------{bcolors.RESET}'
                if args.warning:
                    print(log_info)
                with open(log_file, 'a') as f:
                    f.write(log_info.replace(bcolors.RED, '').replace(bcolors.RESET, ''))
                break

    print(f'{bcolors.BOLD}{bcolors.BLUE}---------------------train end-------------------------{bcolors.RESET}')
