import tempfile

import glob

import string

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path

pkg_dir = os.path.abspath(str(Path(__file__).parent))
sys.path.insert(0, pkg_dir)
sys.path.insert(1, f'{pkg_dir}/network')
from utils import *
from network.RNAformer import DistPredictor
from network.config import n_bins, obj

parser = ArgumentParser()
parser.add_argument('-i', '--msa', help='input MSA file')
parser.add_argument('-ss', '--ss_file', default=None,
                    help='the custom secondary structure (SS) file (SPOT-RNA will be run if this file is not provided)')
parser.add_argument('-ss_fmt', '--ss_fmt', default='dot_bracket', choices=['spot_prob', 'dot_bracket', 'ct'],
                    help='the format of custom SS file; spot_prob/dot_bracket(default)/ct')
parser.add_argument('-o', '--npz', help='output NPZ file')
parser.add_argument('-mdir', '--model_pth', default=f'params/model_1', help='pretrained params directory')
parser.add_argument('-nrows', '--nrows', default=1000, type=int, help='maximum number of rows in the MSA repr.')
parser.add_argument('-gpu', '--gpu', type=str, default='-1', help='use which gpu')
parser.add_argument('-cpu', '--cpu', type=int, default=2, help='number of CPUs to use')
args = parser.parse_args()


def predict(model, msa, ss_, window=150, shift=50):
    if ss_.shape[0] != msa.shape[-1]:
        raise ValueError(f'ss length {ss_.shape[0]}, msa length {msa.shape[1]}!')
    with torch.no_grad():
        feat = torch.from_numpy(msa).to(device)
        ss_ = torch.from_numpy(ss_).to(device)
        L = msa.shape[-1]
        res_id = torch.arange(L, device=device).view(1, L)
        if L > 300:  # predict by crops for long RNA
            pred_dict = {
                'contact': torch.zeros((L, L), device=device),
                'distance': {k: torch.zeros((L, L, n_bins['2D']['distance']), device=device) for k in
                             obj['2D']['distance']},
            }

            count_1d = torch.zeros((L)).to(device)
            count_2d = torch.zeros((L, L)).to(device)
            #
            grids = np.arange(0, L - window + shift, shift)
            ngrids = grids.shape[0]
            print("ngrid:     ", ngrids)
            print("grids:     ", grids)
            print("windows:   ", window)

            idx_pdb = torch.arange(L).long().view(1, L)
            for i in range(ngrids):
                for j in range(i, ngrids):
                    start_1 = grids[i]
                    end_1 = min(grids[i] + window, L)
                    start_2 = grids[j]
                    end_2 = min(grids[j] + window, L)
                    sel = np.zeros((L)).astype(np.bool_)
                    sel[start_1:end_1] = True
                    sel[start_2:end_2] = True

                    input_msa = feat[:, sel]
                    input_ss = ss_[sel][:, sel]
                    mask = torch.sum(input_msa == 4, dim=-1) < .7 * sel.sum()  # remove too gappy sequences

                    input_msa = input_msa[mask]
                    input_idx = idx_pdb[:, sel]
                    input_res_id = res_id[:, sel]

                    print("running crop: %d-%d/%d-%d" % (start_1, end_1, start_2, end_2), input_msa.shape)
                    pred_gemos = model(input_msa, input_ss, res_id=input_res_id.to(device),
                                       msa_cutoff=args.nrows)['geoms']
                    weight = 1
                    sub_idx = input_idx[0].cpu()
                    sub_idx_2d = np.ix_(sub_idx, sub_idx)
                    count_2d[sub_idx_2d] += weight
                    count_1d[sub_idx] += weight

                    for k in obj['2D']:
                        if k == 'contact':
                            pred_dict['contact'][sub_idx_2d] += weight * pred_gemos['contact']
                        else:
                            for a in obj['2D'][k]:
                                pred_dict[k][a][sub_idx_2d] += weight * pred_gemos[k][a]
            for k in obj['2D']:
                if k == 'contact':
                    pred_dict['contact'] /= count_2d
                else:
                    for a in obj['2D'][k]:
                        if pred_dict[k][a].size().__len__() == 3:
                            pred_dict[k][a] /= count_2d[:, :, None]
                        else:
                            pred_dict[k][a] /= count_2d
        else:
            pred_dict = model(feat, ss_, res_id=res_id.to(device), msa_cutoff=args.nrows)['geoms']

    for l in pred_dict:
        if isinstance(pred_dict[l], dict):
            for k in pred_dict[l]:
                pred_dict[l][k] = pred_dict[l][k].cpu().detach().numpy()
        else:
            pred_dict[l] = pred_dict[l].cpu().detach().numpy()

    return pred_dict


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(args.cpu)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    py = sys.executable

    out_dir = os.path.dirname(os.path.abspath(args.npz))
    os.makedirs(out_dir, exist_ok=True)

    cwd = os.getcwd()
    msa = parse_a3m(args.msa, limit=20000)

    if args.ss_file is None:
        # predict SS by SPOT-RNA
        print('predict SS by SPOT-RNA')
        tmp_dir = tempfile.TemporaryDirectory(prefix=out_dir + '/')
        spot_out_dir = tmp_dir.name
        if not os.path.isfile(f'{spot_out_dir}/seq.fasta'):
            os.system(f'head -n 2 {args.msa} >{spot_out_dir}/seq.fasta')
        spot_py = py.replace('trRNA', 'spot_venv')
        if os.path.isdir(f'{pkg_dir}/SPOT-RNA'):
            os.system(f'mv {pkg_dir}/SPOT-RNA {pkg_dir}/SPOT_RNA')
        os.chdir(f'{pkg_dir}/SPOT_RNA')
        if os.path.isdir(f'utils'):
            os.system(f'mv utils utils_spot')
        spot_script = open('SPOT-RNA.py').read()
        with open('SPOT-RNA.py', 'w') as f:
            f.write(spot_script.replace(f'from utils.', f'from utils_spot.'))

        os.system(
            f'nohup {spot_py} SPOT-RNA.py --inputs {spot_out_dir}/seq.fasta --outputs {spot_out_dir}/ --gpu {args.gpu} >{out_dir}/spot.log 2>&1')
        os.chdir(cwd)

        prob_files = glob.glob(f'{spot_out_dir}/*.prob')
        if len(prob_files) == 0: raise ValueError(
            f'Fails to predict SS! Please refer to {out_dir}/spot.log to see what happened.')
        ss = np.loadtxt(prob_files[0])
        if (np.tril(ss) == 0).all() or (np.triu(ss) == 0).all():
            ss += ss.T
    else:
        if args.ss_fmt == 'dot_bracket':
            ss = ss2mat(open(args.ss_file).read().rstrip().splitlines()[-1].strip())
        elif args.ss_fmt == 'ct':
            ss = parse_ct(args.ss_file, length=len(msa[0]))
        elif args.ss_fmt == 'spot_prob':
            ss = np.loadtxt(args.ss_file)
            ss += ss.T
        if len(ss) != len(msa[0]):
            raise ValueError(f'The SS shape {ss.shape} mismatches the MSA shape {msa.shape}!')

    print('predict geometries')
    config = json.load(open(f'{args.model_pth}/config/model_1.json', 'r'))

    model = DistPredictor(dim_2d=config['channels'], layers_2d=config['n_blocks'])

    model_ckpt = torch.load(f'{args.model_pth}/models/model_1.pth.tar', map_location=device)
    model.load_state_dict(model_ckpt)
    model.eval()
    model.to(device)

    pred = predict(model, msa, ss)

    print('done!')
    print('saving......')
    np.savez_compressed(args.npz, **pred)
