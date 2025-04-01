import tempfile

import glob

import sys
import os, time
from pathlib import Path
from folding.arguments import get_args
from folding.utils_cst import npz2cst
from folding.utils_ros import fold_from_cst

if __name__ == '__main__':
    args = get_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.OUT)), exist_ok=True)

    tmpdir = tempfile.TemporaryDirectory(prefix=args.TMPDIR + '/')
    args.tmpdir = tmpdir.name
    print('temp folder:     ', tmpdir.name)

    # parse npz into rosetta-format restraint files
    npz2cst(args)

    # perform energy minimization
    fold_from_cst(args)
