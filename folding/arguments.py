from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-npz', '--NPZ', required=True,
                        help='path to npz file storing the predicted restraints')
    parser.add_argument('-fa', '--FASTA', required=True,
                        help='path to FASTA file')
    parser.add_argument("-out", '--OUT', required=True,
                        help="output model (in PDB format)")
    parser.add_argument('-tmp', '--TMPDIR',
                        default='/dev/shm/',
                        help='temp folder to store all the restraints')
    parser.add_argument('-nm', '--nmodels',
                        default=5, type=int,
                        help='number of decoys to generate')
    parser.add_argument('-dcut', '--dcut',
                        default=0.45, type=float,
                        help='cutoff of distance restraints')
    parser.add_argument('-cpu', '--CPU',
                        type=int, default=5,
                        help='number of CPUs to use')
    args = parser.parse_args()
    return args
