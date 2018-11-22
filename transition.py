# pylint: disable=W0221, C, R, W1202, E1101, E1102, W0401, W0614
from functions import *
import subprocess
import argparse
from itertools import product
import multiprocessing


def find_h(N, L, d):
    assert L >= 1

    if d is None:
        # solve : N = h^2 * L + h
        return round(((N - (N / L) ** 0.5) / L) ** 0.5)

    if L == 1:
        # solve : N = h (d+1)
        return round(N / (d + 1))

    # solve : N = h (d+1) + h^2 * (L-1)
    return round((((d + 1)**2 + 4 * N * (L - 1))**0.5 - (d + 1)) / (2 * (L - 1)))


GLO = [None, None]


def foo(ar):
    p, dim, depth, rep = ar

    global GLO
    args, command = GLO

    max_h = find_h(args.max_factor * p, depth, dim)

    runs = [r() for desc, r in load_dir_functional(args.log_dir) if desc['p'] == p and (desc['dim'] == dim or dim is None) and desc['depth'] == depth and desc['rep'] == rep]
    runs = [r for r in runs if r['last']['train'][0] == 0]
    if len(runs) > 0:
        max_h = min(max_h, min(r['desc']['width'] for r in runs))
        print("runs already done max_h = {}".format(max_h))

    hs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 23, 25, 27, 29, 31, 32, 34, 37, 39, 41, 43, 44, 46, 48, 50, 52, 54, 57, 62, 67, 73, 80, 87, 94, 102, 111, 121, 132, 143, 156, 169, 184, 200, 217, 236, 257, 280, 304, 331, 359, 391, 425, 462, 502, 546, 594, 646, 702, 764, 830, 903]
    hs = [x for x in hs if x <= max_h]
    hs = sorted(hs, reverse=True)

    print(">>> hs={}".format(hs))

    n_unsat = 0
    for h in hs:
        d = dim if dim else h
        N = d * h + h ** 2 * (depth - 1) + h
        cmd = command.format(p=p, h=h, d=d, depth=depth, nd_stop=N // 20 if args.fast else 0)
        print(">>> " + cmd)

        run = subprocess.Popen(cmd.split())
        run.wait()

        desc = {
            "p": p,
            "dim": d,
            "depth": depth,
            "width": h,
            "kappa": 1,
            "rep": rep,
        }

        run = next(r() for desc_, r in load_dir_functional(args.log_dir) if desc_ == desc)

        if run['last']['train'][0] > 0.1 * run['N']:
            n_unsat += 1
            if n_unsat >= args.n_unsat:
                break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--n_parallel", type=int, default=1)
    parser.add_argument("--dim", type=int, nargs='+')
    parser.add_argument("--depth", type=int, nargs='+', required=True)
    parser.add_argument("--args", type=str, default="")
    parser.add_argument("--p", type=int, nargs='+', required=True)
    parser.add_argument("--rep", type=int, nargs='+', default=[0])
    parser.add_argument("--max_factor", type=float)
    parser.add_argument("--n_unsat", type=int, default=1)
    parser.add_argument("--launcher", type=str, default="")
    parser.add_argument("--fast", type=to_bool, default="False")

    args = parser.parse_args()

    if args.dim is None:
        args.dim = [None]

    command = "{} ".format(args.launcher)
    command += "python train.py --log_dir {log_dir} --p {{p}} --dim {{d}} --width {{h}} --depth {{depth}} --nd_stop {{nd_stop}} ".format(
        log_dir=args.log_dir) + args.args

    global GLO
    GLO[0] = args
    GLO[1] = command

    with multiprocessing.Pool(args.n_parallel) as pool:
        list(pool.imap_unordered(foo, product(args.p, args.dim, args.depth, args.rep)))


if __name__ == '__main__':
    main()
