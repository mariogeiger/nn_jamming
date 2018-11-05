# pylint: disable=W0221, C, R, W1202, E1101
import argparse
import subprocess
import time
import os
from itertools import product
from functions import parse_kmg


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--n_parallel", type=int, default=1)
    parser.add_argument("--dim", type=int, nargs='+')
    parser.add_argument("--N", type=parse_kmg, nargs='+', required=True)
    parser.add_argument("--depth", type=int, nargs='+')
    parser.add_argument("--p", type=str, nargs='+', required=True)
    parser.add_argument("--rep", type=int, nargs='+', default=[0])
    parser.add_argument("--args", type=str, default="")
    parser.add_argument("--launcher", type=str, default="")  # srun --partition gpu --qos gpu --gres gpu:1 --time 3-00:00:00 --mem 12G

    args = parser.parse_args()

    command = "{} ".format(args.launcher)
    command += "python train.py --log_dir {log_dir} --p {{p}} --dim {{dim}} --width {{width}} --depth {{depth}} --rep {{rep}} {args}".format(
        log_dir=args.log_dir, args=args.args)

    running = []

    for p, dim, N, depth, rep in product(args.p, args.dim, args.N, args.depth, args.rep):
        while len(running) >= args.n_parallel:
            running = [x for x in running if x.poll() is None]
            time.sleep(2)
        if os.path.isfile("stop"):
            break

        width = find_h(N, depth, dim)

        cmd = command.format(p=p, dim=dim if dim else width, width=width, rep=rep, depth=depth)

        running.append(subprocess.Popen(cmd.split()))
        print(cmd)
        time.sleep(2)

    for x in running:
        x.wait()


if __name__ == '__main__':
    main()
