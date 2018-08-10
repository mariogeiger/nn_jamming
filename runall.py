# pylint: disable=W0221, C, R, W1202, E1101

import argparse
import subprocess
import time
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--n_parallel", type=int, default=1)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--factor_step", type=float, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--args", type=str, default="")

    args = parser.parse_args()

    command = "gpurun python train.py --log_dir {log_dir} --p {{p}} --dim {dim} --width {width} --depth {depth} ".format(
        log_dir=args.log_dir, dim=args.dim, width=args.width, depth=args.depth) + args.args

    if args.depth > 0:
        N = (args.dim + 1) * args.width + (args.depth - 1) * args.width ** 2  # weight
    else:
        N = args.dim  # weight

    ps = sorted(list(set(map(int, np.linspace(args.factor * N, (args.factor + args.factor_step) * N, num=args.num)))))
    print("N={} => p's = {}".format(N, ps))

    running = []

    for p in ps:
        while len(running) >= args.n_parallel:
            running = [x for x in running if x.poll() is None]
            time.sleep(2)
        if os.path.isfile("stop"):
            break

        cmd = command.format(p=p)
        running.append(subprocess.Popen(cmd.split()))
        print(cmd)

    for x in running:
        x.wait()


if __name__ == '__main__':
    main()