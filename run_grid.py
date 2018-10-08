# pylint: disable=W0221, C, R, W1202, E1101
import argparse
import subprocess
import time
import os
from itertools import product


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--n_parallel", type=int, default=1)
    parser.add_argument("--dim", type=int, nargs='+', required=True)
    parser.add_argument("--width", type=int, nargs='+', required=True)
    parser.add_argument("--depth", type=int, nargs='+', required=True)
    parser.add_argument("--p", type=str, nargs='+', required=True)
    parser.add_argument("--rep", type=int, nargs='+', default=[0])
    parser.add_argument("--args", type=str, default="")
    parser.add_argument("--launcher", type=str, default="gpurun")

    args = parser.parse_args()

    command = ""
    if args.launcher == "gpurun":
        command = "gpurun "
    if args.launcher == "srun":
        command = "srun --partition gpu --qos gpu --gres gpu:1 --time 3-00:00:00 --mem 12G "

    command += "python train.py --log_dir {log_dir} --p {{p}} --dim {{dim}} --width {{width}} --depth {{depth}} --rep {{rep}} {args}".format(
        log_dir=args.log_dir, args=args.args)

    running = []

    for p, dim, width, depth, rep in product(args.p, args.dim, args.width, args.depth, args.rep):
        while len(running) >= args.n_parallel:
            running = [x for x in running if x.poll() is None]
            time.sleep(2)
        if os.path.isfile("stop"):
            break

        cmd = command.format(p=p, dim=dim, width=width, depth=depth, rep=rep)
        running.append(subprocess.Popen(cmd.split()))
        print(cmd)
        time.sleep(2)

    for x in running:
        x.wait()


if __name__ == '__main__':
    main()
