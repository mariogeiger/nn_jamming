from functions import *
import subprocess
import argparse
import math


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--args", type=str, default="")
    parser.add_argument("--p", type=int, nargs='+', default=[])
    parser.add_argument("--max_factor", type=float)

    args = parser.parse_args()

    command = "gpurun python train.py --log_dir {log_dir} --p {{p}} --dim {{h}} --width {{h}} --depth {depth} ".format(
        log_dir=args.log_dir, depth=args.depth) + args.args

    for p in args.p:
        max_width = (args.max_factor * p / args.depth) ** 0.5
        hs = list(map(int, intlogspace(2, 20000, 100)))
        hs = [x for x in hs if x <= max_width]
        hs = sorted(hs, reverse=True)

        print(">>> hs={}".format(hs))

        for h in hs:
            cmd = command.format(p=p, h=h)
            print(">>> " + cmd)
            run = subprocess.Popen(cmd.split())
            run.wait()

            run = list(load_dir(args.log_dir))[-1]
            if run['last']['train'][0] > 0.5 * args.depth * h ** 2:
                break


if __name__ == '__main__':
    main()



