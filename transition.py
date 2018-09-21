# pylint: disable=W0221, C, R, W1202, E1101, E1102, W0401, W0614
from functions import *
import subprocess
import argparse


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
    parser.add_argument("--dim", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--args", type=str, default="")
    parser.add_argument("--p", type=int, nargs='+', default=[])
    parser.add_argument("--max_factor", type=float)
    parser.add_argument("--n_unsat", type=int, default=1)
    parser.add_argument("--launcher", type=str, default="gpurun")
    parser.add_argument("--fast", type=to_bool, default="False")

    args = parser.parse_args()

    command = ""
    if args.launcher == "gpurun":
        command = "gpurun "
    if args.launcher == "srun":
        command = "srun --partition gpu --qos gpu_free --gres gpu:1 --time 12:00:00 --mem 8G "

    command += "python train.py --log_dir {log_dir} --p {{p}} --dim {{d}} --width {{h}} --depth {depth} --nd_stop {{nd_stop}} ".format(
        log_dir=args.log_dir, depth=args.depth) + args.args

    for p in args.p:
        max_h = find_h(args.max_factor * p, args.depth, args.dim)
        hs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 23, 25, 27, 29, 31, 32, 34, 37, 39, 41, 43, 44, 46, 48, 50, 52, 54, 57, 62, 67, 73, 80, 87, 94, 102, 111, 121, 132, 143, 156, 169, 184, 200, 217, 236, 257, 280, 304, 331, 359, 391, 425, 462, 502, 546, 594, 646, 702, 764, 830, 903]
        hs = [x for x in hs if x <= max_h]
        hs = sorted(hs, reverse=True)

        print(">>> hs={}".format(hs))

        n_unsat = 0
        for h in hs:
            d = args.dim if args.dim else h
            N = d * h + h ** 2 * (args.depth - 1) + h
            cmd = command.format(p=p, h=h, d=d, nd_stop=N // 20 if args.fast else 0)
            print(">>> " + cmd)

            run = subprocess.Popen(cmd.split())
            run.wait()

            desc = {
                "p": p,
                "dim": d,
                "depth": args.depth,
                "width": h,
                "kappa": 0.5,
                "lamda": None,
                "rep": 0,
            }

            run = next(run for run in load_dir(args.log_dir) if run['desc'] == desc)

            if run['last']['train'][0] > 0.1 * run['N']:
                n_unsat += 1
                if n_unsat >= args.n_unsat:
                    break


if __name__ == '__main__':
    main()
