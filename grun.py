#!/usr/bin/python3
#pylint: disable=C
"""
Wait for available GPUs to execute a command
"""
import argparse
import time
import os
import sys
import subprocess
import random
import GPUtil
import numpy as np
import glob


def check_pid(pid):
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--n", type=int, default=1, help="number of GPUs needed")
    parser.add_argument("command", metavar='CMD', type=str, nargs='?')

    argv = []
    i = 1
    while i < len(sys.argv) and sys.argv[i][:1] == "-":
        argv += sys.argv[i:i + 2]
        i += 2
    args = parser.parse_args(argv)
    args.command = sys.argv[i:]

    directory = os.path.join(os.environ['HOME'], '.grun')
    if not os.path.exists(directory):
        os.makedirs(directory)

    while True:
        try:
            GPUs = GPUtil.getGPUs()
        except FileNotFoundError:
            print("grun: no gpus")
            return

        maxLoad = 0.5
        maxMemory = 0.5
        GPUs = [gpu for gpu in GPUs if gpu.load < maxLoad and gpu.memoryUtil < maxMemory]

        running = [f.split('/')[-1].split('_') for f in glob.glob("/home/*/.grun/*")]
        running = [(int(pid), list(map(int, ids.split(',')))) for pid, ids in running]
        running = [(pid, ids) for pid, ids in running if check_pid(pid)]

        p = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE)
        out = p.stdout.read().decode('UTF-8')
        procs = [int(x.split()[1]) for x in out.split('Processes:')[-1].split('\n') if "iB" in x]
        for gpu in GPUs:
            gpu.nproc = max(len([1 for pid, ids in running if gpu.id in ids]), len([1 for p in procs if p == gpu.id]))

        maxProc = 3
        GPUs = [gpu for gpu in GPUs if gpu.nproc < maxProc]

        if len(GPUs) >= args.n:
            break

        print("grun: waiting for gpus...")
        time.sleep(5)

    GPUs = sorted(GPUs, key=lambda gpu: gpu.nproc)
    GPUs = GPUs[:args.n]

    # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    deviceIds = ",".join(map(str, (gpu.id for gpu in GPUs)))
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceIds

    p = subprocess.Popen(args.command)

    f = os.path.join(directory, "{}_{}".format(p.pid, deviceIds))
    open(f, 'w').close()

    try:
        p.wait()
    except KeyboardInterrupt:
        print("grun: kill process")
        p.kill()
    os.remove(f)


if __name__ == "__main__":
    main()
