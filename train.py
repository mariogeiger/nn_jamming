# pylint: disable=W0221, C, R, W1202, E1101, E1102, W0401, W0614
import torch
import os
from shutil import copyfile
import argparse
import logging
from itertools import count
import copy
import numpy as np
import time_logging
from functions import *
from fire import FIRE


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_dir", type=str, required=True)

    parser.add_argument("--dataset", default="random")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--rep", type=int, default=0)

    parser.add_argument("--optimizer", choices={"sgd", "adam", "adam0", "fire", "adam_rlrop"}, default="adam0")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--n_steps_max", type=int, default=int(1e7))
    parser.add_argument("--compute_hessian", type=to_bool, default="True")
    parser.add_argument("--save_hessian", type=to_bool, default="False")
    parser.add_argument("--checkpoints", type=int, nargs='+', default=[])
    parser.add_argument("--noise", type=float, default=0)

    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--lamda", type=float)

    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--n_steps_lr_decay", type=int)
    parser.add_argument("--lr_decay_factor", type=float)
    parser.add_argument("--min_learning_rate", type=float)
    parser.add_argument("--rlrop_cooldown", type=float)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_steps_bs_grow", type=int)
    parser.add_argument("--bs_grow_factor", type=float)

    args = parser.parse_args()

    if args.optimizer == "adam_rlrop":
        if args.learning_rate is None:
            args.learning_rate = 1e-3
        if args.min_learning_rate is None:
            args.min_learning_rate = 1e-7
        if args.batch_size is None:
            args.batch_size = args.p
        if args.rlrop_cooldown is None:
            args.rlrop_cooldown = 10
    if args.optimizer == "adam0":
        if args.learning_rate is None:
            args.learning_rate = 1e-4
        if args.n_steps_lr_decay is None:
            args.n_steps_lr_decay = 2.5e5
        if args.lr_decay_factor is None:
            args.lr_decay_factor = 10
        if args.min_learning_rate is None:
            args.min_learning_rate = 1e-7
        if args.batch_size is None:
            args.batch_size = args.p
    if args.optimizer == "adam":
        if args.learning_rate is None:
            args.learning_rate = 1e-4
        if args.n_steps_lr_decay is None:
            args.n_steps_lr_decay = 1e5
        if args.lr_decay_factor is None:
            args.lr_decay_factor = 2
        if args.min_learning_rate is None:
            args.min_learning_rate = 1e-6
        if args.batch_size is None:
            args.batch_size = args.p
    if args.optimizer == "fire":
        if args.learning_rate is None:
            args.learning_rate = 1e-1
        if args.n_steps_lr_decay is None:
            args.n_steps_lr_decay = 1e5
        if args.lr_decay_factor is None:
            args.lr_decay_factor = 2
        if args.min_learning_rate is None:
            args.min_learning_rate = 1e-3
        if args.batch_size is None:
            args.batch_size = args.p

    return args


def init(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    try:
        os.mkdir(args.log_dir)
    except FileExistsError:
        pass

    desc = {
        "p": args.p,
        "dim": args.dim,
        "depth": args.depth,
        "width": args.width,
        "kappa": args.kappa,
        "lamda": args.lamda,
        "rep": args.rep,
    }

    logger = logging.getLogger("default")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(args))

    if desc in [run['desc'] for run in load_dir(args.log_dir)]:
        logger.info("{} skiped".format(repr(desc)))
        return None

    logger.info(desc)

    seed = torch.randint(10000, (), dtype=torch.long).item()
    trainset, testset = get_dataset(args.dataset, args.p, args.dim, seed, device)

    model = Model(args.dim, args.width, args.depth, args.kappa, args.lamda)
    model.to(device)

    model.type(torch.float64)

    print("N={}".format(model.N))

    scheduler = None
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=0)
    if args.optimizer == "adam" or args.optimizer == "adam0":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.optimizer == "fire":
        optimizer = FIRE(model.parameters(), dt_max=args.learning_rate, a_start=1 - args.momentum)
    if args.optimizer == "adam_rlrop":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=0, verbose=True, threshold=-1, threshold_mode="rel", cooldown=args.rlrop_cooldown, min_lr=args.min_learning_rate)

    return model, trainset, testset, logger, optimizer, scheduler, device, desc, seed


def train(args, model, trainset, testset, logger, optimizer, scheduler, device, desc, seed):
    noise = args.noise

    measure_points = set(intlogspace(1, args.n_steps_max, 150, with_zero=True, with_end=True))
    dynamics = []

    checkpoints = []

    time_1 = time_logging.start()

    batch_size = args.batch_size
    loader = simple_loader(*trainset, batch_size)

    bins = np.logspace(-9, 4, 130)
    bins = np.concatenate([[-1], bins])

    step = 0
    while True:
        if step > args.n_steps_max:
            break

        if step % 100 == 0:
            print("{:.2f}%      ".format(step / args.n_steps_max * 100), end="\r")

        if step in measure_points or step % 1000 == 0:
            data = {}
            dynamics.append(data)

            data['step'] = step
            data['train'] = error_loss_grad(model, *trainset)
            logger.info("({}) [{}] train={:d} ({:.1f}%), {:.2g}, |Grad|={:.2g}".format(
                    desc['p'],
                    step,
                    data['train'][0],
                    100 * data['train'][0] / args.p,
                    data['train'][1],
                    data['train'][2]
                )
            )
            if testset is not None:
                data['test'] = error_loss_grad(model, *testset)

            if args.optimizer == "adam_rlrop":
                noise /= optimizer.param_groups[0]["lr"]
                scheduler.step(data['train'][1])
                noise *= optimizer.param_groups[0]["lr"]

            data['batch_size'] = batch_size
            data['optimizer'] = {
                'state': simplify(optimizer.state),
                'param_groups': simplify(optimizer.param_groups),
            }

            deltas = get_deltas(model, *trainset)
            h_pos = None
            if data['train'][1] > 0:
                x = deltas.clone()
                x[x < 0] = 0
                x = x / data['train'][1] ** 0.5
                h_pos, _ = np.histogram(x.detach().cpu().numpy(), bins, density=True)
            x = -deltas.clone()
            x[x < 0] = 0
            h_neg, _ = np.histogram(x.detach().cpu().numpy(), bins, density=True)

            data['deltas'] = {
                'bins': bins,
                'positive': h_pos,
                'negative': h_neg,
            }

            time_1 = time_logging.end("error and loss", time_1)

            if data['train'][0] == 0:  # with the hinge, no errors => finished
                break

        if step in args.checkpoints:
            logger.info("({}) checkpoint".format(desc['p']))

            hessian = None
            if 8 * model.N**2 < 2e9 and args.compute_hessian:
                logger.info("({}) compute the hessian".format(desc['p']))
                hess1, hess2, e, e1, e2 = compute_hessian_evalues(model, *trainset)

                hessian = {
                    "hess_eval": e.cpu(),
                    "hess1_eval": e1.cpu(),
                    "hess2_eval": e2.cpu(),
                }
                if args.save_hessian:
                    hessian["hess1"] = hess1.cpu()
                    hessian["hess2"] = hess2.cpu()
                time_1 = time_logging.end("hessian", time_1)

            with torch.no_grad():
                deltas = get_deltas(model, *trainset)
            error_loss = error_loss_grad(model, *trainset)

            checkpoints.append({
                "step": step,
                "train": error_loss,
                "state": copy.deepcopy(model.cpu().state_dict()),
                "deltas": deltas.cpu(),
                "hessian": hessian,
            })
            model.to(device)

        if args.n_steps_lr_decay and step > 0 and step % args.n_steps_lr_decay == 0:
            for pg in optimizer.param_groups:
                if isinstance(optimizer, FIRE):
                    pg['dt_max'] = max(args.min_learning_rate, pg['dt_max'] / args.lr_decay_factor)
                    logger.info("({}) dt_max set to {}".format(desc['p'], pg['dt_max']))
                else:
                    pg['lr'] = max(args.min_learning_rate, pg['lr'] / args.lr_decay_factor)
                    logger.info("({}) learning rate set to {}".format(desc['p'], pg['lr']))

            noise = noise / args.lr_decay_factor

        if args.n_steps_bs_grow and step > 0 and step % args.n_steps_bs_grow == 0:
            batch_size = min(args.p, int(batch_size * args.bs_grow_factor))
            loader = simple_loader(*trainset, batch_size)
            logger.info("({}) batch size set to {}".format(desc['p'], batch_size))

        data, target = next(loader)
        time_1 = time_logging.end("load data", time_1)

        make_a_step(model, optimizer, data, target)
        if noise > 0:
            for p in model.parameters():
                with torch.no_grad():
                    p.add_(noise * p.pow(2).mean().sqrt(), torch.empty_like(p).normal_())

        time_1 = time_logging.end("make a step", time_1)

        step += 1

    del optimizer

    run = {
        "desc": desc,
        "args": args,
        "seed": seed,
        "N": model.N,
        "dynamics": dynamics,
        "checkpoints": checkpoints,
    }

    for i in count():
        path_script = os.path.join(args.log_dir, "script_{}.py".format(i))
        if not os.path.isfile(path_script):
            copyfile(__file__, path_script)
            break
    run["script"] = path_script

    with torch.no_grad():
        deltas = get_deltas(model, *trainset)
    error_loss = error_loss_grad(model, *trainset)

    run["last"] = {
        "train": error_loss,
        "state": None,
        "deltas": deltas.cpu(),
        "hessian": None,
        "Neff": n_effective(model, trainset[0], n_derive=1),
    }

    if 8 * model.N**2 < 2e9 and args.compute_hessian:
        try:
            logger.info("({}) compute the hessian".format(desc['p']))
            hess1, hess2, e, e1, e2 = compute_hessian_evalues(model, *trainset)

            hessian = {
                "hess_eval": e.cpu(),
                "hess1_eval": e1.cpu(),  # H0
                "hess2_eval": e2.cpu(),  # Hp
            }
            if args.save_hessian:
                hessian["hess1"] = hess1.cpu()  # H0
                hessian["hess2"] = hess2.cpu()  # Hp

            run["last"]["hessian"] = hessian

            del hess1, hess2, e, e1, e2
            time_1 = time_logging.end("hessian", time_1)
        except RuntimeError:
            pass

    run["last"]["state"] = model.cpu().state_dict()

    dump_run(args.log_dir, run)

    logger.info(time_logging.text_statistics())


def main():
    args = parse()
    objs = init(args)

    if objs is None:
        return

    time_0 = time_logging.start()
    train(args, *objs)
    time_logging.end("run", time_0)


if __name__ == '__main__':
    main()