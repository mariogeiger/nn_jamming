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

    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--act", choices={"relu", "tanh"}, default="relu")

    parser.add_argument("--optimizer", choices={"sgd", "adam", "adam0", "fire", "adam_rlrop"}, default="adam0")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--n_steps_min", type=int)
    parser.add_argument("--n_steps_max", type=int, default=int(1e7))
    parser.add_argument("--stop_grad", type=float, default=1e-10)
    parser.add_argument("--stop_loss", type=float, default=1e-20)
    parser.add_argument("--save_hessian", action="store_true")
    parser.add_argument("--checkpoints", type=int, nargs='+', default=[])
    parser.add_argument("--init", choices={"pytorch", "orth"}, default="orth")
    parser.add_argument("--noise", type=float, default=0)

    parser.add_argument("--normalize_weights", action="store_true")
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
        if args.n_steps_min is None:
            args.n_steps_min = 1e6
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
        if args.n_steps_min is None:
            args.n_steps_min = 1e6
        if args.batch_size is None:
            args.batch_size = args.p
    if args.optimizer == "adam":
        if args.learning_rate is None:
            args.learning_rate = 1e-4
        if args.n_steps_lr_decay is None:
            args.n_steps_lr_decay = 1e5
        if args.lr_decay_factor is None:
            args.lr_decay_factor = 2
        if args.n_steps_min is None:
            args.n_steps_min = 2e5
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
        if args.n_steps_min is None:
            args.n_steps_min = 2e5
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
        "act": args.act,
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
    trainset = RandomBinaryDataset(args.p, args.dim, seed=seed)
    x = torch.stack([x for x, y in trainset]).to(device)  # [p, dim]
    y = torch.stack([y for x, y in trainset]).to(device).view(-1)  # [p]

    x = x.type(torch.float64)
    y = y.type(torch.float64)
    trainset = (x, y)
    del x, y

    model = Model(args.width, args.depth, args.dim, args.init, args.act, args.kappa, args.lamda)
    model.to(device)

    model.type(torch.float64)

    if args.normalize_weights:
        model.normalize_weights()

    N = sum(layer.weight.numel() for layer in model.layers)
    print("N={}".format(N))

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

    return model, trainset, logger, optimizer, scheduler, N, device, desc, seed


def train(args, model, trainset, logger, optimizer, scheduler, N, device, desc, seed):
    noise = args.noise

    measure_points = set(intlogspace(1, args.n_steps_max, 150, with_zero=True, with_end=True))
    dynamics = []

    checkpoints = []

    time_1 = time_logging.start()

    backup_best = None
    best_error_loss = (args.p, 1)

    batch_size = args.batch_size
    loader = simple_loader(*trainset, batch_size)

    bins = np.logspace(-9, 4, 130)
    bins = np.concatenate([[-1], bins])

    step = 0
    while True:
        if step > args.n_steps_max:
            break

        if step % 100 == 0:
            print("{:.2f}% {:.2f}%      ".format(step / args.n_steps_min * 100, step / args.n_steps_max * 100), end="\r")

        if step in measure_points or step % 1000 == 0:
            data = {}
            dynamics.append(data)

            data['step'] = step
            data['train'] = error_loss_grad(model, *trainset)
            logger.info("[{}] train={:d} ({:.1f}%), {:.2g}, |Grad| / avg_small(|Grad|)={:.2g}/{:.2g} = {:.2g}".format(
                    step, data['train'][0], 100 * data['train'][0] / args.p,
                    data['train'][1],
                    data['train'][2], data['train'][3],
                    data['train'][2] / data['train'][3] if data['train'][3] > 0 else 0
                )
            )

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

            if data['train'] < best_error_loss:
                best_error_loss = data['train']
                backup_best = copy.deepcopy(model)

            time_1 = time_logging.end("error and loss", time_1)

            if data['train'][0] == 0.0 and data['train'][1] < args.stop_loss:
                logger.info("loss smaller than stop_loss !")
                break

            if data['train'][2] < args.stop_grad and step > args.n_steps_min:
                logger.info("gradient small !")
                break

        if step in args.checkpoints:
            logger.info("checkpoint")

            hessian = None
            if 8 * N**2 < 1e9:
                logger.info("compute the hessian")
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
                    logger.info("dt_max set to {}".format(pg['dt_max']))
                else:
                    pg['lr'] = max(args.min_learning_rate, pg['lr'] / args.lr_decay_factor)
                    logger.info("learning rate set to {}".format(pg['lr']))

            noise = noise / args.lr_decay_factor

        if args.n_steps_bs_grow and step > 0 and step % args.n_steps_bs_grow == 0:
            batch_size = min(args.p, int(batch_size * args.bs_grow_factor))
            loader = simple_loader(*trainset, batch_size)
            logger.info("batch size set to {}".format(batch_size))

        data, target = next(loader)
        time_1 = time_logging.end("load data", time_1)

        make_a_step(model, optimizer, data, target)
        if args.normalize_weights:
            model.normalize_weights()
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
        "N": N,
        "dynamics": dynamics,
        "checkpoints": checkpoints,
    }

    for model, name in zip([model, backup_best], ["last", "best"]):

        hessian = None
        if 8 * N**2 < 1e9:
            logger.info("compute the hessian")
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

        run[name] = {
            "train": error_loss,
            "state": model.cpu().state_dict(),
            "deltas": deltas.cpu(),
            "hessian": hessian,
        }

    for i in count():
        path_script = os.path.join(args.log_dir, "script_{}.py".format(i))
        if not os.path.isfile(path_script):
            copyfile(__file__, path_script)
            break
    run["script"] = path_script

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