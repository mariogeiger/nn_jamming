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
import collections


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str)
    parser.add_argument("--log_dir", type=str, required=True)

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", choices={"fc", "cnn"}, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--p", type=parse_kmg, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--rep", type=int, default=0)

    parser.add_argument("--optimizer", choices={"sgd", "adam", "adam0", "fire", "fire_simple", "adam_rlrop", "adam_simple", "fdr"}, required=True)
    parser.add_argument("--n_steps_max", type=parse_kmg, required=True)
    parser.add_argument("--compute_hessian", type=to_bool, default="False")
    parser.add_argument("--compute_neff", type=to_bool, default="False")
    parser.add_argument("--compute_activities", type=to_bool, default="False")
    parser.add_argument("--save_hessian", type=to_bool, default="False")
    parser.add_argument("--checkpoints", type=int, nargs='+', default=[])
    parser.add_argument("--nd_stop", type=int, default=0)
    parser.add_argument("--losspp_stop", type=float, default=0)

    parser.add_argument("--kappa", type=float, default=1)

    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--n_steps_lr_decay", type=int)
    parser.add_argument("--fdr_epoch", type=int)
    parser.add_argument("--lr_decay_factor", type=float)
    parser.add_argument("--min_learning_rate", type=float)
    parser.add_argument("--rlrop_cooldown", type=float)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--train_last", type=to_bool, default="False")

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_steps_bs_grow", type=int)
    parser.add_argument("--bs_grow_factor", type=float)

    parser.add_argument("--precision", choices={"f32", "f64"}, default="f64")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--activation", choices={"relu", "tanh"}, default="relu")

    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.architecture == "fc":
        assert args.depth is not None
    if args.architecture == "cnn":
        assert args.depth is None

    if args.optimizer == "sgd":
        if args.batch_size is None:
            args.batch_size = args.p
    if args.optimizer == "fdr":
        if args.learning_rate is None:
            args.learning_rate = 1e-2
        if args.batch_size is None:
            args.batch_size = args.p
        if args.lr_decay_factor is None:
            args.lr_decay_factor = 1 / 0.9
        if args.fdr_epoch is None:
            args.fdr_epoch = 500
    if args.optimizer == "adam_simple":
        if args.learning_rate is None:
            args.learning_rate = 1e-4
        if args.batch_size is None:
            args.batch_size = args.p
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
    if args.optimizer == "fire_simple":
        if args.learning_rate is None:
            args.learning_rate = 1e-2
        if args.batch_size is None:
            args.batch_size = args.p

    if args.chunk is None:
        args.chunk = args.p

    return args


def init(args):
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
        "rep": args.rep,
    }

    if desc in load_dir_desc2(args.log_dir):
        print("{} skiped".format(repr(desc)))
        return None

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    dtype = torch.float32 if args.precision == "f32" else torch.float64
    torch.set_default_dtype(dtype)

    logger = logging.getLogger("default")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    with FSLocker(os.path.join(args.log_dir, "output.pkl.lock")):
        for i in count():
            path_log = os.path.join(args.log_dir, "log_{:04d}".format(i))
            if not os.path.isfile(path_log):
                run_id = i
                fh = logging.FileHandler(path_log)
                break
        logger.addHandler(fh)

        copyfile(__file__, os.path.join(args.log_dir, "script_{:04d}.py".format(run_id)))

        logger.info("%s", repr(args))

    logger.info(desc)

    seed = torch.randint(2 ** 62, (), dtype=torch.long).item()
    trainset, testset = get_dataset(args.dataset, args.p, args.dim, seed, device, dtype)
    _x, y = trainset
    n_classes = 1 if y.ndimension() == 1 else y.size(1)

    activation = F.relu if args.activation == "relu" else torch.tanh
    if args.architecture == "fc":
        trainset = (trainset[0].flatten(1), trainset[1])
        if testset is not None:
            testset = (testset[0].flatten(1), testset[1])
        model = FC(args.dim, args.width, args.depth, activation, kappa=args.kappa, n_classes=n_classes)
    if args.architecture == "cnn":
        assert trainset[0].ndimension() == 4
        assert testset[0].ndimension() == 4
        model = CNN(args.dim, args.width, activation, kappa=args.kappa, n_classes=n_classes)
    model.to(device)
    model.type(dtype)

    logger.info("N={}".format(model.N))

    if args.train_last:
        parameters = model.layers[-1].parameters()
    else:
        parameters = model.parameters()

    scheduler = None
    if args.optimizer == "sgd" or args.optimizer == "fdr":
        optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=0)
    if args.optimizer == "adam" or args.optimizer == "adam0" or args.optimizer == "adam_simple":
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    if args.optimizer == "fire" or args.optimizer == "fire_simple":
        optimizer = FIRE(parameters, dt_max=args.learning_rate, a_start=1 - args.momentum)
    if args.optimizer == "adam_rlrop":
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=0, verbose=True, threshold=-1, threshold_mode="rel", cooldown=args.rlrop_cooldown, min_lr=args.min_learning_rate)

    return model, trainset, testset, logger, optimizer, scheduler, device, desc, seed, run_id


def train(args, model, trainset, testset, logger, optimizer, scheduler, device, desc, seed, run_id):
    measure_points = set(intlogspace(1, args.n_steps_max, 150, with_zero=True, with_end=True))
    dynamics = []

    checkpoints = []

    time_1 = time_logging.start()

    batch_size = args.batch_size

    bins = np.logspace(-9, 4, 130)
    bins = np.concatenate([[-1], bins])

    fluctuation = 0
    dissipation = 0

    init_state = copy.deepcopy(model.state_dict())

    if args.compute_activities:
        init_act = get_activities(model, trainset[0])

    step = 0
    while True:
        if step > args.n_steps_max:
            break

        if step % 100 == 0:
            print("{:.2f}%      ".format(step / args.n_steps_max * 100), end="\r")

        if step in measure_points or step % 1000 == 0:
            data = {}
            dynamics.append(data)

            data['state'] = {
                "norm": collections.OrderedDict([(n, p.norm().item()) for n, p in model.named_parameters()]),
                "displacement": collections.OrderedDict([(n, (p - init_state[n]).norm().item()) for n, p in model.named_parameters()]),
            }

            data['step'] = step
            data['train'] = error_loss_grad(model, *trainset)
            logger.info("id={} P={} step={} nd={:d} nd/P={:.1f}% Loss={:.2g} |Grad|={:.2g} |w-w0|={:.2g} |w|={:.2g}".format(
                    run_id,
                    desc['p'],
                    step,
                    data['train'][0],
                    100 * data['train'][0] / args.p,
                    data['train'][1],
                    data['train'][2],
                    sum(p ** 2 for n, p in data['state']['displacement'].items()) ** 0.5,
                    sum(p ** 2 for n, p in data['state']['norm'].items()) ** 0.5,
                )
            )
            if testset is not None:
                data['test'] = error_loss_grad(model, *testset)

            if args.compute_activities:
                acti = get_activities(model, trainset[0])
                data['activities'] = {
                    "continuous": [(a - a0).norm().div(a0.norm()).item() for a, a0 in zip(acti, init_act)],
                    "binary": [((a > 0) != (a0 > 0)).long().sum().item() for a, a0 in zip(acti, init_act)],
                }

            if args.optimizer == "adam_rlrop":
                scheduler.step(data['train'][1])

            data['batch_size'] = batch_size
            data['optimizer'] = {
                'state': simplify(optimizer.state),
                'param_groups': simplify(optimizer.param_groups),
            }

            with torch.no_grad():
                deltas = get_deltas(model, *trainset, 1024)
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

            if data['train'][0] <= args.nd_stop:  # with the hinge, no errors => finished
                break

            if data['train'][1] * args.p < args.losspp_stop * data['train'][0]:
                break

        if step in args.checkpoints:
            logger.info("({}|{}) checkpoint".format(run_id, desc['p']))

            hessian = None
            if 8 * model.N**2 < 2e9 and args.compute_hessian:
                logger.info("({}|{}) compute the hessian".format(run_id, desc['p']))
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
                deltas = get_deltas(model, *trainset, 1024)
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
                    logger.info("({}|{}) dt_max set to {}".format(run_id, desc['p'], pg['dt_max']))
                else:
                    pg['lr'] = max(args.min_learning_rate, pg['lr'] / args.lr_decay_factor)
                    logger.info("({}|{}) learning rate set to {}".format(run_id, desc['p'], pg['lr']))

        if args.n_steps_bs_grow and step > 0 and step % args.n_steps_bs_grow == 0:
            batch_size = min(args.p, int(batch_size * args.bs_grow_factor))
            logger.info("({}|{}) batch size set to {}".format(run_id, desc['p'], batch_size))

        time_1 = time_logging.end("load data", time_1)

        make_a_step(model, optimizer, *trainset, batch_size)

        if args.optimizer == "fdr":
            fluctuation += sum(torch.dot(p.view(-1), p.grad.view(-1)) for p in model.parameters()).item()

            assert len(optimizer.param_groups) == 1
            group = optimizer.param_groups[0]
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            assert weight_decay == 0
            assert nesterov == False

            for p in group['params']:
                if p.grad is None:
                    continue

                if momentum != 0:
                    param_state = optimizer.state[p]
                    v = param_state['momentum_buffer']
                    dissipation += 0.5 * (1 + momentum) / (1 - dampening) * lr * v.norm() ** 2
                else:
                    v = p.grad
                    dissipation += 0.5 * lr * v.norm() ** 2

            if step > 0 and step % args.fdr_epoch == 0:
                fluctuation /= args.fdr_epoch
                dissipation /= args.fdr_epoch
                logger.info("({}|{}) fluctuation = {}, dissipation = {}".format(run_id, desc['p'], fluctuation, dissipation))
                if abs(fluctuation / dissipation - 1) < 0.01:
                    group['lr'] = lr / args.lr_decay_factor
                    logger.info("({}|{}) learning rate set to {}".format(run_id, desc['p'], group['lr']))

                fluctuation = 0
                dissipation = 0

        time_1 = time_logging.end("make a step", time_1)

        step += 1

    del optimizer

    run = {
        "id": run_id,
        "desc": desc,
        "args": args,
        "seed": seed,
        "N": model.N,
        "dynamics": dynamics,
        "checkpoints": checkpoints,
    }

    error_loss = error_loss_grad(model, *trainset)
    with torch.no_grad():
        deltas = get_deltas(model, *trainset, 1024)

    run["init"] = {
        "state": collections.OrderedDict([(n, p.cpu()) for n, p in init_state.items()]),
    }

    run["last"] = {
        "train": error_loss,
        "state": None,
        "deltas": deltas.cpu(),
        "hessian": None,
        "Neff": None,
    }
    if 8 * model.N**2 < 1e9 and args.compute_neff:
        try:
            run['last']['Neff'] = n_effective(model, trainset[0], n_derive=1)
        except RuntimeError:
            pass

    if testset is not None:
        run['last']['test'] = error_loss_grad(model, *testset)
        with torch.no_grad():
            deltas_test = get_deltas(model, *testset, 1024)
        run['last']["deltas_test"] = deltas_test.cpu(),
        run['p_test'] = len(testset[0])

    if 8 * model.N**2 < 2e9 and args.compute_hessian:
        try:
            logger.info("({}|{}) compute the hessian".format(run_id, desc['p']))
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

    dump_run2(args.log_dir, run)

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