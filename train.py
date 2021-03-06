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
import collections


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str)
    parser.add_argument("--log_dir", type=str, required=True)

    parser.add_argument("--init_seed", type=int)
    parser.add_argument("--data_seed", type=int)
    parser.add_argument("--skip", type=to_bool, default="True")

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--p", type=parse_kmg, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--rep", type=int, default=0)

    parser.add_argument("--init", choices={"orth", "normal"}, default="orth", required=True)
    parser.add_argument("--init_gain", type=float, default=1)

    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--lr_width_exponent", type=float, default=0)
    parser.add_argument("--n_steps_max", type=parse_kmg, required=True)
    parser.add_argument("--compute_hessian", type=to_bool, default="False")
    parser.add_argument("--compute_neff", type=to_bool, default="False")
    parser.add_argument("--compute_activities", type=to_bool, default="False")
    parser.add_argument("--compute_input_gradients", type=to_bool, default="False")
    parser.add_argument("--compute_outputs", type=to_bool, default="False")
    parser.add_argument("--subtract_init", type=to_bool, default="False")
    parser.add_argument("--save_hessian", type=to_bool, default="False")
    parser.add_argument("--checkpoints", type=int, nargs='+', default=[])
    parser.add_argument("--nd_stop", type=int, default=0)
    parser.add_argument("--losspp_stop", type=float, default=0)

    parser.add_argument("--kappa", type=float, default=1)

    parser.add_argument("--max_learning_rate", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--train_last", type=to_bool, default="False")
    
    parser.add_argument("--dropout", type=to_bool, default="False")

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_steps_bs_grow", type=int)
    parser.add_argument("--bs_grow_factor", type=float)

    parser.add_argument("--precision", choices={"f32", "f64"}, default="f64")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--activation", choices={"relu", "tanh"}, default="relu")

    args = parser.parse_args()


    if args.device is None:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.optimizer == "sgd":
        if args.batch_size is None:
            args.batch_size = args.p
    if args.optimizer == "adam":
        if args.eps is None:
            args.eps = 1e-8
        if args.max_learning_rate is None:
            args.max_learning_rate = 1e-4
        if args.learning_rate is None:
            args.learning_rate = 1e-4
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

    if args.skip and desc in load_dir_desc2(args.log_dir):
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

    init_seed = torch.randint(2 ** 62, (), dtype=torch.long).item() if args.init_seed is None else args.init_seed
    data_seed = torch.randint(2 ** 62, (), dtype=torch.long).item() if args.data_seed is None else args.data_seed

    trainset, testset = get_dataset(args.dataset, args.p, args.dim, data_seed)
    trainset = (trainset[0].type(dtype).to(device), trainset[1].type(dtype).to(device))
    if testset is not None:
        testset = (testset[0].type(dtype).to(device), testset[1].type(dtype).to(device))

    _x, y = trainset
    n_classes = 1 if y.ndimension() == 1 else y.size(1)

    torch.manual_seed(init_seed)
    activation = F.relu if args.activation == "relu" else torch.tanh

    trainset = (trainset[0].flatten(1), trainset[1])
    if testset is not None:
        testset = (testset[0].flatten(1), testset[1])
    model = FC(args.dim, args.width, args.depth, activation, kappa=args.kappa, n_classes=n_classes, dropout=args.dropout)

    for n, p in model.named_parameters():
        if 'bias' in n:
            nn.init.zeros_(p)
        if 'weight' in n:
            if args.init == "orth":
                orthogonal_(p, gain=args.init_gain)
            elif args.init == "normal":
                nn.init.normal_(p, std=args.init_gain / p.size(1) ** 0.5)
            else:
                raise ValueError()

    model.to(device)
    model.type(dtype)

    logger.info("N={}".format(model.N))

    if args.train_last:
        parameters = model.layers[-1].parameters()
    else:
        parameters = model.parameters()

    if args.subtract_init:
        f = model
        f0 = copy.deepcopy(f)
        for p in f0.parameters():
            p.requires_grad = False
        model = SumModules([f, f0], [1, -1])
        model.kappa = f.kappa
        model.preactivations = f.preactivations
        model.act = f.act
        model.N = f.N

    scheduler = None
    learning_rate = min(args.learning_rate * args.width ** args.lr_width_exponent, args.max_learning_rate)
    logger.info("learning rate = {}".format(learning_rate))

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=args.momentum, weight_decay=0)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, eps=args.eps)
    if args.optimizer == "fire":
        from fire import FIRE
        optimizer = FIRE(parameters, dt_max=learning_rate, a_start=1 - args.momentum)

    return model, trainset, testset, logger, optimizer, scheduler, device, desc, init_seed, data_seed, run_id


def train(args, model, trainset, testset, logger, optimizer, scheduler, device, desc, init_seed, data_seed, run_id):
    measure_points = set(intlogspace(1, args.n_steps_max, 150, with_zero=True, with_end=True))
    dynamics = []

    checkpoints = []

    time_0 = time_1 = time_logging.start()

    batch_size = args.batch_size

    bins = np.logspace(-9, 4, 130)
    bins = np.concatenate([[-1], bins])

    init_state = copy.deepcopy(model.state_dict())

    if args.compute_activities:
        init_act = get_activities(model, trainset[0], 1024)

    step = 0
    while True:
        if step > args.n_steps_max:
            break

        if step in measure_points or step % 1000 == 0:
            data = {}
            dynamics.append(data)

            data['state'] = {
                "norm": collections.OrderedDict([(n, p.norm().item()) for n, p in model.named_parameters()]),
                "displacement": collections.OrderedDict([(n, (p - init_state[n]).norm().item()) for n, p in model.named_parameters()]),
            }

            data['outnorm'] = {
                "train": get_outputs(model, trainset[0], 1024).pow(2).mean().item(),
                "test": get_outputs(model, testset[0], 1024).pow(2).mean().item() if testset is not None else None,
            }

            data['step'] = step
            data['train'] = error_loss_grad(model, *trainset)
            logger.info("id={} P={} d={} L={} h={} step={} nd={:d} nd/P={:.1f}% Loss={:.2g} |Grad|={:.2g} |w-w0|={:.2g} |w|={:.2g}".format(
                    run_id,
                    desc['p'],
                    desc['dim'],
                    desc['depth'],
                    desc['width'],
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
                acti = get_activities(model, trainset[0], 1024)
                data['activities'] = {
                    "continuous": [(a - a0).norm().div(a0.norm()).item() for a, a0 in zip(acti, init_act)],
                    "binary": [((a > 0) != (a0 > 0)).long().sum().item() for a, a0 in zip(acti, init_act)],
                }

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

        if args.n_steps_bs_grow and step > 0 and step % args.n_steps_bs_grow == 0:
            batch_size = min(args.p, int(batch_size * args.bs_grow_factor))
            logger.info("({}|{}) batch size set to {}".format(run_id, desc['p'], batch_size))

        time_1 = time_logging.end("load data", time_1)

        make_a_step(model, optimizer, *trainset, batch_size)

        time_1 = time_logging.end("make a step", time_1)

        step += 1

    del optimizer

    run = {
        "id": run_id,
        "desc": desc,
        "args": args,
        "init_seed": init_seed,
        "data_seed": data_seed,
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

    grads = None
    if args.compute_input_gradients:
        logger.info("({}|{}) compute input gradients".format(run_id, desc['p']))
        grads = {
            "train": get_gradients(model, trainset[0]).cpu(),
            "test": get_gradients(model, testset[0]).cpu() if testset is not None else None,
        }
        time_1 = time_logging.end("input gradients", time_1)

    outputs = None
    if args.compute_outputs:
        outputs = {
            "train": get_outputs(model, trainset[0], 1024).cpu(),
            "test": get_outputs(model, testset[0], 1024).cpu() if testset is not None else None,
        }

    run["last"] = {
        "train": error_loss,
        "state": None,
        "deltas": deltas.cpu(),
        "hessian": None,
        "Neff": None,
        "grads": grads,
        "outputs": outputs,
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

    time_logging.end("total", time_0)
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
