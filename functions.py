# pylint: disable=W0221, C, R, W1202, E1101, E1102
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle
import os
import math
import numpy as np
import fcntl


class FSLocker:
    def __init__(self, filename):
        self.f = None
        self.filename = filename
    def __enter__(self):
        self.f = open(self.filename, 'w')
        fcntl.lockf(self.f, fcntl.LOCK_EX)
        self.f.write(str(os.getpid()))
        self.f.flush()
    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.lockf(self.f, fcntl.LOCK_UN)
        self.f.close()


class RandomBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, p, d, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        points = torch.randn(p, d)
        self.points = d ** 0.5 * points / torch.sum(points ** 2, dim=1, keepdim=True) ** 0.5

        self.labels = torch.randint(2, size=(p, 1)) * 2 - 1

    def __getitem__(self, index):
        return self.points[index], self.labels[index]

    def __len__(self):
        return len(self.points)


class Model(nn.Module):

    def __init__(self, width, depth, dim, init, act, kappa=0.5, lamda=None):
        super().__init__()

        layers = nn.ModuleList()

        f = dim

        for _ in range(depth):
            lin = nn.Linear(f, width, bias=True)
            if init == "pytorch":
                pass
            if init == "orth":
                nn.init.orthogonal_(lin.weight)
                nn.init.zeros_(lin.bias)

            layers += [lin]
            f = width

        lin = nn.Linear(f, 1, bias=True)
        nn.init.normal_(lin.weight, std=1e-2 / f ** 0.5)
        nn.init.normal_(lin.bias, std=1e-2)

        layers += [lin]

        if lamda is not None:
            self.kappa = nn.Parameter(torch.tensor(kappa))
            self.lamda = lamda
        else:
            self.kappa = kappa
            self.lamda = 0

        self.act = act
        self.depth = depth
        self.width = width
        self.layers = layers
        self.preactivations = None
        self.N = sum(layer.weight.numel() for layer in self.layers)

    def forward(self, x):
        self.preactivations = []

        for layer in self.layers[:-1]:
            x = layer(x)
            self.preactivations.append(x)

            if self.act == "relu":
                x = F.relu(x)
            if self.act == "tanh":
                x = torch.tanh(x)

        return self.layers[-1](x).view(-1)

    def normalize_weights(self):
        for layer in self.layers:
            weight = layer.weight  # [out, in]
            with torch.no_grad():
                norm = weight.norm(dim=1, keepdim=True)  # [out, 1]
                weight.div_(norm)


def gradient(x, params):
    params = list(params)
    row = torch.autograd.grad(x, params, retain_graph=True, allow_unused=True, create_graph=True)
    row = [x if x is not None else torch.zeros_like(y) for x, y in zip(row, params)]
    row = torch.cat([x.contiguous().view(-1) for x in row])
    return row


def get_deltas(model, data_x, data_y):
    output = model(data_x)  # [p]
    model.preactivations = None  # free memory
    delta = model.kappa - output * data_y  # [p]
    return delta


def get_mistakes(model, data_x, data_y):
    with torch.no_grad():
        output = model(data_x)  # [p]
        model.preactivations = None  # free memory
        delta = model.kappa - output * data_y  # [p]
        mask = (delta > 0)
    return data_x[mask], data_y[mask]


def get_activities(model, data_x):
    model.eval()

    with torch.no_grad():
        model(data_x)

        activities = torch.stack([
            a > 0
            for j, a in enumerate(model.preactivations)
        ], dim=1)
        model.preactivations = None  # free memory

    return activities


def compute_hessian(model, data_x, data_y):
    from hessian_pytorch import hessian

    model.eval()
    p = len(data_x)
    Ntot = sum(p.numel() for p in model.parameters())

    mist_x, mist_y = get_mistakes(model, data_x, data_y)
    if len(mist_x) == 0:
        H = data_x.new_zeros(Ntot, Ntot)
        return H, H

    delta = get_deltas(model, mist_x, mist_y)
    #assert (delta > 0).all(), delta

    H0 = data_x.new_zeros(Ntot, Ntot)  # da Delta_i db Delta_i
    for de in delta:
        g = gradient(de, model.parameters()).detach()
        H0.add_(g.view(-1, 1) * g.view(1, -1) / p)

    Hp = hessian((delta.detach() * delta).sum() / p, model.parameters())  # Delta_i da db Delta_i

    return H0, Hp


def compute_hessian_evalues(model, data_x, data_y):
    H0, Hp = compute_hessian(model, data_x, data_y)

    e0, _ = torch.eig(H0, eigenvectors=False)
    e0 = e0[:, 0]

    ep, _ = torch.eig(Hp, eigenvectors=False)
    ep = ep[:, 0]

    H = H0.add_(Hp)

    e, _ = torch.eig(H, eigenvectors=False)
    e = e[:, 0]

    H0 = H.sub_(Hp)

    return H0, Hp, e, e0, ep


def error_loss_grad(model, data_x, data_y):
    model.eval()

    output = model(data_x)  # [p]
    model.preactivations = None  # free memory

    delta = model.kappa - output * data_y  # [p]
    total_mistake = (delta > 0).long().sum().item()

    loss = 0.5 * (delta > 0).type_as(data_x) * delta.abs().pow(2) - model.lamda * model.kappa  # [p]
    total_loss = loss.mean().item()

    grad_sum_norm = gradient(loss.mean(), model.parameters()).norm().item()

    loss = loss[delta > 0]
    delta = delta[delta > 0]

    if loss.size(0) > 0:
        n = int(math.ceil(loss.size(0) / 10))
        grad_avg_small_norm = sum([gradient(loss[i] / data_x.size(0), model.parameters()).norm().item() for i in delta.sort()[1][:n]]) / n
    else:
        grad_avg_small_norm = 0

    return total_mistake, total_loss, grad_sum_norm, grad_avg_small_norm


def make_a_step(model, optimizer, data_x, data_y):
    '''
    data_x [batch, k] (?, dim)
    data_y [batch, class] (?,)
    '''
    model.train()

    mist = get_mistakes(model, data_x, data_y)
    if mist[0].size(0) == 0:
        return 0

    deltas = get_deltas(model, *mist)
    loss = 0.5 * deltas.pow(2).sum() / data_x.size(0) - model.lamda * model.kappa

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def load_dir(directory):
    path = os.path.join(directory, "output.pkl")
    if not os.path.isfile(path):
        return
    with FSLocker(os.path.join(directory, "output.pkl.lock")):
        with open(path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


def dump_run(directory, run):
    with FSLocker(os.path.join(directory, "output.pkl.lock")):
        with open(os.path.join(directory, "output.pkl"), "ab") as f:
            pickle.dump(run, f)


def simplify(stuff):
    if isinstance(stuff, list):
        return [simplify(x) for x in stuff]
    if isinstance(stuff, dict):
        return {simplify(key): simplify(value) for key, value in stuff.items()}
    if isinstance(stuff, (int, float, str, bool)):
        return stuff
    return None


def simple_loader(x, y, batch_size):
    if x.size(0) <= batch_size:
        while True:
            yield x, y
    while True:
        i = torch.randperm(x.size(0), device=x.device)[:batch_size]
        yield x[i], y[i]


def intlogspace(begin, end, num, with_zero=False, with_end=True):
    '''
    >>> intlogspace(1, 100, 5)
    array([  1,   3,  10,  32, 100])
    '''
    if with_zero:
        output = intlogspace(begin, end, num - 1, with_zero=False, with_end=with_end)
        return np.concatenate([[0], output])
    if not with_end:
        return intlogspace(begin, end, num + 1, with_zero=with_zero, with_end=True)[:-1]

    assert not with_zero
    assert with_end

    if num >= end - begin + 1:
        return np.arange(begin, end + 1).astype(np.int64)

    n = num
    while True:
        inc = (end / begin) ** (1 / (n - 1))
        output = np.unique(np.round(begin * inc ** np.arange(0, n)).astype(np.int64))
        if len(output) < num:
            n += 1
        else:
            return output
