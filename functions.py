# pylint: disable=W0221, C, R, W1202, E1101, E1102
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle
import os
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


def get_dataset(dataset, p, dim, seed, device):
    torch.manual_seed(seed)

    if dataset == "random":
        x = torch.randn(p, dim, dtype=torch.float64, device=device)

    if dataset == "cifar":
        import torchvision
        from itertools import chain

        proj = torch.empty(dim, 3 * 32 ** 2, dtype=torch.float64)
        orthogonal_(proj)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470322, 0.243485, 0.261587849]),
            lambda x: proj @ x.view(-1).type(torch.float64)
        ])

        def target_transform(y):
            return torch.tensor(0 if y in [0, 1, 8, 9] else 1)

        trainset = torchvision.datasets.CIFAR10('../cifar10', train=True, download=True, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.CIFAR10('../cifar10', train=False, transform=transform, target_transform=target_transform)

        dataset = []
        for i, xy in enumerate(trainset):
            dataset.append(xy)
            if i % 100 == 0: print("cifar10 {:.1f}%".format(100 * i / (len(trainset) + len(testset))), end="        \r")
        for i, xy in enumerate(testset):
            dataset.append(xy)
            if i % 100 == 0: print("cifar10 {:.1f}%".format(100 * (len(trainset) + i) / (len(trainset) + len(testset))), end="        \r")

        classes = [[x for x, y in dataset if y == i] for i in range(2)]
        for xs in classes:
            xs = [xs[i] for i in torch.randperm(len(xs))]

        xs = list(chain(*zip(*classes)))
        assert p <= len(xs)

        x = torch.stack(xs)
        x = x[:p].to(device)

    x = x - x.mean(0)
    x = dim ** 0.5 * x / x.norm(dim=1, keepdim=True)
    y = (torch.arange(p, dtype=torch.float64, device=device) % 2) * 2 - 1

    return x, y


def orthogonal_(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new_empty(rows, cols).normal_(0, 1)

    for i in range(0, rows, cols):
        # Compute the qr factorization
        q, r = torch.qr(flattened[i:i + cols].t())
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        q *= torch.diag(r, 0).sign()
        q.t_()

        with torch.no_grad():
            tensor[i:i + cols].view_as(q).copy_(q)

    with torch.no_grad():
        tensor.mul_(gain)
    return tensor


class Model(nn.Module):

    def __init__(self, dim, width, depth, kappa=0.5, lamda=None):
        super().__init__()

        layers = nn.ModuleList()

        f = dim

        for _ in range(depth):
            lin = nn.Linear(f, width, bias=True)
            orthogonal_(lin.weight)
            nn.init.zeros_(lin.bias)

            layers += [lin]
            f = width

        lin = nn.Linear(f, 1, bias=True)
        orthogonal_(lin.weight, gain=kappa)
        nn.init.zeros_(lin.bias)

        layers += [lin]

        if lamda is not None:
            self.kappa = nn.Parameter(torch.tensor(kappa))
            self.lamda = lamda
        else:
            self.kappa = kappa
            self.lamda = 0

        self.dim = dim
        self.width = width
        self.depth = depth
        self.layers = layers
        self.preactivations = None
        self.N = sum(layer.weight.numel() for layer in self.layers)

    def forward(self, x):
        self.preactivations = []

        for layer in self.layers[:-1]:
            x = layer(x)
            self.preactivations.append(x)
            x = F.relu(x)

        return self.layers[-1](x).view(-1)


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


def compute_h0(model, deltas):
    '''
    Compute extensive
    '''
    Ntot = sum(p.numel() for p in model.parameters())
    H0 = deltas.new_zeros(Ntot, Ntot)  # da Delta_i db Delta_i
    for delta in deltas:
        g = gradient(delta, model.parameters()).detach()
        H0.add_(g.view(-1, 1) * g.view(1, -1))
    return H0


def compute_hp(model, deltas):
    from hessian_pytorch import hessian
    return hessian((deltas.detach() * deltas).sum(), model.parameters())  # Delta_i da db Delta_i


def compute_hessian(model, data_x, data_y):
    model.eval()
    p = len(data_x)
    Ntot = sum(p.numel() for p in model.parameters())

    mist_x, mist_y = get_mistakes(model, data_x, data_y)
    if len(mist_x) == 0:
        H = data_x.new_zeros(Ntot, Ntot)
        return H, H

    deltas = get_deltas(model, mist_x, mist_y)
    return compute_h0(model, deltas) / p, compute_hp(model, deltas) / p


def compute_hessian_evalues(model, data_x, data_y):
    H0, Hp = compute_hessian(model, data_x, data_y)

    e0, _ = torch.symeig(H0)
    ep, _ = torch.symeig(Hp)

    H = H0.add_(Hp)
    e, _ = torch.symeig(H)
    H0 = H.sub_(Hp)

    return H0, Hp, e, e0, ep


def error_loss_grad(model, data_x, data_y):
    model.eval()

    output = model(data_x)  # [p]
    model.preactivations = None  # free memory

    delta = model.kappa - output * data_y  # [p]
    loss = 0.5 * ((delta > 0).type_as(data_x) * delta.abs().pow(2)).mean() - model.lamda * model.kappa
    grad_sum_norm = gradient(loss, model.parameters()).norm().item()

    return (delta > 0).long().sum().item(), loss.item(), grad_sum_norm


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
