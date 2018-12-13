# pylint: disable=W0221, C, R, W1202, E1101, E1102
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle
import os
import numpy as np
import fcntl
import copy
from hessian import gradient
import functools


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


def get_dataset(dataset, p, dim, seed=None):
    if seed is None:
        seed = torch.randint(2 ** 32, (), dtype=torch.long).item()

    return _get_dataset(dataset, p, dim, seed)


@functools.lru_cache(maxsize=2)
def _get_dataset(dataset, p, dim, seed):
    torch.manual_seed(seed)

    y = None
    yg = None

    if dataset == "random":
        x = torch.randn(p, dim, dtype=torch.float64)
        xg = None

    elif dataset.startswith("mnist"):
        import torchvision
        from itertools import chain

        if dataset == "mnist_scale":
            w = int(dim ** 0.5)
            assert dim == w * w
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(w),
                torchvision.transforms.ToTensor(),
                lambda x: x.type(torch.float64)
            ])
        elif dataset == "mnist_flat":
            assert dim == 28 * 28
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                lambda x: x.type(torch.float64).view(-1),
            ])
        elif dataset == "mnist_pca":
            m, v, e = torch.load('../mnist/pca.pkl')
            assert dim <= (e > 0).long().sum().item()
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                lambda x: x.type(torch.float64).view(-1),
                lambda x: (x - m) @ v[:, :dim] / e[:dim] ** 0.5,
            ])
        else:
            raise ValueError("unknown dataset")

        target_transform = lambda y: torch.tensor(y % 2)

        trainset = torchvision.datasets.MNIST('../mnist', train=True, download=True, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.MNIST('../mnist', train=False, transform=transform, target_transform=target_transform)

        dataset = []
        for i, xy in enumerate(trainset):
            dataset.append(xy)
            if i % 100 == 0: print("\rmnist {:.1f}%".format(100 * i / (len(trainset) + len(testset))), end=" ")
        for i, xy in enumerate(testset):
            dataset.append(xy)
            if i % 100 == 0: print("\rmnist {:.1f}%".format(100 * (len(trainset) + i) / (len(trainset) + len(testset))), end=" ")
        print("\rmnist complete")

        classes = [[x for x, y in dataset if y == i] for i in range(2)]
        classes = [[xs[i] for i in torch.randperm(len(xs))] for xs in classes]

        xs = list(chain(*zip(*classes)))
        assert p <= len(xs), "p={} and we have {} images".format(p, len(xs))

        x = torch.stack(xs)
        xg = x[p:]
        x = x[:p]

    elif dataset == "cifar100":
        import torchvision
        from itertools import chain

        assert dim == 3 * 32 * 32
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda x: x.type(torch.float64)
        ])

        trainset = torchvision.datasets.CIFAR100('../cifar100', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100('../cifar100', train=False, transform=transform)

        dataset = []
        for i, xy in enumerate(trainset):
            dataset.append(xy)
            if i % 100 == 0: print("\rcifar100 {:.1f}%".format(100 * i / (len(trainset) + len(testset))), end=" ")
        for i, xy in enumerate(testset):
            dataset.append(xy)
            if i % 100 == 0: print("\rcifar100 {:.1f}%".format(100 * (len(trainset) + i) / (len(trainset) + len(testset))), end=" ")
        print("\rcifar100 complete")

        classes = [[x for x, y in dataset if y == i] for i in range(100)]
        classes = [[xs[i] for i in torch.randperm(len(xs))] for xs in classes]

        xs = list(chain(*zip(*classes)))
        assert p <= len(xs), "p={} and we have {} images".format(p, len(xs))

        y = -torch.ones(len(xs), 100, dtype=torch.float64)
        for i in range(len(xs)):
            y[i, i % 100] = 1
        yg = y[p:]
        y = y[:p]

        x = torch.stack(xs)
        xg = x[p:]
        x = x[:p]

    elif dataset == "cifar10":
        import torchvision
        from itertools import chain

        assert dim == 3 * 32 * 32
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            lambda x: x.type(torch.float64)
        ])

        trainset = torchvision.datasets.CIFAR10('../cifar10', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10('../cifar10', train=False, transform=transform)

        dataset = []
        for i, xy in enumerate(trainset):
            dataset.append(xy)
            if i % 100 == 0: print("\rcifar10 {:.1f}%".format(100 * i / (len(trainset) + len(testset))), end=" ")
        for i, xy in enumerate(testset):
            dataset.append(xy)
            if i % 100 == 0: print("\rcifar10 {:.1f}%".format(100 * (len(trainset) + i) / (len(trainset) + len(testset))), end=" ")
        print("\rcifar10 complete")

        classes = [[x for x, y in dataset if y == i] for i in range(10)]
        classes = [[xs[i] for i in torch.randperm(len(xs))] for xs in classes]

        xs = list(chain(*zip(*classes)))
        assert p <= len(xs), "p={} and we have {} images".format(p, len(xs))

        y = -torch.ones(len(xs), 10, dtype=torch.float64)
        for i in range(len(xs)):
            y[i, i % 10] = 1
        yg = y[p:]
        y = y[:p]

        x = torch.stack(xs)
        xg = x[p:]
        x = x[:p]

    elif dataset.startswith("cifar2"):
        import torchvision
        from itertools import chain

        if dataset == "cifar2_scale":
            w = int((dim / 3) ** 0.5)
            assert dim == 3 * w * w
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(w),
                torchvision.transforms.ToTensor(),
                lambda x: x.type(torch.float64)
            ])
        elif dataset == "cifar2":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                lambda x: x.type(torch.float64)
            ])
        elif dataset == "cifar2_scale_gray":
            w = int(dim ** 0.5)
            assert dim == w * w
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(w),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
                lambda x: x.type(torch.float64)
            ])
        elif dataset == "cifar2_orth_proj":
            proj = torch.empty(dim, 3 * 32 ** 2, dtype=torch.float64)
            orthogonal_(proj)

            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                lambda x: proj @ x.view(-1).type(torch.float64)
            ])
        elif dataset == "cifar2_pca":
            m, v, e = torch.load('../cifar10/pca.pkl')
            assert dim <= (e > 0).long().sum().item()
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                lambda x: x.view(-1).type(torch.float64),
                lambda x: (x - m) @ v[:, :dim] / e[:dim] ** 0.5,
            ])
        elif dataset == "cifar2_pca_rot":
            m, v, e = torch.load('../cifar10/pca.pkl')
            assert dim <= (e > 0).long().sum().item()
            proj = torch.empty(dim, dim, dtype=torch.float64)
            orthogonal_(proj)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                lambda x: x.view(-1).type(torch.float64),
                lambda x: (x - m) @ v[:, :dim] / e[:dim] ** 0.5,
                lambda x: proj @ x
            ])
        else:
            raise ValueError("unknown dataset")

        target_transform = lambda y: torch.tensor(0 if y in [0, 1, 2, 3, 4] else 1)

        trainset = torchvision.datasets.CIFAR10('../cifar10', train=True, download=True, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.CIFAR10('../cifar10', train=False, transform=transform, target_transform=target_transform)

        dataset = []
        for i, xy in enumerate(trainset):
            dataset.append(xy)
            if i % 100 == 0: print("\rcifar10 {:.1f}%".format(100 * i / (len(trainset) + len(testset))), end=" ")
        for i, xy in enumerate(testset):
            dataset.append(xy)
            if i % 100 == 0: print("\rcifar10 {:.1f}%".format(100 * (len(trainset) + i) / (len(trainset) + len(testset))), end=" ")
        print("\rcifar10 complete")

        classes = [[x for x, y in dataset if y == i] for i in range(2)]
        classes = [[xs[i] for i in torch.randperm(len(xs))] for xs in classes]

        xs = list(chain(*zip(*classes)))
        assert p <= len(xs), "p={} and we have {} images".format(p, len(xs))

        x = torch.stack(xs)
        xg = x[p:]
        x = x[:p]

    else:
        raise ValueError("unknown dataset")

    x = x - x.mean(0)
    x = x.flatten(1).size(1) ** 0.5 * x / x.flatten(1).norm(dim=1).view(-1, *(1,) * (x.ndimension() - 1))
    if y is None:
        y = (torch.arange(p, dtype=torch.float64) % 2) * 2 - 1

    if xg is not None and len(x) > 0:
        xg = xg - xg.mean(0)
        xg = xg.flatten(1).size(1) ** 0.5 * xg / xg.flatten(1).norm(dim=1).view(-1, *(1,) * (x.ndimension() - 1))
        if yg is None:
            yg = (torch.arange(p, p + len(xg), dtype=torch.float64) % 2) * 2 - 1

        return (x, y), (xg, yg)

    return (x, y), None


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


class FC(nn.Module):

    def __init__(self, d, h, depth, act=F.relu, kappa=1, n_classes=1):
        super().__init__()

        layers = nn.ModuleList()

        f = d

        for i in range(depth):
            n = int(h + 1 - (i + 1) / depth)
            lin = nn.Linear(f, n, bias=True)

            layers += [lin]
            f = n

        lin = nn.Linear(f, n_classes, bias=True)

        layers += [lin]
        self.layers = layers

        self.d = d
        self.h = h
        self.depth = depth
        self.act = act
        self.kappa = kappa
        self.n_classes = n_classes
        self.preactivations = None
        self.N = sum(layer.weight.numel() for layer in self.layers)

    def forward(self, x):
        assert self.preactivations is None or self.preactivations == []

        for layer in self.layers[:-1]:
            x = layer(x)
            if self.preactivations is not None:
                self.preactivations.append(x)
            x = self.act(x)

        x = self.layers[-1](x)
        if self.n_classes == 1:
            return x.view(-1)
        else:
            return x


class CNN(nn.Module):

    def __init__(self, d, h, act, kappa=1, n_classes=1):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(d, h, 3, stride=3),
            nn.Conv2d(h, h, 3, stride=2),
            nn.Conv2d(h, n_classes, 3)
        ])

        for name, p in self.layers.named_parameters():
            if 'weight' in name:
                orthogonal_(p)
            if 'bias' in name:
                nn.init.zeros_(p)

        self.act = act
        self.n_classes = n_classes
        self.kappa = kappa

        self.preactivations = None
        self.N = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        assert self.preactivations is None or self.preactivations == []

        for layer in self.layers[:-1]:
            x = layer(x)
            if self.preactivations is not None:
                self.preactivations.append(x)
            x = self.act(x)

        x = self.layers[-1](x)

        x = x.flatten(2).mean(2)

        if self.n_classes == 1:
            return x.view(-1)
        else:
            return x


def expand_basis(basis, vectors, eps=1e-12):
    vectors = iter(vectors)
    assert basis is None or basis.ndimension() == 2

    def extand(basis, vs):
        vs = torch.stack(vs)
        _u, s, v = vs.svd()
        vs = v[:, s > eps].t()
        if basis is None:
            return vs
        vs = torch.cat([basis, vs])
        del basis
        _u, s, v = vs.svd()
        return v[:, s > eps].t()

    while True:
        vs = []
        while len(vs) == 0 or len(vs) < vs[0].size(0):
            try:
                vs.append(next(vectors))
            except StopIteration:
                if len(vs) == 0:
                    return basis
                else:
                    return extand(basis, vs)

        basis = extand(basis, vs)


def n_effective(f, x, n_derive=1):
    assert x.dtype == torch.float64

    basis = expand_basis(None, (gradient(o, f.parameters(), retain_graph=True) for o in f(x).view(-1)))

    if n_derive <= 0:
        return basis.size(0)

    def it():
        for i in x:
            a = torch.tensor(i, requires_grad=True)
            fx = f(a)

            for fxo in fx.view(-1):

                for k in range(n_derive):
                    u = i.clone().normal_()
                    fxo = gradient(fxo, a, create_graph=True) @ u
                    if fxo.grad_fn is None: break  # the derivative is strictly zero

                    yield gradient(fxo, f.parameters(), retain_graph=(k < n_derive - 1))

    while True:
        ws = expand_basis(basis, it())
        if basis.size(0) == ws.size(0):
            return basis.size(0)
        basis = ws


def get_deltas(model, data_x, data_y, chunk=None):
    if chunk is None:
        chunk = len(data_x)

    return torch.cat([
        (model.kappa - model(data_x[i: i + chunk]) * data_y[i: i + chunk]).flatten()
        for i in range(0, len(data_x), chunk)
    ])


def get_gradients(model, data_x):
    gs = []
    for i in range(len(data_x)):
        i = torch.tensor(data_x[i], requires_grad=True, device=data_x.device)
        g = gradient(model(i)[0], i).detach()
        gs.append(g)
    return torch.stack(gs)


def get_mistakes(model, data_x, data_y, chunk=None):
    if chunk is None:
        chunk = len(data_x)

    with torch.no_grad():
        mask = []
        for i in range(0, len(data_x), chunk):
            output = model(data_x[i: i + chunk])  # [p]
            delta = model.kappa - output * data_y[i: i + chunk]  # [p]
            if delta.ndimension() == 2:
                mask.append((delta > 0).any(1))
            else:
                mask.append(delta > 0)
        mask = torch.cat(mask)
    return data_x[mask], data_y[mask]


def get_activities(model, data_x):
    model.eval()

    with torch.no_grad():
        model.preactivations = []
        model(data_x)

        activities = [model.act(a) for a in model.preactivations]
        model.preactivations = None  # free memory

    return activities


def compute_h0(model, deltas, out=None):
    '''
    Compute extensive H0
    '''
    Ntot = sum(p.numel() for p in model.parameters())
    if out is None:
        out = deltas.new_zeros(Ntot, Ntot)  # da Delta_i db Delta_i

    for delta in deltas:
        g = gradient(delta, model.parameters(), retain_graph=True)
        out.add_(g.view(-1, 1) * g.view(1, -1))
    return out


def compute_hp(model, deltas, out=None):
    '''
    Compute extensive Hp
    '''
    from hessian import hessian
    return hessian((deltas.detach() * deltas).sum(), model.parameters(), out=out)  # Delta_i da db Delta_i


def compute_hessian(model, data_x, data_y):
    model.eval()
    p = len(data_x)
    Ntot = sum(p.numel() for p in model.parameters())
    H = data_x.new_zeros(Ntot, Ntot)

    mist_x, mist_y = get_mistakes(model, data_x, data_y, 1024)
    if len(mist_x) == 0:
        return H, H

    h0, hp = H, H.clone()
    for i in range(0, len(mist_x), 1024):
        deltas = get_deltas(model, mist_x[i: i + 1024], mist_y[i: i + 1024])
        compute_h0(model, deltas, out=h0)
        compute_hp(model, deltas, out=hp)

    h0.div_(p)
    hp.div_(p)

    return h0, hp


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

    cons = 0
    loss = 0
    erro = 0

    model.zero_grad()

    for i in range(0, len(data_x), 1024):
        output = model(data_x[i: i + 1024])  # [p, ?]
        delta = model.kappa - output * data_y[i: i + 1024]  # [p, ?]

        if delta.ndimension() == 1:
            l = 0.5 * F.relu(delta).pow(2).sum() / len(data_x)
            cons += (delta > 0).long().sum().item()
            erro += (delta >= model.kappa).long().sum().item()
        else:
            l = 0.5 * F.relu(delta).pow(2).sum(1).sum() / len(data_x)
            cons += (delta > 0).any(1).long().sum().item()
            erro += output.argmax(1).ne(data_y.argmax(1)).long().sum().item()

        l.backward()
        loss += l.item()

    grad_norm = sum(p.grad.pow(2).sum() for p in model.parameters()).pow(0.5).item()
    return cons, loss, grad_norm, erro


def make_a_step(model, optimizer, data_x, data_y, batch_size, chunk=None):
    '''
    data_x [batch, k] (?, dim)
    data_y [batch, class] (?,)
    '''
    model.train()

    if chunk is None:
        chunk = len(data_x)

    with torch.no_grad():

        perm = torch.randperm(data_x.size(0), device=data_x.device)
        mask = []
        for i in range(0, len(data_x), chunk):
            idx = perm[i: i + chunk]
            output = model(data_x[idx])  # [p (, ?)]
            delta = model.kappa - output * data_y[idx]  # [p (, ?)]
            if delta.ndimension() == 2:
                mask.append(idx[(delta > 0).any(1)])
            else:
                mask.append(idx[delta > 0])

            if sum(len(idx) for idx in mask) >= batch_size:
                break
        mask = torch.cat(mask)[:batch_size]

        mist_x = data_x[mask]
        mist_y = data_y[mask]

        if mist_x.size(0) == 0:
            return 0

    deltas = get_deltas(model, mist_x, mist_y)
    loss = 0.5 * deltas.pow(2).sum() / batch_size

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


def load_dir2(directory):
    with FSLocker(os.path.join(directory, "index.pkl.lock")):
        path = os.path.join(directory, "index.pkl")
        if not os.path.isfile(path):
            return

        with open(path, "rb") as f:
            index = pickle.load(f)

    for num in range(len(index)):
        with open(os.path.join(directory, "run_{:04d}.pkl".format(num)), "rb") as f:
            yield pickle.load(f)


def load_dir_desc(directory):
    for run in load_dir(directory):
        yield run['desc']


def load_dir_desc2(directory):
    with FSLocker(os.path.join(directory, "index.pkl.lock")):
        path = os.path.join(directory, "index.pkl")
        if not os.path.isfile(path):
            return

        with open(path, "rb") as f:
            index = pickle.load(f)

        for desc in index:
            yield desc


def load_dir_functional(directory):
    with FSLocker(os.path.join(directory, "index.pkl.lock")):
        path = os.path.join(directory, "index.pkl")
        if not os.path.isfile(path):
            return

        with open(path, "rb") as f:
            index = pickle.load(f)

        for num, desc in enumerate(index):
            fi = os.path.join(directory, "run_{:04d}.pkl".format(num))
            if os.path.isfile(fi):
                def foo():
                    with open(fi, "rb") as f:
                        return pickle.load(f)
                yield desc, foo


def dump_run(directory, run):
    with FSLocker(os.path.join(directory, "output.pkl.lock")):
        with open(os.path.join(directory, "output.pkl"), "ab") as f:
            pickle.dump(run, f)


def dump_run2(directory, run):
    with FSLocker(os.path.join(directory, "index.pkl.lock")):
        path = os.path.join(directory, "index.pkl")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                index = pickle.load(f)
        else:
            index = []

        num = len(index)
        index.append(run['desc'])

        with open(path, "wb") as f:
            pickle.dump(index, f)

    with open(os.path.join(directory, "run_{:04d}.pkl".format(num)), "wb") as f:
        pickle.dump(run, f)


def copy_runs(src, dst):
    ds = {frozenset(run['desc'].items()) for run in load_dir(dst)}
    for run in load_dir(src):
        if frozenset(run['desc'].items()) not in ds:
            dump_run(dst, run)
            ds.add(frozenset(run['desc'].items()))


def copy_runs2(src, dst):
    ds = {frozenset(desc.items()) for desc in load_dir_desc2(dst)}
    for run in load_dir2(src):
        if frozenset(run['desc'].items()) not in ds:
            dump_run2(dst, run)
            ds.add(frozenset(run['desc'].items()))


def load_run(run):
    dtype = torch.float32 if run['args'].precision == "f32" else torch.float64
    torch.set_default_dtype(dtype)

    trainset, testset = get_dataset(run['args'].dataset, run['desc']['p'], run['desc']['dim'], run['data_seed'])
    _x, y = trainset
    n_classes = 1 if y.ndimension() == 1 else y.size(1)

    act = F.relu if run['args'].activation == "relu" else torch.tanh
    model_init = FC(run['desc']['dim'], run['desc']['width'], run['desc']['depth'], act, kappa=run['desc']['kappa'], n_classes=n_classes)
    model_init.type(dtype)
    model_init.load_state_dict(run["init"]['state'])

    model_last = copy.deepcopy(model_init)
    model_last.load_state_dict(run['last']['state'])

    return model_init, model_last, trainset, testset


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


def to_bool(arg):
    if arg == "True": return True
    if arg == "False": return False
    raise ValueError()


def parse_kmg(arg):
    arg = arg.strip()
    if arg.endswith("k"):
        return int(arg[:-1]) * 1000
    if arg.endswith("M"):
        return int(arg[:-1]) * 1000 ** 2
    if arg.endswith("G"):
        return int(arg[:-1]) * 1000 ** 3
    if arg.endswith("T"):
        return int(arg[:-1]) * 1000 ** 4
    return int(arg)
