# pylint: disable=E1101, C0103, C0111, C0301
import torch
from torch.optim import Optimizer


class FIRE(Optimizer):

    def __init__(self, params, dt_max, a_start=0.1, N_min=10, f_inc=1.1, f_dec=0.5, f_alpha=0.99):
        defaults = dict(dt_max=dt_max, a_start=a_start, N_min=N_min, f_inc=f_inc, f_dec=f_dec, f_alpha=f_alpha)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for i, group in enumerate(self.param_groups):
            if i not in self.state:
                self.state[i] = dict(
                    dt=group['dt_max'],
                    a=group['a_start'],
                    V=[torch.zeros_like(p) for p in group['params']],
                    it=0,
                    cut=0,
                )
            state = self.state[i]

            ff = 0
            vv = 0
            vf = 0
            for v, p in zip(state['V'], group['params']):
                vv += v.norm()**2
                if p.grad is None:
                    continue
                f = -p.grad.detach()
                ff += f.norm()**2
                vf += (v * f).sum()

            if ff == 0:
                continue

            if vf < 0:
                for v in state['V']:
                    v.zero_()
                state['cut'] = state['it']
                state['dt'] *= group['f_dec']
                state['a'] = group['a_start']
            if vf >= 0:
                cF = state['a'] * (vv / ff)**0.5
                cV = 1 - state['a']

                V = []
                for v, p in zip(state['V'], group['params']):
                    v = cV * v
                    if p.grad is not None:
                        v -= cF * p.grad.detach()
                    V.append(v)
                state['V'] = V

                if state['it'] - state['cut'] > group['N_min']:
                    state['dt'] = min(state['dt'] * group['f_inc'], group['dt_max'])
                    state['a'] *= group['f_alpha']

            # update parameter
            for v, p in zip(state['V'], group['params']):
                if p.grad is not None:
                    v.sub_(state['dt'] * p.grad.detach())
                p.data.add_(state['dt'] * v)

            state['it'] += 1

        return loss
