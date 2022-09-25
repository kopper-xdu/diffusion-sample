import torch
import numpy as np
from tqdm import tqdm

time_steps = 1000


def linear_schedule(start, end, time_steps):
    return torch.linspace(start, end, time_steps)


betas = linear_schedule(1e-4, 2e-2, time_steps)

alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)
one_minus_alphas_cumprod_sqrt = torch.sqrt(1 - alphas_cumprod)

sqrt_recip_alphas = torch.sqrt(1. / alphas)
alphas_cumprod_prev = torch.cat((torch.tensor([1.]), alphas_cumprod[:-1]), 0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
log_posterior_variance = torch.log(posterior_variance.clamp(min=1e-20))  # 数值稳定


def p_sample(model, x, t, DDIM, var_type, next_t):
    device = x.device
    with torch.no_grad():
        z_pred = model(x, t)
        noise = torch.randn(x.shape, device=device) if t != 0 else torch.zeros(x.shape, device=device)
        if DDIM:
            n = 0
            var_t = n * n * posterior_variance[t]
            x_zero_pred = (x - one_minus_alphas_cumprod_sqrt[t].to(device) * z_pred) / alphas_cumprod_sqrt[t].to(device)
            if next_t is None:
                return x_zero_pred
            mean = torch.sqrt(alphas_cumprod[next_t]).to(device) * x_zero_pred + torch.sqrt(
                1 - alphas_cumprod[next_t] - var_t).to(device) * z_pred
            return mean + torch.sqrt(var_t).to(device) * noise
        else:
            if var_type == 'fixed_large':
                logvar = torch.log(betas[t]).to(device)
            else:
                logvar = log_posterior_variance[t].to(device)
            mean = sqrt_recip_alphas[t].to(device) * (
                    x - betas[t].to(device) / one_minus_alphas_cumprod_sqrt[t].to(device) * z_pred)
            return mean + torch.exp(0.5 * logvar) * noise


def p_sample_loop(model, x, s=1000, clip_denoised=False, DDIM=False, var_type='fixed_large', schedule='linear'):
    device = next(model.parameters()).device
    x = x.to(device)
    res = []

    if schedule == 'linear':
        seq = (np.linspace(0, time_steps - 1, s))
    else:
        seq = (np.linspace(0, np.sqrt(time_steps - 1), s) ** 2)
    seq = [int(s) for s in list(seq)]
    seq = list(set(seq))
    seq.sort(reverse=True)
    for i, t in enumerate(tqdm(seq)):
        if i == len(seq) - 1:
            next_t = None
        else:
            next_t = seq[i + 1]
        x = p_sample(model, x, torch.tensor([t], device=device), DDIM, var_type, next_t)
        res.append(x)
    if clip_denoised:
        res = [torch.clip(x, -1, 1) for x in res]
    return res


def VP_SDE(model, x, t, var_type):
    device = x.device
    with torch.no_grad():
        z_pred = model(x, t)
        score = -(1 / one_minus_alphas_cumprod_sqrt[t].to(device)) * z_pred
        noise = torch.randn(x.shape, device=device) if t != 0 else torch.zeros(x.shape, device=device)
        if var_type == 'fixed_large':
            logvar = torch.log(betas[t]).to(device)
        else:
            logvar = log_posterior_variance[t].to(device)
        mean = (2 - torch.sqrt(alphas[t].to(device))) * x + betas[t].to(device) * score
        # corrector
        return mean + torch.exp(0.5 * logvar) * noise


def VP_SDE_loop(model, x, s=1000, clip_denoised=False, var_type='fixed_large', schedule='linear'):
    device = next(model.parameters()).device
    x = x.to(device)
    res = []

    if schedule == 'linear':
        seq = (np.linspace(0, time_steps - 1, s))
    else:
        seq = (np.linspace(0, np.sqrt(time_steps - 1), s) ** 2)
    seq = [int(s) for s in list(seq)]
    seq = list(set(seq))
    seq.sort(reverse=True)
    for i, t in enumerate(tqdm(seq)):
        x = VP_SDE(model, x, torch.tensor([t], device=device), var_type)
        res.append(x)
    if clip_denoised:
        res = [torch.clip(x, -1, 1) for x in res]
    return res


def ODE(model, x, s=1000, clip_denoised=False, schedule='linear'):
    with torch.no_grad():
        device = next(model.parameters()).device
        res = []

        if schedule == 'linear':
            seq = (np.linspace(0, time_steps - 1, s))
        else:
            seq = (np.linspace(0, np.sqrt(time_steps - 1), s) ** 2)
        seq = [int(s) for s in list(seq)]
        seq = list(set(seq))
        seq.sort(reverse=True)

        from scipy import integrate
        def func(t, x):
            t = torch.tensor([t], dtype=torch.long)
            x = torch.tensor(x, device='cuda', dtype=torch.float32)
            x = x.reshape(1, 3, 32, 32)
            z_pred = model(x, t.cuda())
            score = -(1 / one_minus_alphas_cumprod_sqrt[t].to(device)) * z_pred
            return (-1 / 2 * betas[t].to(device) * x - 1 / 2 * betas[t].to(device) * score).cpu().reshape(-1)

        a = integrate.solve_ivp(func, (999, 0), x[0].reshape(-1).cpu(), atol=1e-5, rtol=1e-5)
        res = torch.tensor(a.y).transpose(0, 1).reshape(-1, 3, 32, 32)

        if clip_denoised:
            res = [torch.clip(x, -1, 1) for x in res]
        return res
