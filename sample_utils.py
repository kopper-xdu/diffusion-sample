import torch

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


def ODE(model, x, clip_denoised=False):
    with torch.no_grad():
        device = next(model.parameters()).device

        from scipy import integrate
        def func(t, x):
            t = torch.tensor([t], dtype=torch.long)
            x = torch.tensor(x, device='cuda', dtype=torch.float32)
            x = x.reshape(1, 3, 32, 32)
            z_pred = model(x, t.cuda())
            score = -(1 / one_minus_alphas_cumprod_sqrt[t].to(device)) * z_pred
            return (-1 / 2 * betas[t].to(device) * x - 1 / 2 * betas[t].to(device) * score).cpu().reshape(-1)

        a = integrate.solve_ivp(func, (999, 0), x.reshape(-1).cpu(), atol=1e-5, rtol=1e-5)
        res = torch.tensor(a.y).transpose(0, 1).reshape(-1, 3, 32, 32)

        if clip_denoised:
            res = [torch.clip(x, -1, 1) for x in res]
        return res
