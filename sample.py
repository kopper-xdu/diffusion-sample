from load_model import from_pretrained
from sample_utils import p_sample_loop, VP_SDE_loop, ODE
import torch
from torchvision.utils import save_image
import os
import argparse

device = torch.device("cuda")


def generate(img_nums=64, sample_steps=1000, schedule='linear', is_DDIM=False):
    model = from_pretrained('cifar10')
    model.to(device)
    model.eval()

    x = torch.randn((img_nums, 3, 32, 32), device=device)

    # res = p_sample_loop(model, x, s=sample_steps, DDIM=is_DDIM, schedule='linear')
    # res = VP_SDE_loop(model, x, s=sample_steps, schedule='linear')
    res = ODE(model, x, s=sample_steps, schedule='linear')
    res = [(x + 1) / 2 for x in res]

    res_dir = f'images/sample-steps-{sample_steps}'
    if is_DDIM:
        res_dir += '-DDIM'
    if schedule != 'linear':
        res_dir += '-' + schedule
    os.makedirs(res_dir)
    for i, x in enumerate(res):
        save_image(x, res_dir + '/' + str(i + 1) + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', default=1000, type=int, help='sample-steps')
    parser.add_argument('--img_nums', default=64, type=int)
    parser.add_argument('--DDIM', default=False, action='store_true')
    parser.add_argument('--schedule', default='linear', type=str, help='sample-step-schedule')

    args = parser.parse_args()
    generate(args.img_nums, args.steps, args.schedule, args.DDIM)


