from load_model import from_pretrained
from sample_utils import ODE
import torch
from torchvision.utils import save_image
import os

device = torch.device("cuda")


def generate():
    model = from_pretrained('cifar10')
    model.to(device)
    model.eval()

    x = torch.randn((3, 32, 32), device=device)

    # res = p_sample_loop(model, x, s=sample_steps, DDIM=is_DDIM, schedule='linear')
    # res = VP_SDE_loop(model, x, s=sample_steps, schedule='linear')
    res = ODE(model, x)
    res = [(x + 1) / 2 for x in res]

    res_dir = f'images/sample'
    os.makedirs(res_dir)
    for i, x in enumerate(res):
        save_image(x, res_dir + '/' + str(i + 1) + '.png')


if __name__ == '__main__':
    generate()


