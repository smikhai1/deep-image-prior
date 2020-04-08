from argparse import ArgumentParser
import torch
from tqdm import tqdm
import os
import cv2
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from src.models.unet import UNet


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--num_iter", type=int, help="Number of optimization steps")
    parser.add_argument("--save_step", type=int, help="Step for saving results to TB")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--result_dir", type=str, help="Path to the dir with the result")
    parser.add_argument("--log_dir", type=str, help="Path to dir with logs")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    parser.add_argument("--image_size", type=int, help="Size of the image")
    parser.add_argument("--use_exp_avg", action="store_true", help="Whether to use exponential moving average")
    parser.add_argument("--beta", type=float, default=0.99, help="Exponential moving average parameter")

    args = parser.parse_args()

    return args

def load_image(image_path, image_size):
    pass

def main():

    args = parse_args()
    writer = SummaryWriter(args.log_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Active device: {device}")
    model = UNet(32, 3, [128]*5, [128]*5, [4]*5, [3]*5, [3]*5, [1]*5, "bilinear")
    model.to(device)


    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    #img = img[100:args.image_size+100, 100:args.image_size+100]
    img = cv2.resize(img, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, axes=(0, 3, 1, 2))

    input_ = torch.rand([1, 32, img.shape[2], img.shape[3]]) * 0.1
    print(input_.shape)

    img = torch.from_numpy(img)
    img = img.to(torch.float32)

    img = img.to(device)
    input_ = input_.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = nn.MSELoss()

    print("[*] Optimization started")
    for i in tqdm(range(1, args.num_iter+1)):
        optimizer.zero_grad()
        out = model(input_)
        loss_val = loss(out, img)
        loss_val.backward()
        optimizer.step()

        if i == 1:
            out_avg = deepcopy(out.detach())
        out_avg = out_avg * args.beta + out * (1 - args.beta)
        writer.add_scalar("loss", loss_val, global_step=i)
        if i % args.save_step == 0:
            out_ = out_avg.detach().cpu()
            out_ = out_[0]
            writer.add_image("result", out_, i)


if __name__ == "__main__":
    main()