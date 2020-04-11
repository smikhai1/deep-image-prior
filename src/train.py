from datetime import datetime
from glob import glob
import os
import logging
from shutil import copy
from time import time
import torch
from torch import nn
from tqdm import tqdm

from src.config import Config
from src.models import UNet
from src.utils import load_img, save_img, central_crop, numpy2tensor, tensor2numpy

# TODO: compare the architecture with the original implementation
# TODO: add support of flexible input initiailization
# TODO: add support of configurable optimizer and loss

class Experiment:

    def __init__(self, config_path):
        config = Config(config_path)
        self.config_path = config_path
        self.experiment_name = config.experiment_name
        self.paths = config.paths
        self.training = config.training
        self.data_conf = config.data
        self._check_dirs()

        model_conf = config.model
        self.in_channels = int(model_conf['in_channels'])
        self.out_channels = int(model_conf['out_channels'])
        self.n_filters = list(map(int, model_conf['n_filters']))
        self.n_skips = list(map(int, model_conf['n_skips']))
        self.k_d = int(model_conf['k_d'])
        self.k_u = int(model_conf['k_u'])
        self.k_s = int(model_conf['k_s'])
        self.upsampling = model_conf['upsampling']
        self.num_scales = len(self.n_filters)

    def _check_dirs(self):
        logging.info("Checking directories ...")
        if not os.path.exists(self.paths['data']):
            raise RuntimeError(f"Data doesn't exist at {self.paths['data']}")
        experiment_dir = os.path.join('experiments', self.experiment_name)
        logs_dir = os.path.join(experiment_dir, 'logs')
        results_dir = os.path.join(experiment_dir, 'results')
        originals_dir = os.path.join(experiment_dir, 'originals')
        if os.path.exists(experiment_dir):
            logging.warning(f"Directory with the experiment name '{self.experiment_name}' already exists!"
                            "Archiving this directory..."
                            )
            os.rename(experiment_dir, experiment_dir + f"_archive_{str(datetime.now())}")
        os.makedirs(experiment_dir)
        os.makedirs(logs_dir)
        os.makedirs(results_dir)
        os.makedirs(originals_dir)
        copy(self.config_path, experiment_dir)

        self.logs_dir = logs_dir
        self.results_dir = results_dir
        self.originals_dir = originals_dir
        logging.info("Directories checked!")

    def _get_image(self, img_path):
        raw_img = load_img(img_path, to_rgb=True, color_img=self.data_conf['color_image'])
        height, width = raw_img.shape[:2]
        if height % 2**self.num_scales != 0 or width % 2**self.num_scales != 0 or self.data_conf['use_crop']:
            crop_size = int(self.data_conf['crop_size']) if self.data_conf['use_crop'] else min(height, width)
            crop_size = crop_size // 2**self.num_scales * 2**self.num_scales
            raw_img = central_crop(raw_img, crop_size)
        img_name = os.path.basename(img_path)
        save_img(raw_img, os.path.join(self.originals_dir, img_name))
        return raw_img


    def _denoise_image(self, img_path):
        img_name = os.path.basename(img_path)
        logging.info(f"Denoising {img_name} ...")

        device = torch.device(self.training['device'])
        logging.info(f"Device set: {device}")

        model = UNet(self.in_channels,
                     self.out_channels,
                     self.n_filters,
                     self.k_d,
                     self.k_u,
                     self.n_skips,
                     self.k_s,
                     upsampling=self.upsampling
                     )
        model = model.to(device)
        model.train()
        logging.info("Model built!")

        optim = torch.optim.Adam(model.parameters(), lr=float(self.training['lr']))
        loss_layer = nn.MSELoss()
        logging.info("Optimizer built!")

        img = self._get_image(img_path)
        img_tensor = numpy2tensor(img, in_range=(0, 255), out_range=(-1, 1), device=device)
        z_in = torch.rand(1, self.in_channels, img_tensor.shape[2], img_tensor.shape[3], device=device) * 0.1

        results_dump = []
        gen_avg = None
        for i in tqdm(range(int(self.training['max_iter']))):
            optim.zero_grad()
            gen_img = model(z_in)
            loss = loss_layer(gen_img, img_tensor)
            loss.backward()
            optim.step()

            gen_img = gen_img.detach()
            if gen_avg is None:
                gen_avg = gen_img.clone()
            else:
                gen_avg = gen_avg * float(self.training['gamma']) + (1 - float(self.training['gamma'])) * gen_img

            if self.training['use_ema']:
                gen_img = torch.clamp(gen_avg, min=-1.0, max=1.0)
            gen_img = tensor2numpy(gen_img, in_range=(-1, 1), out_range=(0, 255))

            if (i+1) % int(self.training['print_every']) == 0:
                save_img(gen_img, os.path.join(self.logs_dir, img_name.split('.')[0]+f"_{i+1:04}.{img_name.split('.')[1]}"), to_bgr=True)

        save_img(gen_img, os.path.join(self.results_dir, img_name), to_bgr=True)
        logging.info(f"Denoising of {img_name} finished!")

    def run(self):
        logging.info(f"Experiment {self.experiment_name} is running ...")
        tick = time()
        if os.path.isfile(self.paths['data']):
            self._denoise_image(self.paths['data'])
        else:
            img_paths = glob(os.path.join(self.paths['data'], '*.*'))
            for img_path in tqdm(img_paths):
                self._denoise_image(img_path)
        logging.info(f"Experiment {self.experiment_name} finished in {time() - tick:.4f} s.")