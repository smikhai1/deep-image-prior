import cv2
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

# TODO: add conversion from tensor to numpy array (and vice versa)
# TODO: add conversion form one range to another
# TODO: add visualization of evolution
# TODO: add computing of exponential weighted average
# TODO: add support of flexible input initiailization
# TODO: add support of configurable optimizer and loss

class Experiment:

    def __init__(self, config_path):
        config = Config(config_path)
        self.config_path = config_path
        self.experiment_name = config.experiment_name
        self.paths = config.paths
        self.training = config.training
        self.model_conf = config.model
        self.data = config.data
        self._check_dirs()

    def _check_dirs(self):
        logging.info("Checking directories ...")
        if not os.path.exists(self.paths['data']):
            raise RuntimeError(f"Data doesn't exist at {self.paths['data']}")
        experiment_dir = os.path.join('experiments', self.experiment_name)
        logs_dir = os.path.join(experiment_dir, 'logs')
        results_dir = os.path.join(experiment_dir, 'results')
        if os.path.exists(experiment_dir):
            logging.warning(f"Directory with the experiment name '{self.experiment_name}' already exists!"
                            "Archiving this directory..."
                            )
            os.rename(experiment_dir, experiment_dir + f"archive_{str(datetime.now())}")
        os.makedirs(experiment_dir)
        os.makedirs(logs_dir)
        os.makedirs(results_dir)
        copy(self.config_path, experiment_dir)

        self.logs_dir = logs_dir
        self.results_dir = results_dir
        logging.info("Directories checked!")

    def _denoise_image(self, img_path):
        img_name = os.path.basename(img_path)
        logging.info(f"Denoising {img_name} ...")

        device = self.training['device']
        logging.info(f"Device set: {device}")

        model = UNet(**self.model_conf)
        model = model.to(device)
        model.train()
        logging.info("Model built!")

        optim = torch.optim.Adam(model.parameters(), lr=self.training['lr'])
        loss_layer = nn.MSELoss()
        logging.info("Optimizer built!")

        img = self._get_image(img_path)
        img_tensor = numpy2tensor(img, device=device)
        z_in = torch.rand(1, self.model['in_channels'], img.shape[2], img.shape[3], device=device) * 0.1

        results_dump = []
        for i in range(self.training['max_iter']):
            optim.zero_grad()
            gen_img = model(z_in)
            loss = loss_layer(gen_img, img_tensor)
            loss.backward()
            optim.step()

            gen_img = gen_img.detach().cpu().numpy()
            gen_img = tensor2numpy(gen_img)
            if (i+1) % self.training['print_every']:
                results_dump.append(gen_img)

        cv2.imwrite(os.path.join(self.results_dir, img_name), gen_img)
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