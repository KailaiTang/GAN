import time
import os
import torch
from apex import amp
import utils
import torchvision
from torch_utils import to_cuda
from utils import load_checkpoint, amp_state_has_overflow, wrap_models
from models.generator import Generator
from models.discriminator import Discriminator
from models.running_average_generator import AverageGenerator
from data_tools.dataloaders_v2 import load_dataset
from models import loss
from data_tools.data_utils import denormalize_img
from argparse import ArgumentParser
from config_parser import load_config, validate_config, print_config
from collections import namedtuple
import logger
import config_parser

torch.backends.cudnn.benchmark = True


def init_model(latent_size, start_channel_dim, image_channels):
    discriminator = Discriminator(image_channels, start_channel_dim)

    generator = Generator(start_channel_dim, image_channels, latent_size)
    discriminator, generator = wrap_models([discriminator, generator])
    return discriminator, generator


class SampleGenerator(object):

    def __init__(self, config):
        self.default_device = torch.device('cpu')
        if torch.cuda.is_available():
            self.default_device = torch.device('cuda')
        # Set Hyperparameters
        self.running_average_generator_decay = config.models.generator.running_average_decay

        # Image settings
        self.current_imsize = 4
        self.image_channels = config.models.image_channels
        self.latent_size = config.models.latent_size

        # Logging variables
        self.checkpoint_path = config.model_path

        # Transition settings
        self.transition_variable = 1.
        self.transition_iters = config.train_config.transition_iters
        self.is_transitioning = False
        self.transition_step = 0
        self.start_channel_size = config.models.start_channel_size
        self.latest_switch = 0
        self.opt_level = config.train_config.amp_opt_level
        self.start_time = time.time()
        self.discriminator, self.generator = init_model(self.latent_size,
                                                        self.start_channel_size,
                                                        self.image_channels)
        self.init_running_average_generator()
        self.num_skipped_steps = 0

        self.num_ims_per_log = config.logging.num_ims_per_log
        self.num_ims_per_save_image = config.logging.num_ims_per_save_image
        self.num_ims_per_checkpoint = config.logging.num_ims_per_checkpoint

        self.load_checkpoint()

    def load_checkpoint(self):
        try:
            map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
            ckpt = torch.load(self.checkpoint_path, map_location=map_location)
            # Transition settings
            self.is_transitioning = ckpt["is_transitioning"]
            self.transition_step = ckpt["transition_step"]
            self.current_imsize = ckpt["current_imsize"]
            self.latest_switch = ckpt["latest_switch"]
            self.num_skipped_steps = ckpt["num_skipped_steps"]

            # Tracking stats
            self.start_time = time.time() - ckpt["total_time"] * 60

            # Models
            self.discriminator.load_state_dict(ckpt['D'])

            self.generator.load_state_dict(ckpt['G'])
            self.running_average_generator.load_state_dict(
                ckpt["running_average_generator"])
            to_cuda([self.generator, self.discriminator,
                     self.running_average_generator])
        except FileNotFoundError as e:
            print(e)
            print(' [*] No checkpoint!')

    def init_running_average_generator(self):
        self.running_average_generator = AverageGenerator(
            self.start_channel_size, self.image_channels,
            self.latent_size, self.running_average_generator_decay)
        self.running_average_generator = wrap_models(
            self.running_average_generator)
        to_cuda(self.running_average_generator)

    def generate(self, num_samples):
        self.running_average_generator.eval()
        latent_variable = self.generator.generate_latent_variable(num_samples)
        with torch.no_grad():
            fake_data_sample = denormalize_img(
                self.running_average_generator(latent_variable).data)
        self.save_images("fitsdata/results", fake_data_sample)

    def save_images(self, image_dir, images, tag="fakes"):
        imsize = images.shape[2]

        for idx, img in enumerate(images):
            filename = "{0}_{1}_{2}x{2}.jpg".format(tag, idx, imsize)
            filepath = os.path.join(image_dir, filename)

            img = torchvision.transforms.functional.to_pil_image(img.cpu())
            img.save(filepath)


def initialize_and_validate_config(additional_arguments=[]):
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="Set the config of the model")
    parser.add_argument("-m", "--model", help="Set the path of the model")
    parser.add_argument("-n", "--num_samples", type=int, default=32, help="Set the number of sample the model to gen")
    for additional_arg in additional_arguments:
        parser.add_argument(f'--{additional_arg["name"]}', default=additional_arg["default"])

    args = parser.parse_args()
    assert os.path.isfile(args.config), "Did not find config file:".format(args.config)

    config = load_config(args.config)

    config_dir = os.path.dirname(args.config)

    new_config_fields = {
        "num_samples": args.num_samples,
        "config_path": args.config,
        "model_path": args.model,
        "checkpoint_dir": os.path.join(config_dir, "checkpoints"),
        "generated_data_dir": os.path.join(config_dir, "generated_data"),
        "summaries_dir": os.path.join(config_dir, "summaries")
    }
    for additional_arg in additional_arguments:
        new_config_fields[additional_arg["name"]] = vars(args)[additional_arg["name"]]
    config = namedtuple("Config", list(config._asdict().keys()) + list(new_config_fields.keys()))(
        *(list(config._asdict().values()) + list(new_config_fields.values()))
    )

    validate_config(config)

    print_config(config, first=True)
    return config


if __name__ == '__main__':
    config = initialize_and_validate_config()
    generator = SampleGenerator(config)
    generator.generate(config.num_samples)
