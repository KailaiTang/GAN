import time
import os
import numpy as np
import torch
from apex import amp
import utils
from torch_utils import to_cuda
from utils import load_checkpoint, save_checkpoint, amp_state_has_overflow, wrap_models
from models.generator import Generator
from models.discriminator import Discriminator
from models.running_average_generator import AverageGenerator
from data_tools.dataloaders_v2 import load_dataset
import config_parser
from models import loss
from data_tools.data_utils import denormalize_img
import logger


torch.backends.cudnn.benchmark = True


def init_model(latent_size, start_channel_dim, image_channels):
    discriminator = Discriminator(image_channels, start_channel_dim)

    generator = Generator(start_channel_dim, image_channels, latent_size)
    discriminator, generator = wrap_models([discriminator, generator])
    return discriminator, generator


class Trainer:

    def __init__(self, config):
        self.default_device = "cpu"
        if torch.cuda.is_available():
            self.default_device = "cuda"
        # Set Hyperparameters
        self.batch_size_schedule = config.train_config.batch_size_schedule
        self.dataset = config.dataset
        self.learning_rate = config.train_config.learning_rate
        self.running_average_generator_decay = config.models.generator.running_average_decay
        self.full_validation = config.use_full_validation
        self.load_fraction_of_dataset = config.load_fraction_of_dataset

        # Image settings
        self.current_imsize = 4
        self.image_channels = config.models.image_channels
        self.latent_size = config.models.latent_size
        self.max_imsize = config.max_imsize

        # Logging variables
        self.checkpoint_dir = config.checkpoint_dir
        self.model_name = self.checkpoint_dir.split("/")[-2]
        self.config_path = config.config_path
        self.global_step = 0 # TODO important

        # Transition settings
        self.transition_variable = 1.
        self.transition_iters = config.train_config.transition_iters # 600000
        self.is_transitioning = False
        self.transition_step = 0
        self.start_channel_size = config.models.start_channel_size # 512
        self.latest_switch = 0
        self.opt_level = config.train_config.amp_opt_level
        self.start_time = time.time()
        self.discriminator, self.generator = init_model(self.latent_size,
                                                        self.start_channel_size,
                                                        self.image_channels)
        self.init_running_average_generator()
        self.criterion = loss.WGANLoss(self.discriminator, self.generator, self.opt_level)
        self.logger = logger.Logger(config.summaries_dir, config.generated_data_dir)
        self.num_skipped_steps = 0
        if not self.load_checkpoint():
            self.init_optimizers()

        self.batch_size = self.batch_size_schedule[self.current_imsize]
        self.logger.log_variable("stats/batch_size", self.batch_size)

        self.num_ims_per_log = config.logging.num_ims_per_log
        self.next_log_point = self.global_step
        self.num_ims_per_save_image = config.logging.num_ims_per_save_image
        self.next_image_save_point = self.global_step
        self.num_ims_per_checkpoint = config.logging.num_ims_per_checkpoint
        self.next_validation_checkpoint = self.global_step

        self.static_latent_variable = self.generator.generate_latent_variable(64)
        self.dataloader_train = load_dataset(
            self.dataset, self.batch_size, self.current_imsize, self.full_validation, self.load_fraction_of_dataset)

    def save_transition_checkpoint(self):
        filedir = os.path.join(os.path.dirname(self.config_path), "transition_checkpoints")
        os.makedirs(filedir, exist_ok=True)
        filepath = os.path.join(filedir, f"imsize{self.current_imsize}.ckpt")
        self.save_checkpoint(filepath)

    def save_checkpoint(self, filepath=None):
        if filepath is None:
            filename = "step_{}.ckpt".format(self.global_step)
            filepath = os.path.join(self.checkpoint_dir, filename)
        state_dict = {
            "D": self.discriminator.state_dict(),
            "G": self.generator.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            "transition_step": self.transition_step,
            "is_transitioning": self.is_transitioning,
            "global_step": self.global_step,
            "total_time": self.total_time,
            "running_average_generator": self.running_average_generator.state_dict(),
            "latest_switch": self.latest_switch,
            "current_imsize": self.current_imsize,
            "transition_step": self.transition_step,
            "num_skipped_steps": self.num_skipped_steps
        }
        save_checkpoint(state_dict,
                        filepath,
                        max_keep=2)

    def load_checkpoint(self):
        try:
            map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
            ckpt = load_checkpoint(self.checkpoint_dir,
                                   map_location=map_location)
            # Transition settings
            self.is_transitioning = ckpt["is_transitioning"]
            self.transition_step = ckpt["transition_step"]
            self.current_imsize = ckpt["current_imsize"]
            self.latest_switch = ckpt["latest_switch"]
            self.num_skipped_steps = ckpt["num_skipped_steps"]

            # Tracking stats
            self.global_step = ckpt["global_step"]
            self.start_time = time.time() - ckpt["total_time"] * 60

            # Models
            self.discriminator.load_state_dict(ckpt['D'])

            self.generator.load_state_dict(ckpt['G'])
            self.running_average_generator.load_state_dict(
                ckpt["running_average_generator"])
            to_cuda([self.generator, self.discriminator,
                     self.running_average_generator])
            self.running_average_generator = amp.initialize(
                self.running_average_generator, None, opt_level=self.opt_level)
            self.init_optimizers()
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
            return True
        except FileNotFoundError as e:
            print(e)
            print(' [*] No checkpoint!')
            return False

    def init_running_average_generator(self):
        self.running_average_generator = AverageGenerator(
            self.start_channel_size, self.image_channels,
            self.latent_size, self.running_average_generator_decay)
        self.running_average_generator = wrap_models(
            self.running_average_generator)
        to_cuda(self.running_average_generator)
        self.running_average_generator = amp.initialize(
            self.running_average_generator, None, opt_level=self.opt_level)

    def extend_running_average_generator(self):
        self.running_average_generator.extend()
        to_cuda(self.running_average_generator)
        self.running_average_generator = amp.initialize(
            self.running_average_generator, None, opt_level=self.opt_level)

    def extend_models(self):
        self.discriminator.extend()
        self.generator.extend()
        self.extend_running_average_generator()

        self.current_imsize *= 2

        self.batch_size = self.batch_size_schedule[self.current_imsize]
        self.transition_step += 1

    def update_running_average_generator(self):
        self.running_average_generator.network.update(self.generator)

    def init_optimizers(self):
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=self.learning_rate,
                                            betas=(0.0, 0.99))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                            lr=self.learning_rate,
                                            betas=(0.0, 0.99))
        self.initialize_amp()
        self.criterion.update_optimizers(self.d_optimizer, self.g_optimizer)

    def initialize_amp(self):
        to_cuda([self.generator, self.discriminator])
        [self.generator, self.discriminator], [self.g_optimizer, self.d_optimizer] = amp.initialize(
            [self.generator, self.discriminator],
            [self.g_optimizer, self.d_optimizer],
            opt_level=self.opt_level,
            num_losses=4)

    def save_transition_image(self, before):
        fake_data = self.generator(self.static_latent_variable)
        fake_data = denormalize_img(fake_data.detach())[:8]
        tag = "before" if before else "after"
        imsize = self.current_imsize if before else self.current_imsize // 2
        imname = "transition/{}_{}_".format(tag, imsize)
        self.logger.save_images(imname, fake_data, log_to_writer=False)

    def validate_model(self, real_data):
        import copy  # bad solution
        running_average_generator = copy.deepcopy(self.running_average_generator)
        running_average_generator = running_average_generator.eval().cpu()
        with torch.no_grad():
            fake_data_sample = denormalize_img(
                running_average_generator(self.static_latent_variable.cpu()).data)
        self.logger.save_images("fakes", fake_data_sample,
                                log_to_validation=True)
        self.logger.save_images("reals", denormalize_img(real_data),
                                log_to_validation=True)
        self.running_average_generator.eval().cuda()

    def log_loss_scales(self):
        self.logger.log_variable("amp/num_skipped_steps", self.num_skipped_steps)
        for loss_idx, loss_scaler in enumerate(amp._amp_state.loss_scalers):
            self.logger.log_variable("amp/loss_scale_{}".format(loss_idx),
                                     loss_scaler._loss_scale)

    def train_step(self, real_data):
        self.total_time = (time.time() - self.start_time) / 60
        res = self.criterion.step(real_data)
        while res is None:
            res = self.criterion.step(real_data)
            self.num_skipped_steps += 1
        wasserstein_distance, gradient_pen, real_scores, fake_scores, epsilon_penalty = res
        # self.logger.log_gradients(self.generator)
        if self.global_step >= self.next_log_point and not amp_state_has_overflow():
            time_spent = time.time() - self.batch_start_time
            nsec_per_img = time_spent / (self.global_step - self.next_log_point + self.num_ims_per_log)
            self.logger.log_variable("stats/nsec_per_img", nsec_per_img)
            self.next_log_point = self.global_step + self.num_ims_per_log
            self.batch_start_time = time.time()
            self.log_loss_scales()
            self.logger.log_variable(
                'discriminator/wasserstein-distance',
                wasserstein_distance.item())
            self.logger.log_variable(
                'discriminator/gradient-penalty',
                gradient_pen.item())
            self.logger.log_variable("discriminator/real-score",
                                     real_scores.item())
            self.logger.log_variable("discriminator/fake-score",
                                     fake_scores.mean().item())
            self.logger.log_variable("discriminator/epsilon-penalty",
                                     epsilon_penalty.item())
            self.logger.log_variable("stats/transition-value",
                                     self.transition_variable)
            self.logger.log_variable("stats/batch_size", self.batch_size)
            self.logger.log_variable("stats/learning_rate", self.learning_rate)
            self.logger.log_variable(
                "stats/training_time_minutes", self.total_time)

    def update_transition_value(self):
        self.transition_variable = utils.compute_transition_value(
            self.global_step, self.is_transitioning, self.transition_iters, self.latest_switch
        )
        self.logger.log_variable("stats/transition-value",
                                 self.transition_variable)
        self.discriminator.update_transition_value(self.transition_variable)
        self.generator.update_transition_value(self.transition_variable)
        self.running_average_generator.update_transition_value(self.transition_variable)

    def train(self):
        self.batch_start_time = time.time()
        while True:
            self.update_transition_value()
            self.dataloader_train.update_next_transition_variable(self.transition_variable)
            train_iter = iter(self.dataloader_train)
            next_transition_value = utils.compute_transition_value(
                self.global_step + self.batch_size, self.is_transitioning, self.transition_iters, self.latest_switch
            )
            self.dataloader_train.update_next_transition_variable(next_transition_value)
            for i, real_data in enumerate(train_iter):
                self.logger.update_global_step(self.global_step)

                self.train_step(real_data)

                # Log data
                self.update_running_average_generator()

                if self.global_step >= self.next_image_save_point:
                    self.next_image_save_point = self.global_step + self.num_ims_per_save_image
                    self.validate_model(real_data)
                self.global_step += self.batch_size

                # save checkpoint
                if self.global_step % self.num_ims_per_checkpoint == 0:
                    self.save_checkpoint()

                if i % 4 == 0:
                    self.update_transition_value()
                if self.global_step >= (self.latest_switch + self.transition_iters):
                    self.latest_switch += self.transition_iters
                    if self.is_transitioning:
                        # Stop transitioning
                        self.is_transitioning = False
                        self.update_transition_value()
                        print(f"Stopping transition. Global step: {self.global_step}, transition_variable: {self.transition_variable}, Current imsize: {self.current_imsize}")
                        # self.save_checkpoint()
                    elif self.current_imsize < self.max_imsize:
                        # Save image before transition
                        self.save_transition_checkpoint()
                        # self.save_transition_image(True)
                        self.extend_models()
                        del self.dataloader_train
                        print(f"Start transition. Global step: {self.global_step}, transition_variable: {self.transition_variable}, Current imsize: {self.current_imsize}")
                        self.dataloader_train = load_dataset(
                            self.dataset, self.batch_size, self.current_imsize, self.full_validation, self.load_fraction_of_dataset)
                        self.is_transitioning = True

                        self.init_optimizers()
                        self.update_transition_value()
                        print(f"New transition value: {self.transition_variable}")

                        # Save image after transition
                        # self.save_transition_image(False)
                        break
                if (i + 1) % 4 == 0:
                    next_transition_value = utils.compute_transition_value(
                        self.global_step + self.batch_size, self.is_transitioning, self.transition_iters,
                        self.latest_switch
                    )
                    self.dataloader_train.update_next_transition_variable(next_transition_value)


if __name__ == '__main__':
    config = config_parser.initialize_and_validate_config()
    trainer = Trainer(config)
    trainer.train()
