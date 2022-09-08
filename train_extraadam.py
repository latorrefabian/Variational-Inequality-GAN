"""
Original train extraadam script
"""
import pdb

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import wandb
from tqdm import tqdm
from autoparse import autoparse

import models
import utils
from optim import ExtraAdam


class Logger:
    def log_normalized_images(
            self,
            images: torch.Tensor,
            name: str,
            step: int) -> None:
        images = torchvision.utils.make_grid(
            images,
            normalize=True,
            value_range=(-1.0, 1.0),
            nrow=10).to('cpu')
        images = images.permute(1, 2, 0)
        images = wandb.Image(images.numpy(), caption="")
        wandb.log(
            {name: images},
            step=step,
            commit=True)


def main(
        device: str = 'cpu',
        batch_size: int = 200,
        num_iter: int = 500000,
        learning_rate_dis: float = 5e-4,
        learning_rate_gen: float = 5e-5,
        beta1: float = 0.5,
        beta2: float = 0.9,
        ema: float = 0.9999,
        num_latent: int = 128,
        num_filters_dis: int = 128,
        num_filters_gen: int = 128,
        gradient_penalty: float = 10.,
        mode: str = 'wgan',
        distribution: str = 'normal',
        batchnorm_dis: bool = False,
        seed: int = 1318,
        ):
    logger = Logger()
    torch.manual_seed(seed)
    np.random.seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transform,
        download=True)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transform,
        download=True)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size)

    gen = models.ResNet32Generator(
        n_in=num_latent,
        n_out=3,
        num_filters=num_filters_gen,
        batchnorm=True)

    dis = models.ResNet32Discriminator(
        n_in=3,
        n_out=1,
        num_filters=num_filters_dis,
        batchnorm=batchnorm_dis)

    gen = gen.to(device)
    dis = dis.to(device)

    gen.apply(lambda x: utils.weight_init(x, mode='normal'))
    dis.apply(lambda x: utils.weight_init(x, mode='normal'))

    dis_optimizer = ExtraAdam(
        dis.parameters(),
        lr=learning_rate_dis,
        betas=(beta1, beta2))

    gen_optimizer = ExtraAdam(
        gen.parameters(),
        lr=learning_rate_gen,
        betas=(beta1, beta2))

    examples, _ = next(iter(testloader))
    logger.log_normalized_images(examples, name='reference_images', step=0)

    z_examples = utils.sample(distribution, (100, num_latent))
    z_examples = z_examples.to(device)

    gen_param_avg = []
    gen_param_ema = []

    for param in gen.parameters():
        gen_param_avg.append(param.data.clone())
        gen_param_ema.append(param.data.clone())

    n_gen_update = 0
    n_iteration_t = 0
    pbar = tqdm(total=num_iter)
    while n_gen_update < num_iter:
        pbar.set_description('n_gen_update: ' + str(n_gen_update))
        penalty = Variable(torch.Tensor([0.]))
        penalty = penalty.to(device)
        for i, (x_true, _) in enumerate(trainloader):
            print(i)
            x_true = Variable(x_true)
            z = Variable(utils.sample(distribution, (len(x_true), num_latent)))
            x_true = x_true.to(device)
            z = z.to(device)
            x_gen = gen(z)
            p_true, p_gen = dis(x_true), dis(x_gen)
            gen_loss = utils.compute_gan_loss(p_true, p_gen, mode=mode)
            dis_loss = - gen_loss.clone()
            if gradient_penalty:
                penalty = dis.get_penalty(x_true.data, x_gen.data)
                dis_loss += gradient_penalty * penalty

            for p in gen.parameters():
                p.requires_grad = False
            dis_optimizer.zero_grad()
            dis_loss.backward(retain_graph=True)

            if (n_iteration_t + 1) % 2 != 0:
                dis_optimizer.extrapolation()
            else:
                dis_optimizer.step()

            for p in gen.parameters():
                p.requires_grad = True

            for p in dis.parameters():
                p.requires_grad = False
            gen_optimizer.zero_grad()
            gen_loss.backward()

            if (n_iteration_t + 1) % 2 != 0:
                gen_optimizer.extrapolation()
            else:
                n_gen_update += 1
                pbar.update(1)
                gen_optimizer.step()
                for j, param in enumerate(gen.parameters()):
                    gen_param_avg[j] = (
                            gen_param_avg[j] * n_gen_update / (n_gen_update+1.)
                            + param.data.clone() / (n_gen_update+1.))
                    gen_param_ema[j] = gen_param_ema[j]*ema+ param.data.clone()*(1-ema)

            for p in dis.parameters():
                p.requires_grad = True

            n_iteration_t += 1

        x_gen = gen(z_examples)
        logger.log_normalized_images(x_gen, name='my_image', step=n_gen_update)


if __name__ == '__main__':
    args = autoparse(main, verbose=True)
    wandb.init(
        project='fairgan_cifar10',
        config=args,
        name='var_ineq_gan',
        dir='.wandb')
    main(**args)

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', choices=('resnet', 'dcgan'), default='resnet')
# parser.add_argument('--cuda', action='store_true')
# parser.add_argument('-bs','--batch-size', default=200, type=int)
# parser.add_argument('--num-iter', default=500000, type=int)
# parser.add_argument('-lrd', '--learning-rate-dis', default=5e-4, type=float)
# parser.add_argument('-lrg', '--learning-rate-gen', default=5e-5, type=float)
# parser.add_argument('-b1','--beta1', default=0.5, type=float)
# parser.add_argument('-b2','--beta2', default=0.9, type=float)
# parser.add_argument('-ema', default=0.9999, type=float)
# parser.add_argument('-nz','--num-latent', default=128, type=int)
# parser.add_argument('-nfd','--num-filters-dis', default=128, type=int)
# parser.add_argument('-nfg','--num-filters-gen', default=128, type=int)
# parser.add_argument('-gp', '--gradient-penalty', default=10, type=float)
# parser.add_argument('-m', '--mode', choices=('gan','ns-gan', 'wgan'), default='wgan')
# parser.add_argument('-c', '--clip', default=0.01, type=float)
# parser.add_argument('-d', '--distribution', choices=('normal', 'uniform'), default='normal')
# parser.add_argument('--batchnorm-dis', action='store_true')
# parser.add_argument('--seed', default=1234, type=int)
# parser.add_argument('--default', action='store_true')
# args = parser.parse_args()


# CUDA = args.cuda
# MODEL = args.model
# GRADIENT_PENALTY = args.gradient_penalty
# BATCH_SIZE = args.batch_size
# N_ITER = args.num_iter
# LEARNING_RATE_G = args.learning_rate_gen
# LEARNING_RATE_D = args.learning_rate_dis
# BETA_1 = args.beta1
# BETA_2 = args.beta2
# BETA_EMA = args.ema
# N_LATENT = args.num_latent
# N_FILTERS_G = args.num_filters_gen
# N_FILTERS_D = args.num_filters_dis
# MODE = args.mode
# CLIP = args.clip
# DISTRIBUTION = args.distribution
# BATCH_NORM_G = True
# BATCH_NORM_D = args.batchnorm_dis
# N_SAMPLES = 50000
# RESOLUTION = 32
# N_CHANNEL = 3
# START_EPOCH = 0
# EVAL_FREQ = 10000
# SEED = args.seed
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# n_gen_update = 0
# n_dis_update = 0
# total_time = 0
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=(0.5, 0.5, 0.5),
#         std=(0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=True,
#     transform=transform,
#     download=True)
#
# trainloader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=BATCH_SIZE,
#     shuffle=True)
#
# testset = torchvision.datasets.CIFAR10(
#     root='./data',
#     train=False,
#     transform=transform,
#     download=True)
# testloader = torch.utils.data.DataLoader(
#     testset,
#     batch_size=BATCH_SIZE)
#
# if not os.path.exists(os.path.join('output', 'checkpoints')):
#     os.makedirs(os.path.join('output', 'checkpoints'))
# if not os.path.exists(os.path.join('output', 'gen')):
#     os.makedirs(os.path.join('output', 'gen'))
#
# gen = models.ResNet32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, BATCH_NORM_G)
# dis = models.ResNet32Discriminator(N_CHANNEL, 1, N_FILTERS_D, BATCH_NORM_D)
#
# if CUDA:
#     gen = gen.cuda(0)
#     dis = dis.cuda(0)
#
# gen.apply(lambda x: utils.weight_init(x, mode='normal'))
# dis.apply(lambda x: utils.weight_init(x, mode='normal'))
#
# dis_optimizer = ExtraAdam(dis.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))
# gen_optimizer = ExtraAdam(gen.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))
#
# examples, _ = next(iter(testloader))
# torchvision.utils.save_image(
#     tensor=utils.unormalize(examples),
#     fp=os.path.join('output', 'examples.png'),
#     nrow=10)
#
# z_examples = utils.sample(DISTRIBUTION, (100, N_LATENT))
# if CUDA:
#     z_examples = z_examples.cuda(0)
#
# gen_param_avg = []
# gen_param_ema = []
#
# for param in gen.parameters():
#     gen_param_avg.append(param.data.clone())
#     gen_param_ema.append(param.data.clone())
#
# print('Training...')
# n_iteration_t = 0
# gen_inception_score = 0
# while n_gen_update < N_ITER:
#     print('n_gen_update: ' + str(n_gen_update))
#     avg_loss_G = 0
#     avg_loss_D = 0
#     avg_penalty = 0
#     num_samples = 0
#     penalty = Variable(torch.Tensor([0.]))
#     if CUDA:
#         penalty = penalty.cuda(0)
#     for i, (x_true, _) in enumerate(trainloader):
#         print(i)
#         x_true = Variable(x_true)
#         z = Variable(utils.sample(DISTRIBUTION, (len(x_true), N_LATENT)))
#         if CUDA:
#             x_true = x_true.cuda(0)
#             z = z.cuda(0)
#
#         x_gen = gen(z)
#         p_true, p_gen = dis(x_true), dis(x_gen)
#
#         gen_loss = utils.compute_gan_loss(p_true, p_gen, mode=MODE)
#         dis_loss = - gen_loss.clone()
#         if GRADIENT_PENALTY:
#             penalty = dis.get_penalty(x_true.data, x_gen.data)
#             dis_loss += GRADIENT_PENALTY*penalty
#
#         for p in gen.parameters():
#             p.requires_grad = False
#         dis_optimizer.zero_grad()
#         dis_loss.backward(retain_graph=True)
#
#         if (n_iteration_t+1)%2 != 0:
#             dis_optimizer.extrapolation()
#         else:
#             dis_optimizer.step()
#
#         for p in gen.parameters():
#             p.requires_grad = True
#
#         for p in dis.parameters():
#             p.requires_grad = False
#         gen_optimizer.zero_grad()
#         gen_loss.backward()
#
#         if (n_iteration_t+1)%2 != 0:
#             gen_optimizer.extrapolation()
#         else:
#             n_gen_update += 1
#             gen_optimizer.step()
#             for j, param in enumerate(gen.parameters()):
#                 gen_param_avg[j] = gen_param_avg[j]*n_gen_update/(n_gen_update+1.) + param.data.clone()/(n_gen_update+1.)
#                 gen_param_ema[j] = gen_param_ema[j]*BETA_EMA+ param.data.clone()*(1-BETA_EMA)
#
#         for p in dis.parameters():
#             p.requires_grad = True
#
#
#         n_iteration_t += 1
#
#     x_gen = gen(z_examples)
#     grid = torchvision.utils.make_grid(
#         x_gen,
#         normalize=True,
#         value_range=(-1.0, 1.0),
#         nrow=10).to('cpu')
#     grid = grid.permute(1, 2, 0)
#     image = wandb.Image(grid.numpy(), caption="")
#     wandb.log(
#         {'my_image': image},
#         step=n_gen_update,
#         commit=False)
#     x = utils.unormalize(x_gen)
#     other_grid = torchvision.utils.make_grid(
#         x,
#         nrow=10).to('cpu')
#     other_grid = other_grid.permute(1, 2, 0)
#     image = wandb.Image(other_grid.numpy(), caption="")
#     wandb.log(
#         {'original_images': image},
#         step=n_gen_update,
#         commit=True)
