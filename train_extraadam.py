"""
Original train extraadam script
"""
import pdb

import torch
from torch.autograd import Variable
import time
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import json
import csv

import models
import utils
from optim import ExtraAdam

def main(
        output: str,
        cuda: bool = False,
        batch_size: int = 64,
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
        clip: float = 0.1,
        distribution: str = 'normal',
        batchnorm_dis: float = False,
        seed: int = 1318,
        ):
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
        batch_size=BATCH_SIZE,
        shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transform,
        download=True)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE)

    gen = models.ResNet32Generator(
        N_LATENT,
        N_CHANNEL,
        N_FILTERS_G,
        BATCH_NORM_G)
    dis = models.ResNet32Discriminator(
        N_CHANNEL,
        1,
        N_FILTERS_D,
        BATCH_NORM_D)

    if CUDA:
        gen = gen.cuda(0)
        dis = dis.cuda(0)

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


parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--model', choices=('resnet', 'dcgan'), default='resnet')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('-bs','--batch-size', default=64, type=int)
parser.add_argument('--num-iter', default=500000, type=int)
parser.add_argument('-lrd', '--learning-rate-dis', default=5e-4, type=float)
parser.add_argument('-lrg', '--learning-rate-gen', default=5e-5, type=float)
parser.add_argument('-b1','--beta1', default=0.5, type=float)
parser.add_argument('-b2','--beta2', default=0.9, type=float)
parser.add_argument('-ema', default=0.9999, type=float)
parser.add_argument('-nz','--num-latent', default=128, type=int)
parser.add_argument('-nfd','--num-filters-dis', default=128, type=int)
parser.add_argument('-nfg','--num-filters-gen', default=128, type=int)
parser.add_argument('-gp', '--gradient-penalty', default=10, type=float)
parser.add_argument('-m', '--mode', choices=('gan','ns-gan', 'wgan'), default='wgan')
parser.add_argument('-c', '--clip', default=0.01, type=float)
parser.add_argument('-d', '--distribution', choices=('normal', 'uniform'), default='normal')
parser.add_argument('--batchnorm-dis', action='store_true')
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--default', action='store_true')
args = parser.parse_args()

CUDA = args.cuda
MODEL = args.model
GRADIENT_PENALTY = args.gradient_penalty
OUTPUT_PATH = args.output
BATCH_SIZE = args.batch_size
N_ITER = args.num_iter
LEARNING_RATE_G = args.learning_rate_gen # It is really important to set different learning rates for the discriminator and generator
LEARNING_RATE_D = args.learning_rate_dis
BETA_1 = args.beta1
BETA_2 = args.beta2
BETA_EMA = args.ema
N_LATENT = args.num_latent
N_FILTERS_G = args.num_filters_gen
N_FILTERS_D = args.num_filters_dis
MODE = args.mode
CLIP = args.clip
DISTRIBUTION = args.distribution
BATCH_NORM_G = True
BATCH_NORM_D = args.batchnorm_dis
N_SAMPLES = 50000
RESOLUTION = 32
N_CHANNEL = 3
START_EPOCH = 0
EVAL_FREQ = 10000
SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
n_gen_update = 0
n_dis_update = 0
total_time = 0

if GRADIENT_PENALTY:
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, '%s_%s-gp'%(MODEL, MODE), '%s/lrd=%.1e_lrg=%.1e/s%i/%i'%('extra_adam', LEARNING_RATE_D, LEARNING_RATE_G, SEED, int(time.time())))
else:
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, '%s_%s'%(MODEL, MODE), '%s/lrd=%.1e_lrg=%.1e/s%i/%i'%('extra_adam', LEARNING_RATE_D, LEARNING_RATE_G, SEED, int(time.time())))

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
    batch_size=BATCH_SIZE,
    shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transform,
    download=True)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE)

print('Init....')
if not os.path.exists(os.path.join(OUTPUT_PATH, 'checkpoints')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'checkpoints'))
if not os.path.exists(os.path.join(OUTPUT_PATH, 'gen')):
    os.makedirs(os.path.join(OUTPUT_PATH, 'gen'))

if MODEL == "resnet":
    gen = models.ResNet32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, BATCH_NORM_G)
    dis = models.ResNet32Discriminator(N_CHANNEL, 1, N_FILTERS_D, BATCH_NORM_D)
elif MODEL == "dcgan":
    gen = models.DCGAN32Generator(N_LATENT, N_CHANNEL, N_FILTERS_G, batchnorm=BATCH_NORM_G)
    dis = models.DCGAN32Discriminator(N_CHANNEL, 1, N_FILTERS_D, batchnorm=BATCH_NORM_D)

if CUDA:
    gen = gen.cuda(0)
    dis = dis.cuda(0)

gen.apply(lambda x: utils.weight_init(x, mode='normal'))
dis.apply(lambda x: utils.weight_init(x, mode='normal'))

dis_optimizer = ExtraAdam(dis.parameters(), lr=LEARNING_RATE_D, betas=(BETA_1, BETA_2))
gen_optimizer = ExtraAdam(gen.parameters(), lr=LEARNING_RATE_G, betas=(BETA_1, BETA_2))

dataiter = iter(testloader)
examples, labels = dataiter.next()
torchvision.utils.save_image(
    tensor=utils.unormalize(examples),
    fp=os.path.join(OUTPUT_PATH, 'examples.png'),
    nrow=10)

z_examples = utils.sample(DISTRIBUTION, (100, N_LATENT))
if CUDA:
    z_examples = z_examples.cuda(0)

gen_param_avg = []
gen_param_ema = []
for param in gen.parameters():
    gen_param_avg.append(param.data.clone())
    gen_param_ema.append(param.data.clone())

f = open(os.path.join(OUTPUT_PATH, 'results.csv'), 'ab')
f_writter = csv.writer(f)

print('Training...')
n_iteration_t = 0
gen_inception_score = 0
while n_gen_update < N_ITER:
    print('n_gen_update: ' + str(n_gen_update))
    t = time.time()
    avg_loss_G = 0
    avg_loss_D = 0
    avg_penalty = 0
    num_samples = 0
    penalty = Variable(torch.Tensor([0.]))
    if CUDA:
        penalty = penalty.cuda(0)
    for i, data in enumerate(trainloader):
        print(i)
        _t = time.time()
        x_true, _ = data
        x_true = Variable(x_true)

        z = Variable(utils.sample(DISTRIBUTION, (len(x_true), N_LATENT)))
        if CUDA:
            x_true = x_true.cuda(0)
            z = z.cuda(0)

        x_gen = gen(z)
        p_true, p_gen = dis(x_true), dis(x_gen)

        gen_loss = utils.compute_gan_loss(p_true, p_gen, mode=MODE)
        dis_loss = - gen_loss.clone()
        if GRADIENT_PENALTY:
            penalty = dis.get_penalty(x_true.data, x_gen.data)
            dis_loss += GRADIENT_PENALTY*penalty

        for p in gen.parameters():
            p.requires_grad = False
        dis_optimizer.zero_grad()
        dis_loss.backward(retain_graph=True)

        if (n_iteration_t+1)%2 != 0:
            dis_optimizer.extrapolation()
        else:
            dis_optimizer.step()

        for p in gen.parameters():
            p.requires_grad = True

        for p in dis.parameters():
            p.requires_grad = False
        gen_optimizer.zero_grad()
        gen_loss.backward()

        if (n_iteration_t+1)%2 != 0:
            gen_optimizer.extrapolation()
        else:
            n_gen_update += 1
            gen_optimizer.step()
            for j, param in enumerate(gen.parameters()):
                gen_param_avg[j] = gen_param_avg[j]*n_gen_update/(n_gen_update+1.) + param.data.clone()/(n_gen_update+1.)
                gen_param_ema[j] = gen_param_ema[j]*BETA_EMA+ param.data.clone()*(1-BETA_EMA)

        for p in dis.parameters():
            p.requires_grad = True

        if MODE =='wgan' and not GRADIENT_PENALTY:
            for p in dis.parameters():
                p.data.clamp_(-CLIP, CLIP)

        total_time += time.time() - _t

        if (n_iteration_t+1)%2 == 0:

            avg_loss_D += dis_loss.item()*len(x_true)
            avg_loss_G += gen_loss.item()*len(x_true)
            avg_penalty += penalty.item()*len(x_true)
            num_samples += len(x_true)

        n_iteration_t += 1

    avg_loss_G /= num_samples
    avg_loss_D /= num_samples
    avg_penalty /= num_samples

    print('Iter: %i, Loss Generator: %.4f, Loss Discriminator: %.4f, Penalty: %.2e, IS: %.2f, Time: %.4f'%(n_gen_update, avg_loss_G, avg_loss_D, avg_penalty, gen_inception_score, time.time() - t))

    f_writter.writerow((n_gen_update, avg_loss_G, avg_loss_D, avg_penalty, time.time() - t))
    f.flush()

    x_gen = gen(z_examples)
    x = utils.unormalize(x_gen)
    torchvision.utils.save_image(x.data, os.path.join(OUTPUT_PATH, 'gen/%i.png' % n_gen_update), 10)
