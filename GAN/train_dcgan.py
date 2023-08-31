"""
Train a Deep Convolutional Generative Adversarial Networks.
To run an example configuration, use this command from the base directory:
$python3 GAN/train_dcgan.py --dataroot <path_to_data> --niter 25 --cuda --loggerName training.log
Copyright (c) 2023 Global Health Labs, Inc
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

from __future__ import print_function
import argparse
import os
import sys
import random
from xmlrpc.client import boolean
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import pickle

import metric
from metric import make_dataset
import numpy as np
from GANmodels import Generator,Discriminator
from logger import Logger

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='consolidation', help='current only support the default type')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='dimension of the latent z vector')
    parser.add_argument('--nc', type=int, default=1, help='number of channels')
    parser.add_argument('--ngf', type=int, default=64, help='relates to the depth of feature maps carried through the generator')
    parser.add_argument('--ndf', type=int, default=64, help='sets the depth of feature maps propagated through the discriminator')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='learning rate for generator, default=1e-4')
    parser.add_argument('--lr_d', type=float, default=5e-5, help='learning rate for discriminator, default=5e-5')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=4, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--loggerName', type=str, help='logger name')
    parser.add_argument('--noisyDiscriminator', type=boolean, default=False)
    parser.add_argument('--dropout', type=float, default=0.25, help='dropout')
    

    ########################################################
    #### For evaluation ####
    parser.add_argument('--sampleSize', type=int, default=2000, help='number of samples for evaluation')
    ########################################################

    opt = parser.parse_args()
    print(opt)
    
    # got timestamp
    from datetime import date
    today = date.today()
    timestamp = today.strftime("%Y-%m-%d-")

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    print("assigned GPU:",torch.cuda.current_device())

    # set up output files
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    
    outfile = timestamp+"imageSize:{0}-batchSize:{1}-nz:{2}-lr_g:{3}-lr_d:{4}-Seed:{5}-Iter:{6}".format(
        opt.imageSize,opt.batchSize,opt.nz,opt.lr_g,
        opt.lr_d,opt.manualSeed,opt.niter)
    
    output_dir = os.path.join(opt.outf,outfile)
    
    try:
        os.makedirs(output_dir)
    except OSError:
        pass
    
    # set up logger file
    logger_path = os.path.join(output_dir,opt.loggerName)
    logg = Logger(logger_path,True)

    logg.append(opt)
    
    
    #########################
    #### Dataset prepare ####
    #########################
    output_shape = (opt.imageSize,opt.imageSize,opt.nc)
    dataset = make_dataset(dataset=opt.dataset, dataroot=opt.dataroot, output_shape=output_shape)
    assert dataset

    print("The size of sample set",len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=opt.workers)

    #########################
    #### Models building ####
    #########################
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)

    # load model checkpoints if they are given
    netG = Generator(ngpu,nz,ngf,nc).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    logg.append(netG)

    netD = Discriminator(ngpu,ndf,nc,opt.noisyDiscriminator,opt.dropout).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    logg.append(netD)
    
    # Loss function
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))


    # [emd-mmd-knn(knn,real,fake,precision,recall)]*4 - IS - mode_score - FID
    score_tr = np.zeros((opt.niter, 4*7+3))
    avgs = np.zeros((opt.niter,4))

    # compute initial score
    s,avg = metric.compute_score_raw(opt.dataset, output_shape, opt.dataroot, opt.sampleSize, opt.batchSize, output_dir+'/real/', output_dir+'/fake/',
                                 netG, opt.nz, device=device,conv_model='resnet34', workers=int(opt.workers))
    score_tr[0] = s
    avgs[0]=avg
    np.save('%s/score_tr.npy' % (output_dir), score_tr)

    #########################
    #### Models training ####
    #########################
    G_losses = []
    D_losses = []
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            #TODO: check vutils.save_imag
            if i % 50 == 0:
                logg.append('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f\n'
                      % (epoch, opt.niter, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.jpg' % output_dir,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d..jpg' % (output_dir, epoch),
                        normalize=True)

        # do checkpointing
        if epoch%5==4:
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (output_dir, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (output_dir, epoch))

        ################################################
        #### metric scores computing (key function) ####
        ################################################
        s,avg = metric.compute_score_raw(opt.dataset, output_shape, opt.dataroot, opt.sampleSize, opt.batchSize, output_dir+'/real/', output_dir+'/fake/',\
                                     netG, opt.nz, device=device,conv_model='resnet34', workers=int(opt.workers))

        logg.append('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f\n'
                      % (epoch, opt.niter, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        score_tr[epoch] = s
        avgs[epoch]=avg

        # save losses for generator and discriminator
        np.save('%s/generator_loss.npy' % output_dir, np.array(G_losses))
        np.save('%s/discriminator_loss.npy' % output_dir, np.array(D_losses))
        
        # save final metric scores of all epoches
        np.save('%s/score_tr_ep.npy' % output_dir, score_tr)
    
    print('##### training completed :) #####')
    print('### metric scores are stored at %s/score_tr_ep.npy ###' % output_dir)
