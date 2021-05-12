import os
import yaml
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.loss import GeneratorLoss, DiscriminatorLoss
from torch.optim.lr_scheduler import MultiStepLR
from dataset import ProteinDataset, collate_fn
from models.GANcon import ContactMapGenerator, ContactMapDiscriminator

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'default.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
MULTIGPU = cfg_dict.get('multigpu', True)
MAX_EPOCH = cfg_dict.get('max_epoch', 30)
CHECKPOINT_DIR = cfg_dict.get('checkpoint_dir', 'checkpoint')
GENERATOR = cfg_dict.get('generator', {})
DISCRIMINATOR = cfg_dict.get('discriminator', {})
TRAINING_CFG = cfg_dict.get('training', {})
LOSS = cfg_dict.get('loss', {})
GENERATOR_BATCH_SIZE = GENERATOR.get('batch_size', 1)
CLASS_NUMBER = GENERATOR.get('output_channel', 10)

# Load data & Build dataset
TRAIN_DIR = os.path.join('data', 'train')
TRAIN_FEATURE_DIR = os.path.join(TRAIN_DIR, 'feature')
TRAIN_LABEL_DIR = os.path.join(TRAIN_DIR, 'label')
train_dataset = ProteinDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR)
train_dataloader = DataLoader(train_dataset, batch_size = GENERATOR_BATCH_SIZE, shuffle = True, collate_fn = collate_fn)

VAL_DIR = os.path.join('data', 'val')
VAL_FEATURE_DIR = os.path.join(VAL_DIR, 'feature')
VAL_LABEL_DIR = os.path.join(VAL_DIR, 'label')
val_dataset = ProteinDataset(VAL_FEATURE_DIR, VAL_LABEL_DIR)
val_dataloader = DataLoader(val_dataset, batch_size = GENERATOR_BATCH_SIZE, shuffle = True, collate_fn = collate_fn)

# Build model from configs
generator = ContactMapGenerator(GENERATOR)
discriminator = ContactMapDiscriminator(DISCRIMINATOR)

# Define optimizer
generator_beta = (GENERATOR.get('adam_beta1', 0.9), GENERATOR.get('adam_beta2', 0.999))
generator_learning_rate = GENERATOR.get('learning_rate', 0.01)
discriminator_beta = (DISCRIMINATOR.get('adam_beta1', 0.9), DISCRIMINATOR.get('adam_beta2', 0.999))
discriminator_learning_rate = DISCRIMINATOR.get('learning_rate', 0.01)
generator_optimizer = optim.Adam(generator.parameters(), 
                                 betas = generator_beta, lr = generator_learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters(), 
                                     betas = discriminator_beta, lr = discriminator_learning_rate)

# Define Criterion
LOSS_ALPHA = LOSS.get('alpha', [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
LOSS_BETA = LOSS.get('beta', 1.0)
LOSS_GAMMA = LOSS.get('gamma', 2.0)
LOSS_LAMBDA = LOSS.get('lambda', 1.0)
LOSS_EPS = LOSS.get('eps', 1e-6)
generator_criterion = GeneratorLoss(alpha_ = LOSS_ALPHA, beta_ = LOSS_BETA, gamma_ = LOSS_GAMMA, lambda_ = LOSS_LAMBDA, eps_ = LOSS_EPS)
discriminator_criterion = DiscriminatorLoss()

# Define Scheduler
generator_lr_scheduler = MultiStepLR(generator_optimizer, 
                                     milestones = GENERATOR.get('milestones', []), gamma = GENERATOR.get('gamma', 0.1))
discriminator_lr_scheduler = MultiStepLR(discriminator_optimizer,
                                         milestones = DISCRIMINATOR.get('milestones', []), gamma = DISCRIMINATOR.get('gamma', 0.1))

# Load Checkpoints
start_epoch = 0
if os.path.exists(CHECKPOINT_DIR) == False:
    os.mkdir(CHECKPOINT_DIR)
generator_checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_generator.tar')
discriminator_checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_discriminator.tar')
if os.path.isfile(generator_checkpoint_file) and os.path.isfile(discriminator_checkpoint_file):
    generator_checkpoint = torch.load(generator_checkpoint_file)
    generator.load_state_dict(generator_checkpoint['model_state_dict'])
    generator_optimizer.load_state_dict(generator_checkpoint['optimizer_state_dict'])
    G_epoch = generator_checkpoint['epoch']
    generator_lr_scheduler.load_state_dict(generator_checkpoint['scheduler'])
    print("Load checkpoint {} (epoch {})".format(generator_checkpoint_file, G_epoch))
    discriminator_checkpoint = torch.load(discriminator_checkpoint_file)
    discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
    discriminator_optimizer.load_state_dict(discriminator_checkpoint['optimizer_state_dict'])
    D_epoch = discriminator_checkpoint['epoch']
    discriminator_lr_scheduler.load_state_dict(discriminator_checkpoint['scheduler'])
    print("Load checkpoint {} (epoch {})".format(discriminator_checkpoint_file, D_epoch))
    assert G_epoch == D_epoch
    start_epoch = G_epoch

# Data Parallelism
if MULTIGPU is False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    generator.to(device)
    discriminator.to(device)

if MULTIGPU is True:
    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)

def generator_train_one_epoch():
    generator.train()
    discriminator.eval()
    for idx, data in enumerate(train_dataloader):
        generator_optimizer.zero_grad()
        feature, label, mask = data
        feature = feature.to(device)
        label = label.to(device)
        mask = mask.to(device)
        result = generator(feature)
        with torch.no_grad():
            prediction = discriminator(feature, result).detach()
        loss = generator_criterion(prediction, result, label, mask)
        loss.backward()
        generator_optimizer.step()
        print('--------------- Generator Train Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())


def generator_eval_one_epoch():
    generator.eval()
    discriminator.eval()
    mean_loss = 0
    count = 0
    for idx, data in enumerate(val_dataloader):
        feature, label, mask = data
        feature = feature.to(device)
        label = label.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            result = generator(feature)
        with torch.no_grad():
            prediction = discriminator(feature, result).detach()
            loss = generator_criterion(prediction, result, label, mask)
        print('--------------- Generator Eval Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())
        mean_loss += loss.item()
        count += 1
    mean_loss = mean_loss / count
    return mean_loss


def discriminator_train_one_epoch():
    discriminator.train()
    generator.eval()
    for idx, data in enumerate(train_dataloader):
        discriminator_optimizer.zero_grad()
        feature, label, mask = data
        feature = feature.to(device)
        label = label.to(device)
        mask = mask.to(device)
        real_label = F.one_hot(label, num_classes = CLASS_NUMBER).permute(0, 3, 1, 2).type(torch.float)
        with torch.no_grad():
            fake_label = generator(feature).detach()
        real_result = discriminator(feature, real_label)
        fake_result = discriminator(feature, fake_label)
        real_loss = discriminator_criterion(real_result, torch.ones(real_result.shape).to(device), mask)
        fake_loss = discriminator_criterion(fake_result, torch.zeros(fake_result.shape).to(device), mask)
        loss = (real_loss + fake_loss) / 2.0
        loss.backward()
        discriminator_optimizer.step()
        print('--------------- Discriminator Train Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())


def discriminator_eval_one_epoch():
    discriminator.eval()
    generator.eval()    
    mean_loss = 0
    count = 0
    for idx, data in enumerate(val_dataloader):
        feature, label, mask = data
        feature = feature.to(device)
        label = label.to(device)
        mask = mask.to(device)
        real_label = F.one_hot(label, num_classes = CLASS_NUMBER).permute(0, 3, 1, 2).type(torch.float)
        with torch.no_grad():
            fake_label = generator(feature).detach()
            real_result = discriminator(feature, real_label)
            fake_result = discriminator(feature, fake_label)
        with torch.no_grad():
            real_loss = discriminator_criterion(real_result, torch.ones(real_result.shape).to(device), mask)
            fake_loss = discriminator_criterion(fake_result, torch.zeros(fake_result.shape).to(device), mask)
            loss = (real_loss + fake_loss) / 2.0
        print('--------------- Discriminator Eval Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())
        mean_loss += loss.item()
        count += 1
    mean_loss = mean_loss / count
    return mean_loss 


def train(start_epoch, args = {}):
    generator_training_time_per_epoch = args.get('G_training_times', 1)
    discriminator_training_time_per_epoch = args.get('D_training_times', 3)
    discriminator_warmup_time = args.get('D_warmup_times', 3)

    discriminator_epoch = 0
    generator_epoch = 0

    if start_epoch == 0:   
        for _ in range(discriminator_warmup_time):
            print('**************** Discriminator Epoch %d ****************' % (discriminator_epoch + 1))
            print('learning rate: %f' % (discriminator_lr_scheduler.get_last_lr()[0]))
            discriminator_train_one_epoch()
            loss = discriminator_eval_one_epoch()
            discriminator_lr_scheduler.step()
            discriminator_epoch += 1
            print('Discriminator mean eval loss: %.12f' % loss)

    for epoch in range(start_epoch, MAX_EPOCH):
        for _ in range(generator_training_time_per_epoch):
            print('**************** Generator Epoch %d ****************' % (generator_epoch + 1))
            print('learning rate: %f' % (generator_lr_scheduler.get_last_lr()[0]))
            generator_train_one_epoch()
            loss = generator_eval_one_epoch()
            generator_lr_scheduler.step()
            generator_epoch += 1
            print('Generator mean eval loss: %.12f' % loss)
        for _ in range(discriminator_training_time_per_epoch):
            print('**************** Discriminator Epoch %d ****************' % (discriminator_epoch + 1))
            print('learning rate: %f' % (discriminator_lr_scheduler.get_last_lr()[0]))
            discriminator_train_one_epoch()
            loss = discriminator_eval_one_epoch()
            discriminator_lr_scheduler.step()
            discriminator_epoch += 1
            print('Discriminator mean eval loss: %.12f' % loss)
        if MULTIGPU is False:
            generator_save_dict = {
                'epoch': epoch + 1,
                'optimizer_state_dict': generator_optimizer.state_dict(),
                'model_state_dict': generator.state_dict(),
                'scheduler': generator_lr_scheduler.state_dict()
            }
            discriminator_save_dict = {
                'epoch': epoch + 1,
                'optimizer_state_dict': discriminator_optimizer.state_dict(),
                'model_state_dict': discriminator.state_dict(),
                'scheduler': discriminator_lr_scheduler.state_dict()
            }
        else:
            generator_save_dict = {
                'epoch': epoch + 1,
                'optimizer_state_dict': generator_optimizer.state_dict(),
                'model_state_dict': generator.module.state_dict(),
                'scheduler': generator_lr_scheduler.state_dict()
            }
            discriminator_save_dict = {
                'epoch': epoch + 1,
                'optimizer_state_dict': discriminator_optimizer.state_dict(),
                'model_state_dict': discriminator.module.state_dict(),
                'scheduler': discriminator_lr_scheduler.state_dict()
            }
        torch.save(generator_save_dict, os.path.join(CHECKPOINT_DIR, 'checkpoint_generator.tar'))
        torch.save(discriminator_save_dict, os.path.join(CHECKPOINT_DIR, 'checkpoint_discriminator.tar'))


if __name__ == '__main__':
    train(start_epoch, TRAINING_CFG)