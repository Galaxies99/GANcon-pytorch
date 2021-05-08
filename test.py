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
from utils.criterion import calc_batch_acc
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
LOSS = cfg_dict.get('loss', {})
GENERATOR_BATCH_SIZE = GENERATOR.get('batch_size', 1)

# Load data & Build dataset
TEST_DIR = os.path.join('data', 'test')
TEST_FEATURE_DIR = os.path.join(TEST_DIR, 'feature')
TEST_LABEL_DIR = os.path.join(TEST_DIR, 'label')
test_dataset = ProteinDataset(TEST_FEATURE_DIR, TEST_LABEL_DIR)
test_dataloader = DataLoader(test_dataset, batch_size = GENERATOR_BATCH_SIZE, shuffle = True, collate_fn = collate_fn)

# Build model from configs
generator = ContactMapGenerator(GENERATOR)
discriminator = ContactMapDiscriminator(DISCRIMINATOR)

# Define Criterion
LOSS_ALPHA = LOSS.get('alpha', [0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
LOSS_BETA = LOSS.get('beta', 1.0)
LOSS_GAMMA = LOSS.get('gamma', 2.0)
LOSS_LAMBDA = LOSS.get('lambda', 1.0)
generator_criterion = GeneratorLoss(alpha_ = LOSS_ALPHA, beta_ = LOSS_BETA, gamma_ = LOSS_GAMMA, lambda_ = LOSS_LAMBDA)
discriminator_criterion = DiscriminatorLoss()

# Load Checkpoints
generator_checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_generator.tar')
discriminator_checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_discriminator.tar')
if os.path.isfile(generator_checkpoint_file) and os.path.isfile(discriminator_checkpoint_file):
    generator_checkpoint = torch.load(generator_checkpoint_file)
    generator.load_state_dict(generator_checkpoint['model_state_dict'])
    G_epoch = generator_checkpoint['epoch']
    print("Load checkpoint {} (epoch {})".format(generator_checkpoint_file, G_epoch))
    discriminator_checkpoint = torch.load(discriminator_checkpoint_file)
    discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])
    D_epoch = discriminator_checkpoint['epoch']
    print("Load checkpoint {} (epoch {})".format(discriminator_checkpoint_file, D_epoch))
else:
    raise AttributeError('No checkpoint file!')

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

def generator_test_one_epoch():
    generator.eval()
    discriminator.eval()
    mean_loss = 0
    count = 0
    acc = np.zeros((2, 4))
    for idx, data in enumerate(test_dataloader):
        feature, label, mask = data
        feature = feature.to(device)
        label = label.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            result = generator(feature)
        with torch.no_grad():
            prediction = discriminator(feature, result)
            loss = generator_criterion(prediction, result, label, mask)
        acc_batch, batch_size = calc_batch_acc(label.cpu().numpy(), mask.cpu().numpy(), result.cpu().numpy())
        print('--------------- Generator Eval Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())
        print('acc: ', acc_batch)
        acc += acc_batch * batch_size
        mean_loss += loss.item() * batch_size
        count += batch_size
    mean_loss = mean_loss / count
    acc = acc / count
    return mean_loss, acc


if __name__ == '__main__':
    loss, acc = generator_test_one_epoch()
    print('--------------- Test Result ---------------')
    print('test mean loss: %.12f' % loss)
    print('test mean acc: ', acc)