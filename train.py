# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import torch.utils.data as data_utl
from tqdm import tqdm
from apmeter import APMeter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


######################All for data preparation###############################################
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))
def make_data(split_file, split, root, num_classes=15):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)
    print('split!!!!', split)
    i = 0
    for vid in tqdm(data['database'].keys()):

        if data['database'][vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid + '.npy')):
            continue
        fts = np.load(os.path.join(root, vid + '.npy'))
        num_feat = fts.shape[0]
        label = np.zeros((num_feat, num_classes), np.float32)

        fps = num_feat / data['database'][vid]['duration']
        for ann in data['database'][vid]['annotations']:
            for fr in range(0, num_feat, 1):
                if fr / fps > ann['segment'][0] and fr / fps < ann['segment'][1]:
                    label[fr, ann['label_id']] = 1  # binary classification
        dataset.append((vid, label, data['database'][vid]['duration']))
        i += 1

    return dataset

def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    max_len = 0
    for b in batch:
        if b[0].shape[0] > max_len:
            max_len = b[0].shape[0]

    new_batch = []
    for b in batch:
        f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
        m = np.zeros((max_len), np.float32)
        l = np.zeros((max_len, b[1].shape[1]), np.float32)
        f[:b[0].shape[0]] = b[0]
        m[:b[0].shape[0]] = 1
        l[:b[0].shape[0], :] = b[1]
        new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[2]])

    return default_collate(new_batch)

class MultiThumos(data_utl.Dataset):

    def __init__(self, split_file, split, root, batch_size, classes):

        self.data = make_data(split_file, split, root, classes)
        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root
        self.in_mem = {}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        entry = self.data[index]
        if entry[0] in self.in_mem:
            feat = self.in_mem[entry[0]]
        else:
            # print('here')
            feat = np.load(os.path.join(self.root, entry[0] + '.npy'))
            # print(feat.shape[-1])
            feat = feat.reshape((feat.shape[0], 1, 1, feat.shape[-1]))
            feat = feat.astype(np.float32)

        label = entry[1]

        return feat, label, [entry[0], entry[2]]

    def __len__(self):
        return len(self.data)


def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    max_len = 0
    for b in batch:
        if b[0].shape[0] > max_len:
            max_len = b[0].shape[0]

    new_batch = []
    for b in batch:
        f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
        m = np.zeros((max_len), np.float32)
        l = np.zeros((max_len, b[1].shape[1]), np.float32)
        f[:b[0].shape[0]] = b[0]
        m[:b[0].shape[0]] = 1
        l[:b[0].shape[0], :] = b[1]
        new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[2]])

    return default_collate(new_batch)

########################################################################################3


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(
                                        cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )
    ######################################dataset and dataloader#######################################
    train_data = MultiThumos('/data/stars/user/areka/files_features_swin/mpiigi_15_last_try.json', 'Validation',
                           '/data/stars/user/areka/files_features_swin/mpiigi', 1, 15)
    train_datalod = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=2,
                                              pin_memory=True, collate_fn=mt_collate_fn)
    ###################################################################################################
    new_dataload = []
    for train in train_loader:
        video_id = train[0]['video_id']
        for train1 in train_datalod:
            inputs, mask, labels, other = train1
            if video_id == other[0][0]:
                train[0]['labels'] = labels.to(device)
                break
        new_dataload.append(train)
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            new_dataload,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq
        )



        # save ckpt once in a while
        if (
                (epoch == max_epochs - 1) or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0) and
                        (epoch > 0)
                )
        ):
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )



    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
