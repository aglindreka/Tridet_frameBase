# python imports
import argparse
import os
import glob
import time
from pprint import pprint
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import json
from tqdm import tqdm
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed


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

################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )

    

    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )



    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    ################################################################################################
    val_data = MultiThumos('/data/stars/user/areka/files_features_swin/mpiigi_15_last_try.json', 'Test', '/data/stars/user/areka/features_croped', 1,15)
    val_datalod = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True, collate_fn=mt_collate_fn)
    #################################################################################################


    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()

        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds=val_db_vars['tiou_thresholds']
        )
    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))

    start = time.time()
    new_dataload = []
    for val in val_loader:
        video_id = val[0]['video_id']
        for val1 in val_datalod:

            inputs, mask, labels, other = val1
            if video_id == other[0][0]:
                val[0]['labels'] = labels.to(device)
                break
        new_dataload.append(val)

    # for val in new_dataload:
    #     print(val[0]['feats'].shape, val[0]['labels'].shape)
    #     exit()

    valid_one_epoch(
        new_dataload,
        model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
