import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from apmeter import APMeter


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    decay.add('module.summarization.attn.in_proj_weight')
    decay.add('module.attention_gating.in_proj_weight')
    no_decay.add('module.summarization.tokens')

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
        optimizer,
        optimizer_config,
        num_iters_per_epoch,
        last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # get eta min
        eta_min = optimizer_config["eta_min"]

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get eta min
        eta_min = optimizer_config["eta_min"]

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(
        dataloader,
        model,
        optimizer,
        scheduler,
        curr_epoch,
        model_ema=None,
        clip_grad_l2norm=-1,
        print_freq=20
):
    """Training the model for one epoch"""
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        num_iter += 1

        outputs, loss, probs, err = model(data)

        # [: data[0]['labels'].cpu().numpy()[0].shape[1],:]
        apm.add(probs.detach().cpu().numpy()[0], data[0]['labels'].cpu().numpy()[0])
        error += err.data
        tot_loss += loss.data

        loss.backward()
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        train_map = 100 * apm.value().mean()
        train_map_micro = 100 * apm.value_micro()
    print('train-map macro:', train_map)
    print('train-map micro:', train_map_micro)
    apm.reset()

    epoch_loss = tot_loss / num_iter
    print('Epoch_loss_train = ', epoch_loss, ',  ', 'tot_loss = ', tot_loss, ',  ', 'num_iter = ', num_iter)

    last_lr = optimizer.param_groups[0]['lr']




def valid_one_epoch(
        val_loader,
        model,
        curr_epoch,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    # results = {
    #     'video-id': [],
    #     't-start': [],
    #     't-end': [],
    #     'label': [],
    #     'score': []
    # }
    apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    num_preds = 0
    full_probs = {}

    # loop over validation set
    start = time.time()
    # for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
    for data in val_loader:
        num_iter += 1

        other = [data[0]['video_id'], data[0]['duration']]
        outputs, loss, probs, err = model(data)
        apm.add(probs.detach().cpu().numpy()[0], data[0]['labels'].cpu().numpy()[0])


        error += err.data
        tot_loss += loss.data

        probs = probs.squeeze()

        full_probs[other[0][0]] = probs.data.cpu().numpy().T

    epoch_loss = tot_loss / num_iter
    print('Epoch_loss_val = ', epoch_loss, ',  ', 'tot_loss = ', tot_loss, ',  ', 'num_iter = ', num_iter)


    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    val_map_micro = torch.sum(100 * apm.value_micro()) / torch.nonzero(100 * apm.value_micro()).size()[0]
    print('val-map macro:', val_map)
    print('val-map micro:', val_map_micro)
    # print(apm.value())
    apm.reset()




    #         # upack the results into ANet format
    #         num_vids = len(output)
    #         for vid_idx in range(num_vids):
    #             if output[vid_idx]['segments'].shape[0] > 0:
    #                 results['video-id'].extend(
    #                     [output[vid_idx]['video_id']] *
    #                     output[vid_idx]['segments'].shape[0]
    #                 )
    #                 results['t-start'].append(output[vid_idx]['segments'][:, 0])
    #                 results['t-end'].append(output[vid_idx]['segments'][:, 1])
    #                 results['label'].append(output[vid_idx]['labels'])
    #                 results['score'].append(output[vid_idx]['scores'])
    #
    #     # printing
    #     if (iter_idx != 0) and iter_idx % (print_freq) == 0:
    #         # measure elapsed time (sync all kernels)
    #         torch.cuda.synchronize()
    #         batch_time.update((time.time() - start) / print_freq)
    #         start = time.time()
    #
    #         # print timing
    #         print('Test: [{0:05d}/{1:05d}]\t'
    #               'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
    #             iter_idx, len(val_loader), batch_time=batch_time))
    #
    # # gather all stats and evaluate
    # results['t-start'] = torch.cat(results['t-start']).numpy()
    # results['t-end'] = torch.cat(results['t-end']).numpy()
    # results['label'] = torch.cat(results['label']).numpy()
    # results['score'] = torch.cat(results['score']).numpy()
    #
    # # print('start: ', results['t-start'][0], 'end: ', results['t-end'][0], 'label: ', results['label'][0], 'score: ',
    # #       results['score'][0])
    # # print('start: ', ext_score_file['t-start'][0], 'end: ', ext_score_file['t-end'][0], 'label: ', ext_score_file['label'][0], 'score: ',
    # #       ext_score_file['score'][0])
    #
    # if evaluator is not None:
    #     if (ext_score_file is not None) and isinstance(ext_score_file, str):
    #         results = postprocess_results(results, ext_score_file)
    #
    #
    #     # call the evaluator
    #     print('video_id: ', results['video-id'][0], 'start: ', results['t-start'][0], 'end: ', results['t-end'][0], 'label: ', results['label'][0], 'score: ',
    #                  results['score'][0])
    #
    #     _, mAP = evaluator.evaluate(results, verbose=True)
    #     print(mAP)
    #     exit()
    # else:
    #     print('im here')
    #     # dump to a pickle file that can be directly used for evaluation
    #     with open(output_file, "wb") as f:
    #         pickle.dump(results, f)
    #     mAP = 0.0
    #
    # # log mAP to tb_writer
    # if tb_writer is not None:
    #     tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)
    #
    # return mAP
