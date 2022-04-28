import os
import math
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from muhdi.model.discriminator import get_fc_discriminator, restore_discriminator
from muhdi.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from muhdi.utils.func import loss_calc, bce_loss
from muhdi.utils.loss import kl_divergence, mse_loss
from muhdi.utils.func import prob_2_entropy

def train_advent(model, source_loader, target_loader, cfg):
    ''' UDA training with advent
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True


    # DISCRIMINATOR NETWORK
    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    if cfg.TRAIN.RESTORE_D_MAIN is not '':
        d_main = restore_discriminator(d_main, cfg.TRAIN.RESTORE_D_MAIN)
    d_main.train()
    d_main.to(device)
    d_aux = get_fc_discriminator(num_classes=num_classes)
    if cfg.TRAIN.RESTORE_D_AUX is not '':
        d_aux = restore_discriminator(d_aux, cfg.TRAIN.RESTORE_D_AUX)
    d_aux.train()
    d_aux.to(device)
    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                          betas=(0.9, 0.99))
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                          betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP+1)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = source_loader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux, pred_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # adversarial training to fool the discriminators
        # train on target if pseudo-labels
        _, batch = target_loader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux, pred_trg_main = model(images.float().cuda(device))

        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux)
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux': loss_seg_src_aux,
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP:
                break
        sys.stdout.flush()

def train_advent_muhdi(model, old_model, source_loader, target_loader, cfg):
    ''' UDA training with advent and attention
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES

    if cfg.TRAIN.TEACHER_LOSS == "MSE":
        teacher_loss = mse_loss
    elif cfg.TRAIN.TEACHER_LOSS == "KL":
        teacher_loss = kl_divergence
    else:
        raise NotImplementedError(f"Not yet supported loss {cfg.TRAIN.TEACHER_LOSS}")

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    old_model.eval()
    old_model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # DISCRIMINATOR NETWORK
    # seg maps, i.e. output, level
    d_main = get_fc_discriminator(num_classes=num_classes)
    if cfg.TRAIN.RESTORE_D_MAIN is not '':
        d_main = restore_discriminator(d_main, cfg.TRAIN.RESTORE_D_MAIN)
    d_main.train()
    d_main.to(device)
    d_aux = get_fc_discriminator(num_classes=num_classes)
    if cfg.TRAIN.RESTORE_D_AUX is not '':
        d_aux = restore_discriminator(d_aux, cfg.TRAIN.RESTORE_D_AUX)
    d_aux.train()
    d_aux.to(device)
    # OPTIMIZERS
    # segnet's optimizer
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # discriminators' optimizers
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                              betas=(0.9, 0.99))
    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                              betas=(0.9, 0.99))

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1
    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP+1)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        # UDA Training
        # only train segnet. Don't accumulate grads in disciminators
        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
        # train on source
        _, batch = source_loader_iter.__next__()
        images_source, labels, _, _ = batch
        images_source_cuda = images_source.cuda(device)
        pred_src_aux_agn, pred_src_main_agn, pred_src_aux_spe, pred_src_main_spe, attentions_new = model(images_source_cuda)
        pred_src_aux_old, pred_src_main_old, attentions_old = old_model(images_source_cuda)

        # source distillation
        pred_prob_src_aux_agn = F.softmax(pred_src_aux_agn)
        pred_prob_src_main_agn = F.softmax(pred_src_main_agn)

        feat_distill_loss = 0

        if cfg.TRAIN.DISTILL.FEAT:
            feat_distill_loss += features_distillation(attentions_old, attentions_new)

        kt_distill_loss = 0
        if cfg.TRAIN.DISTILL.KT_LOGITS:
            kt_distill_loss += kl_divergence(pred_src_main_agn, pred_src_main_old)
            if cfg.TRAIN.MULTI_LEVEL:
                kt_distill_loss += cfg.TRAIN.LAMBDA_SEG_AUX/cfg.TRAIN.LAMBDA_SEG_MAIN * kl_divergence(pred_src_aux_agn, pred_src_aux_old)


        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux_spe = interp(pred_src_aux_spe)
            loss_seg_src_aux_spe = loss_calc(pred_src_aux_spe, labels, device)
        else:
            loss_seg_src_aux_spe = 0
        pred_src_main_spe = interp(pred_src_main_spe)
        loss_seg_src_main_spe = loss_calc(pred_src_main_spe, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main_spe
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux_spe
                + cfg.TRAIN.DISTILL.FEAT_LAMBDA * feat_distill_loss
                + cfg.TRAIN.DISTILL.KT_LAMBDA * kt_distill_loss)
        loss.backward()

        # adversarial training to fool the discriminators
        _, batch = target_loader_iter.__next__()
        images, _, _, _ = batch
        pred_trg_aux_agn, pred_trg_main_agn, pred_trg_aux_spe, pred_trg_main_spe, _ = model(images.float().cuda(device))

        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_spe = interp_target(pred_trg_aux_spe)
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux_spe)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main_spe = interp_target(pred_trg_main_spe)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main_spe)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)

        loss_div_trg = 0
        loss_div_aux = 0
        pred_trg_main_spe = pred_trg_main_spe.detach()
        pred_trg_main_agn = interp_target(pred_trg_main_agn)
        loss_div_trg = teacher_loss(pred_trg_main_agn, pred_trg_main_spe)
        loss_kt_trg = loss_div_trg
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_agn = interp_target(pred_trg_aux_agn)
            pred_trg_aux_spe = pred_trg_aux_spe.detach()
            loss_div_aux = teacher_loss(pred_trg_aux_agn, pred_trg_aux_spe)
        loss_kt_trg += cfg.TRAIN.LAMBDA_SEG_AUX * loss_div_aux

        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux
                + cfg.TRAIN.LAMBDA_KT_TARGET * loss_kt_trg)
        loss.backward()

        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux_spe = pred_src_aux_spe.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux_spe)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main_spe = pred_src_main_spe.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main_spe)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_spe = pred_trg_aux_spe.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux_spe)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main_spe = pred_trg_main_spe.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main_spe)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        optimizer.step()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {'loss_seg_src_aux_spe': loss_seg_src_aux_spe,
                          'loss_seg_src_main_spe': loss_seg_src_main_spe,
                          'loss_adv_trg_aux': loss_adv_trg_aux,
                          'loss_adv_trg_main': loss_adv_trg_main,
                          'loss_d_aux': loss_d_aux,
                          'loss_d_main': loss_d_main,
                          'feat_distill_loss': feat_distill_loss,
                          'kt_distill_loss': kt_distill_loss,
                          'loss_kt_trg': loss_kt_trg}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(d_aux.state_dict(), snapshot_dir / f'model_{i_iter}_D_aux.pth')
            torch.save(d_main.state_dict(), snapshot_dir / f'model_{i_iter}_D_main.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP:
                break
        sys.stdout.flush()

def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def features_distillation(
    list_attentions_a,
    list_attentions_b,
    mask=None,
    collapse_channels="local",
    normalize=True,
    labels=None,
    mask_threshold=0.0,
    pod_apply="all",
    pod_deeplab_mask=False,
    pod_deeplab_mask_factor=None,
    interpolate_last=False,
    pod_factor=1.,
    pod_output_factor=0.05,
    n_output_layers=0,
    prepro="pow",
    deeplabmask_upscale=True,
    spp_scales=[1, 2, 4],
    pod_options=None,
    outputs_old=None,
):
    """A mega-function comprising several features-based distillation.
    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    device = list_attentions_a[0].device

    assert len(list_attentions_a) == len(list_attentions_b)

    if pod_deeplab_mask_factor is None:
        pod_deeplab_mask_factor = pod_factor

    #if collapse_channels in ("spatial_tuple", "spp", "spp_noNorm", "spatial_noNorm"):
    normalize = False

    upscale_mask_topk = 1
    mask_position = "all"  # Others choices "all" "backbone"
    use_adaptative_factor = False
    mix_new_old = None
    pod_output_factor = pod_factor * pod_output_factor

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        adaptative_pod_factor = 1.0
        difference_function = "frobenius"
        pool = True
        use_adaptative_factor = False
        normalize_per_scale = False

        if i >= len(list_attentions_a) - n_output_layers:
            pod_factor = pod_output_factor

        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if not pod_deeplab_mask and use_adaptative_factor:
            adaptative_pod_factor = (labels == 0).float().mean()

        if prepro == "pow":
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)
        elif prepro == "none":
            pass
        elif prepro == "abs":
            a = torch.abs(a, 2)
            b = torch.abs(b, 2)
        elif prepro == "relu":
            a = torch.clamp(a, min=0.)
            b = torch.clamp(b, min=0.)

        if collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "local":
            if pod_deeplab_mask and (
                (i == len(list_attentions_a) - 1 and mask_position == "last") or
                mask_position == "all"
            ):
                if pod_deeplab_mask_factor == 0.:
                    continue

                pod_factor = pod_deeplab_mask_factor

                a = F.interpolate(
                    torch.topk(a, k=upscale_mask_topk, dim=1)[0],
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                b = F.interpolate(
                    torch.topk(b, k=upscale_mask_topk, dim=1)[0],
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                if use_adaptative_factor:
                    adaptative_pod_factor = mask.float().mean(dim=(1, 2))

                a = _local_pod(
                    a, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod(
                    b, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
            else:
                mask = None
                a = _local_pod(
                    a, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod(
                    b, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if i == len(list_attentions_a) - 1 and pod_options is not None:
            if "difference_function" in pod_options:
                difference_function = pod_options["difference_function"]
        elif pod_options is not None:
            if "difference_function_all" in pod_options:
                difference_function = pod_options["difference_function_all"]

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        if difference_function == "frobenius":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.frobenius_norm(aa - bb, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.frobenius_norm(a - b, dim=-1)
        elif difference_function == "frobenius_mix":
            layer_loss_old = torch.frobenius_norm(a[0] - b[0], dim=-1)
            layer_loss_new = torch.frobenius_norm(a[1] - b[1], dim=-1)

            layer_loss = mix_new_old * layer_loss_old + (1 - mix_new_old) * layer_loss_new
        elif difference_function == "l1":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.norm(aa - bb, p=1, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.norm(a - b, p=1, dim=-1)
        elif difference_function == "kl":
            d1, d2, d3 = a.shape
            a = (a.view(d1 * d2, d3) + 1e-8).log()
            b = b.view(d1 * d2, d3) + 1e-8

            layer_loss = F.kl_div(a, b, reduction="none").view(d1, d2, d3).sum(dim=(1, 2))
        elif difference_function == "bce":
            d1, d2, d3 = a.shape
            layer_loss = bce(a.view(d1 * d2, d3), b.view(d1 * d2, d3)).view(d1, d2,
                                                                            d3).mean(dim=(1, 2))
        else:
            raise NotImplementedError(f"Unknown difference_function={difference_function}")

        assert torch.isfinite(layer_loss).all(), layer_loss
        assert (layer_loss >= 0.).all(), layer_loss

        layer_loss = torch.mean(adaptative_pod_factor * layer_loss)
        if pod_factor <= 0.:
            continue

        layer_loss = pod_factor * layer_loss
        loss += layer_loss

    return loss / len(list_attentions_a)


def _local_pod(x, mask, spp_scales=[1, 2, 4], square_crop=False, normalize=False, normalize_per_scale=False, per_level=False, median=False, dist=False):
    b = x.shape[0]
    h, w = x.shape[-2:]
    emb = []

    if mask is not None:
        mask = mask[:, None].repeat(1, c, 1, 1)
        x[mask] = 0.

    min_side = min(h, w)

    for scale_index, scale in enumerate(spp_scales):

        nb_regions = scale**2

        if square_crop:
            scale_h = h // min_side * scale
            scale_w = w // min_side * scale
            kh = kw = min_side // scale
        else:
            kh, kw = h // scale, w // scale
            scale_h = scale_w = scale

        if per_level:
            emb_per_level = []

        for i in range(scale_h):
            for j in range(scale_w):
                tensor = x[..., i * kh:(i + 1) * kh, j * kw:(j + 1) * kw]
                if any(shp == 0 for shp in tensor.shape):
                    print(f"Empty tensor {tensor.shape}, with (i={i}, j={j}) and scale_h={scale_h}, scale_w={scale_w}")
                    continue

                if median:
                    horizontal_pool = tensor.median(dim=3, keepdims=True)[0]
                    vertical_pool = tensor.median(dim=2, keepdims=True)[0]
                else:
                    horizontal_pool = tensor.mean(dim=3, keepdims=True)
                    vertical_pool = tensor.mean(dim=2, keepdims=True)

                if dist:
                    # Compute the distance distribution compared to the mean/median
                    horizontal_pool = (tensor - horizontal_pool).mean(dim=3).view(b, -1)
                    vertical_pool = (tensor - vertical_pool).mean(dim=2).view(b, -1)
                else:
                    horizontal_pool = horizontal_pool.view(b, -1)
                    vertical_pool = vertical_pool.view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                    horizontal_pool = horizontal_pool / factor
                    vertical_pool = vertical_pool / factor

                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                if not per_level:
                    emb.append(horizontal_pool)
                    emb.append(vertical_pool)
                else:
                    emb_per_level.append(torch.cat([horizontal_pool, vertical_pool], dim=1))
        if per_level:
            emb.append(torch.stack(emb_per_level, dim=1))

    if not per_level:
        return torch.cat(emb, dim=1)
    return emb
