import os.path as osp
import time
import re

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from muhdi.utils.func import per_class_iu, fast_hist
from muhdi.utils.serialization import pickle_dump, pickle_load


def evaluate_domain_adaptation(models, test_loader_list, cfg,
                               verbose=True):
    device = cfg.GPU_ID
    interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
                         mode='bilinear', align_corners=True)
    # eval
    if cfg.TEST.MODE == 'single':
        eval_single(cfg, models,
                    device, test_loader_list, interp,
                    verbose)
    elif cfg.TEST.MODE == 'best':
        eval_best(cfg, models,
                  device, test_loader_list, interp,
                  verbose)
    else:
        raise NotImplementedError(f"Not yet supported test mode {cfg.TEST.MODE}")


def eval_single(cfg, models,
                device, test_loader_list, interp,
                verbose):
    ## TODO: update for multi target
    assert len(cfg.TEST.RESTORE_FROM) == len(models), 'Number of models are not matched'
    for checkpoint, model in zip(cfg.TEST.RESTORE_FROM, models):
        load_checkpoint_for_evaluation(model, checkpoint, device)
    # eval
    print("Evaluating model ", cfg.TEST.RESTORE_FROM[0])
    num_targets = len(cfg.TARGETS)
    computed_miou_list = []
    for i_target in range(num_targets):
        if cfg.TARGETS[i_target] == "ACDC":
            set = re.split('_',cfg.TEST.SET_TARGET[i_target])
            cfg.TARGETS[i_target] = "ACDC_" + set[0]
        test_loader = test_loader_list[i_target]
        hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
        test_iter = iter(test_loader)
        for index in tqdm(range(len(test_loader))):
            image, label, _, name = next(test_iter)
            with torch.no_grad():
                output = None
                for model, model_weight in zip(models, cfg.TEST.MODEL_WEIGHT):
                    if cfg.TEST.MODEL[0] == 'DeepLabv2':
                        _, pred_main = model(image.cuda(device))
                    else:
                        raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[0]}")

                    interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
                                             mode='bilinear', align_corners=True)
                    output_ = interp(pred_main).cpu().data[0].numpy()
                    if output is None:
                        output = model_weight * output_
                    else:
                        output += model_weight * output_
                assert output is not None, 'Output is None'
                output = output.transpose(1, 2, 0)
                output = np.argmax(output, axis=2)
            label = label.numpy()[0]
            hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
            inters_over_union_classes = per_class_iu(hist)
        computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
        computed_miou_list.append(computed_miou)
        print('\tTarget:', cfg.TARGETS[i_target])
        print(f'mIoU = \t{computed_miou}')
        if verbose:
            name_classes = np.array(test_loader.dataset.info['label'], dtype=np.str)
            display_stats(cfg, name_classes, inters_over_union_classes)
    print('\tMulti-target:', cfg.TARGETS)
    print('\mIoU:', round(np.nanmean(computed_miou_list), 2))



def eval_best(cfg, models,
              device, test_loader_list, interp,
              verbose):
    assert len(models) == 1, 'Not yet supported multi models in this mode'
    assert osp.exists(cfg.TEST.SNAPSHOT_DIR[0]), 'SNAPSHOT_DIR is not found'
    start_iter = cfg.TEST.SNAPSHOT_STEP
    step = cfg.TEST.SNAPSHOT_STEP
    max_iter = cfg.TEST.SNAPSHOT_MAXITER
    all_res_list = []
    cache_path_list = []
    num_targets = len(cfg.TARGETS)
    for target in cfg.TARGETS:
        cache_path = osp.join(osp.join(cfg.TEST.SNAPSHOT_DIR[0], target), cfg.TEST.ALL_RES)
        cache_path_list.append(cache_path)
        if osp.exists(cache_path):
            all_res_list.append(pickle_load(cache_path))
        else:
            all_res_list.append({})
    cur_best_miou = -1
    cur_best_model = ''
    cur_best_miou_list = []
    cur_best_model_list = []
    for i in range(num_targets):
        cur_best_miou_list.append(-1)
        cur_best_model_list.append('')
    for i_iter in range(start_iter, max_iter+1, step): #
        print(f'Loading model_{i_iter}.pth')
        restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{i_iter}.pth')
        if not osp.exists(restore_from):
            # continue
            if cfg.TEST.WAIT_MODEL:
                print('Waiting for model..!')
                while not osp.exists(restore_from):
                    time.sleep(5)
        print("Evaluating model", restore_from)
        load_checkpoint_for_evaluation(models[0], restore_from, device)
        computed_miou_list = []
        for i_target in range(num_targets):
            if cfg.TARGETS[i_target] == "ACDC":
                set = re.split('_',cfg.TEST.SET_TARGET[i_target])
                cfg.TARGETS[i_target] = "ACDC_" + set[0]
            print("On target", cfg.TARGETS[i_target])
            all_res = all_res_list[i_target]
            cache_path = cache_path_list[i_target]
            test_loader = test_loader_list[i_target]
            if i_iter not in all_res.keys():
                # eval
                hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
                test_iter = iter(test_loader)
                for index in tqdm(range(len(test_loader))):
                    image, label, _, name = next(test_iter)
                    with torch.no_grad():
                        output = None
                        if cfg.TEST.MODEL[0] == 'DeepLabv2':
                            _, pred_main = models[0](image.cuda(device))
                        else:
                            raise NotImplementedError(f"Not yet supported {cfg.TEST.MODEL[0]}")
                        interp = nn.Upsample(size=(cfg.TEST.OUTPUT_SIZE_TARGET[1], cfg.TEST.OUTPUT_SIZE_TARGET[0]),
                                                 mode='bilinear', align_corners=True)
                        output = interp(pred_main).cpu().data[0].numpy()
                        output = output.transpose(1, 2, 0)
                        output = np.argmax(output, axis=2)
                    label = label.numpy()[0]
                    hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
                    if verbose and index > 0 and index % 500 == 0:
                        print('{:d} / {:d}: {:0.2f}'.format(
                            index, len(test_loader), 100 * np.nanmean(per_class_iu(hist))))
                inters_over_union_classes = per_class_iu(hist)
                all_res[i_iter] = inters_over_union_classes
                pickle_dump(all_res, cache_path)
            else:
                inters_over_union_classes = all_res[i_iter]
            computed_miou = round(np.nanmean(inters_over_union_classes) * 100, 2)
            computed_miou_list.append(computed_miou)
            if cur_best_miou_list[i_target] < computed_miou:
                cur_best_miou_list[i_target] = computed_miou
                cur_best_model_list[i_target] = restore_from
            print('\tTarget:', cfg.TARGETS[i_target])
            print('\tCurrent mIoU:', computed_miou)
            print('\tCurrent best model:', cur_best_model_list[i_target])
            print('\tCurrent best mIoU:', cur_best_miou_list[i_target])
            if verbose:
                name_classes = np.array(test_loader.dataset.info['label'], dtype=np.str)
                display_stats(cfg, name_classes, inters_over_union_classes)
        computed_miou = round(np.nanmean(computed_miou_list), 2)
        if cur_best_miou < computed_miou:
            cur_best_miou = computed_miou
            cur_best_model = restore_from
        print('\tMulti-target:', cfg.TARGETS)
        print('\tCurrent mIoU:', computed_miou)
        print('\tCurrent best model:', cur_best_model)
        print('\tCurrent best mIoU:', cur_best_miou)


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict, strict=False)
    model.eval()
    model.cuda(device)


def display_stats(cfg, name_classes, inters_over_union_classes):
    for ind_class in range(cfg.NUM_CLASSES):
        print(name_classes[ind_class]
              + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))
