r""" Hypercorrelation Squeeze testing code """
import argparse

import torch.nn.functional as F
import torch.nn as nn
import torch

from model.mymodel import fewshotnet
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np



def compute_miou(Es_mask, qmask):
    Es_mask, qmask = Es_mask.detach().cpu().numpy(), qmask.detach().cpu().numpy()
    ious = 0.0
    Es_mask = np.where(Es_mask> 0.5, 1. , 0.)
    for idx in range(Es_mask.shape[0]):
        notTrue = 1 -  qmask[idx]
        union = np.sum(qmask[idx] + (notTrue * Es_mask[idx]))
        intersection = np.sum(qmask[idx] * Es_mask[idx])
        ious += (intersection / union)
    miou = (ious / Es_mask.shape[0])
    return miou



def test(model, dataloader, nshot):
    r""" Test HSNet """
    miou = 0.0
    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)
    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        miou += compute_miou(pred_mask, batch['query_mask'])
        # name = batch['query_name'][0][:-4]+'_pred.png'
        
        # mask = pred_mask.detach().cpu().numpy()[0]

        # from PIL import Image
        # from matplotlib import cm
        # from matplotlib import pyplot as plt 

        # plt.imsave(name, mask, cmap=cm.gray)

    #     assert pred_mask.size() == batch['query_mask'].size()

    #     # 2. Evaluate prediction
    #     area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
    #     average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
    #     average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)
        

    #     # Visualize predictions
    #     if Visualizer.visualize:
    #         Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
    #                                               batch['query_img'], batch['query_mask'],
    #                                               pred_mask, batch['class_id'], idx,
    #                                               area_inter[1].float() / area_union[1].float())
        
    # # Write evaluation results
    # average_meter.write_result('Test', 0)
    # miou, fb_iou = average_meter.compute_iou()

    return miou/(idx)*100.


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Annotation free few-shot segmentation Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='D:/dataset/fewshot_data/')
    parser.add_argument('--benchmark', type=str, default='fss')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=24)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='./logs/fss_weightsnew.pt') 
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--use_original_imgsize', action='store_true')
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = fewshotnet()
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))
    print('model created and weight file is loaded')

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)


    # Test HSNet
    with torch.no_grad():
        test_mio = test(model, dataloader_test, args.nshot)
        print(f'Test MIO is:{test_mio}')
