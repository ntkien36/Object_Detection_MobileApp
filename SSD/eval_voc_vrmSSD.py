from __future__ import print_function
"""
    Model evaluation on VOC for vggSSD separately
    Execute: python3 eval_voc_vrmSSD.py --trained_model weights/_your_trained_SSD_model_.pth

    Model evaluation on VOC for resnetSSD separately
    Execute: python3 eval_voc_vrmSSD.py --use_res --trained_model weights/_your_trained_SSD_model_.pth

    Model evaluation on VOC for mobileSSD v1 separately
    Execute: python3 eval_voc_vrmSSD.py --use_m1 --trained_model weights/_your_trained_SSD_model_.pth

    Model evaluation on VOC for mobileSSD v2 separately
    Execute: python3 eval_voc_vrmSSD.py --use_m2 --trained_model weights/_your_trained_SSD_model_.pth

    (Take care of different versions of .pth file, can be solved by changing state_dict)
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import *
import torch.utils.data as data

from models.SSD_vggres import build_ssd
from models.SSD_mobile import build_mssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
cv2.setNumThreads(0) # pytorch issue 1355: possible deadlock in DataLoader
# OpenCL may be enabled by default in OpenCV3;
# disable it because it because it's not thread safe and causes unwanted GPU memory allocations
cv2.ocl.setUseOpenCL(False)

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
# 200 in SSD paper, 200 for COCO, 300 for VOC
parser.add_argument('--max_per_image', default=200, type=int,
                    help='Top number of detections kept per image, further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default= VOC_ROOT,# XL_ROOT, for VOC_xlab_products dataset
                    help='Location of XL root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
# for resnet backbone
parser.add_argument("--use_res", dest="use_res", action="store_true")
parser.set_defaults(use_res=False)
# for mobilev1 backbone
parser.add_argument("--use_m1", dest="use_m1", action="store_true")
parser.set_defaults(use_m1=False)
# for mobilev2 backbone
parser.add_argument("--use_m2", dest="use_m2", action="store_true")
parser.set_defaults(use_m2=False)

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

set_type = 'test'
cfg = voc #xl, for VOC_xlab_products dataset

# test function for vggSSD
"""
    Args:
        save_folder: the eval results saving folder
        net: test-type ssd net
        dataset: validation dataset
        transform: BaseTransform - not used here
"""
def test_net(save_folder, net, cuda,
             testset, transform, max_per_image=300, thresh=0.05):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    num_images = len(testset)
    num_classes = cfg['num_classes']
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    #output_dir = get_output_dir('ssd300_120000', set_type) #directory storing output results
    #det_file = os.path.join(output_dir, 'detections.pkl') #file storing output result under output_dir
    det_file = os.path.join(save_folder, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = testset.pull_item(i) # include BaseTransform inside

        x = Variable(im.unsqueeze(0)) #insert a dimension of size one at the dim 0
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        detections = net(x=x, test=True).data # get the detection results
        detect_time = _t['im_detect'].toc(average=False) #store the detection time

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)): # for every class
            dets = detections[0, j, :]#size( ** , 5)
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            #if dets.size(0) == 0:
            #    continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets #[class][imageID] = 1 x 5 where 5 is box_coord + score
        if (i + 1) % 100 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                        num_images, detect_time))

    #write the detection results into det_file
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    APs,mAP = testset.evaluate_detections(all_boxes, save_folder)

if __name__ == '__main__':
    # load net
    num_classes = cfg['num_classes']
    if args.use_res:
        net = build_ssd('test', cfg, 300, num_classes, base='resnet', max_per_image = args.max_per_image) # initialize SSD (resnet)
    elif args.use_m1:
        net = build_mssd('test', cfg, 300, num_classes, base='m1', max_per_image = args.max_per_image) # backbone network is m1
    elif args.use_m2:
        net = build_mssd('test', cfg, 300, num_classes, base='m2', max_per_image = args.max_per_image) # backbone network is m2
    else:
        net = build_ssd('test', cfg, 300, num_classes, base='vgg', max_per_image = args.max_per_image) # initialize SSD (vgg)
    # if you want to eval SSD from original version ssd.pytorch because self.vgg was changed to self.base
    '''
    # load resume SSD network
    state_dict = torch.load(args.trained_model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # change from self.vgg to self.base
        head = k[:4]
        if head == 'vgg.':
            name = 'base.' + k[4:]
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    '''
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
#    dataset = XLDetection(args.voc_root, [set_type], # for VOC_xlab_products dataset
#                           BaseTransform(300, cfg['dataset_mean']),
#                           XLAnnotationTransform())
    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(300, cfg['dataset_mean']),
                           VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, cfg['dataset_mean']), args.max_per_image,
             thresh=args.confidence_threshold)
