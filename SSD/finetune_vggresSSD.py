'''
    Finetune prunned model vggSSD (Train/Test on VOC)
    Execute: python3 finetune_vggresSSD.py --pruned_model prunes/_your_prunned_model_ --lr x --epoch y

    Finetune prunned model resnetSSD (Train/Test on VOC)
    Execute: python3 finetune_vggresSSD.py --use_res --pruned_model prunes/_your_prunned_model_ --lr x --epoch y
'''
import torch
from torch.autograd import Variable
#from torchvision import models
import cv2
cv2.setNumThreads(0) # pytorch issue 1355: possible deadlock in DataLoader
# OpenCL may be enabled by default in OpenCV3;
# disable it because it because it's not thread safe and causes unwanted GPU memory allocations
cv2.ocl.setUseOpenCL(False)
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import dataset
import argparse
from operator import itemgetter
import time

# for testing
import pickle
import os
from data import *
from data import VOC_CLASSES as labelmap
import torch.utils.data as data
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument("--prune_folder", default = "prunes/")
parser.add_argument("--pruned_model", default = "prunes/vggSSD_prunned")
parser.add_argument('--dataset_root', default=VOC_ROOT)
parser.add_argument("--cut_ratio", default=0.2, type=float)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--epoch", default=20, type=int)
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
# for test_net: 200 in SSD paper, 200 for COCO, 300 for VOC
parser.add_argument('--max_per_image', default=200, type=int,
                    help='Top number of detections kept per image, further restrict the number of predictions to parse')
# use resnet or not
parser.add_argument("--use_res", dest="use_res", action="store_true")
parser.set_defaults(use_res=False)
args = parser.parse_args()

cfg = voc

def test_net(save_folder, net, cuda,
             testset, transform, max_per_image=200, thresh=0.05):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    num_images = len(testset)
    num_classes = len(labelmap)                      # +1 for background
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
        net.phase = 'test'
        detections = net(x=x).data #, test=True # get the detection results, max_per_image = 300 takes effect inside
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
            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

    #write the detection results into det_file
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    APs,mAP = testset.evaluate_detections(all_boxes, save_folder)

    return mAP # for model storing

# --------------------------------------------------------------------------- Finetune Part
class FineTuner_vggresSSD:
    def __init__(self, train_loader, testset, criterion, model):
        self.train_data_loader = train_loader
        self.testset = testset

        self.model = model
        self.criterion = criterion
        self.model.train()

    def test(self):
        self.model.eval()
        # evaluation
        map = test_net('prunes/test', self.model, args.cuda, testset,
                 BaseTransform(self.model.size, cfg['dataset_mean']),
                 args.max_per_image, thresh=0.01)
        self.model.train()
        return map

    # epoches: fine tuning for this epoches
    def train(self, optimizer = None, epoches = 5):
        if optimizer is None:
            optimizer = \
                optim.SGD(self.model.parameters(),
                    lr=0.0001, momentum=0.9, weight_decay=5e-4)

        for i in range(epoches):
            print("FineTune... Epoch: ", i+1)
            self.train_epoch(optimizer) # no need for rank_filters
            if i == (epoches-1):
              map = self.test()
              print("Finished fine tuning. mAP is ", map)
              return map

    # batch: images, label: targets
    def train_batch(self, optimizer, batch, label):
        # set gradients of all model parameters to zero
        self.model.zero_grad() # same as optimizer.zero_grad() when SGD() get model.parameters
        # input = Variable(batch)
        input = batch
        # make priors cuda()
        loc_, conf_, priors_ = self.model(input)
        if args.cuda:
            priors_ = priors_.cuda()

        loss_l, loss_c = self.criterion((loc_, conf_, priors_), label)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step() # update params

    # train for one epoch, so the data_loader will not pop StopIteration error
    def train_epoch(self, optimizer = None):
        num_batch = 0
        for batch, label in self.train_data_loader:
            num_batch += 1
            if num_batch % 50 == 0:
                print("Training batch " + repr(num_batch) + "/" + repr(len(self.train_data_loader)-1) + "...")
            batch = Variable(batch.cuda())
            label = [Variable(ann.cuda(), volatile=True) for ann in label]
            self.train_batch(optimizer, batch, label)

if __name__ == '__main__':
    if not args.cuda:
        print("this file only supports cuda version now!")

    # store pruning models
    if not os.path.exists(args.prune_folder):
        os.mkdir(args.prune_folder)

    print(args)
    # load model from previous pruning
    model = torch.load(args.pruned_model).cuda()
    print('Finished loading model!')

    # data
    dataset = VOCDetection(root=args.dataset_root,
                           transform=SSDAugmentation(cfg['min_dim'], cfg['dataset_mean']))
    testset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'test')],
                                transform=BaseTransform(cfg['min_dim'], cfg['testset_mean']))
    data_loader = data.DataLoader(dataset, 32, num_workers=4,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True) #len(data_loader) == 518

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    fine_tuner = FineTuner_vggresSSD(data_loader, testset, criterion, model)

    # ------------------------ adjustable part
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    map = fine_tuner.train(optimizer = optimizer, epoches = args.epoch)
    # ------------------------ adjustable part

    print('Saving finetuned model with map ', map, '...')
    if args.use_res:
        torch.save(model, 'prunes/resnetSSD_finetuned_{0:.2f}'.format(map*100))
    else:
        torch.save(model, 'prunes/vggSSD_finetuned_{0:.2f}'.format(map*100))
