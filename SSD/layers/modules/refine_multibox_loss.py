# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, refine_match, log_sum_exp, decode

from data import coco as cfg # TODO: for self.variance, need to udpate for different dataset

class RefineMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 object_score = 0, use_gpu=True):
        super(RefineMultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.object_score = object_score
        self.variance = [0.1,0.2]

    def forward(self, odm_data, priors, targets, arm_data = None, filter_object = False):
        """Multibox Loss for RefineSSD
        Args:
            odm_data (tuple): A tuple containing loc data, conf data from RefineSSD net
                conf shape: batch_size, num_priors, num_classes
                loc shape: batch_size, num_priors, 4

            priors: prior boxes from SSD net
                shape: num_priors, 4

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).

            arm_data (tuple): arm branch containing arm_loc and arm_conf

            filter_object: whether filter out the prediction according to the arm conf score
        """

        loc_data, conf_data = odm_data
        # when base on ARM filtering result
        if arm_data:
            arm_loc, arm_conf = arm_data
        #priors = priors.data[:loc_data.size(1), :]
        priors = priors.data
        num = loc_data.size(0) #batch size
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        # loc and conf results after matching will store in following two Tensor
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)# top label for each prior
        for idx in range(num): # for each image
            truths = targets[idx][:,:-1].data
            labels = targets[idx][:,-1].data
            #for object detection
            if self.num_classes == 2:
                labels = labels > 0
            if arm_data:
                #match and calculate loc&conf based on arm output
                refine_match(self.threshold, truths, priors, self.variance, labels,
                             loc_t, conf_t, idx, arm_loc[idx].data)
            else:
                match(self.threshold, truths, priors, self.variance, labels,
                      loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        # positive samples
        # Negative Anchor Filtering
        if arm_data and filter_object:
            # for arm_conf, only two classes, filtering only focus on forground objects - class index == 1
            # the arm_conf will not be considered for loss, but it will be used for filtering
            arm_conf_data = arm_conf.data[:,:,1]
            pos = conf_t > 0 # original pos without filtering by arm
            object_score_index = arm_conf_data <= self.object_score # discard positive confidence smaller than 0.01 OR negative confidence larger than 0.99
            pos[object_score_index] = 0 # false

        else:
            pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)# filtering
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1,self.num_classes) # batch_size*num_priors
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))# get the score indicated by the list named conf_t

        # Hard Negative Mining
        loss_c[pos] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _,loss_idx = loss_c.sort(1, descending=True)# only sort within every image, so that filter negative based on one image itself
        _,idx_rank = loss_idx.sort(1)# so that as long as the number itself < num_neg, the loss it represents should be kept
        num_pos = pos.long().sum(1,keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)# keep how many neg example
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data) # pos_idx only focus on pos samples
        neg_idx = neg.unsqueeze(2).expand_as(conf_data) # neg_idx only focus on negative samples
        # so adding them tgt won't affect each other, but doing combination
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes) # score for every class for each box
        targets_weighted = conf_t[(pos+neg).gt(0)] # only the target label
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        loss_l/=N
        loss_c/=N
        return loss_l,loss_c
