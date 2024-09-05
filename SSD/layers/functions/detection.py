import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from typing import List

@torch.jit.script
class Detect: #(Function)
    """
    At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k/max_per_image number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes: int, cfg, max_per_image: int, conf_thresh: float, nms_thresh: float):  #, bkg_label
        self.num_classes = num_classes
        # self.background_label = bkg_label
        self.max_per_image = max_per_image
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        # self.variance = cfg['variance']
        self.variance = [0.1, 0.2]  #cfg['variance']

    # def forward(self, loc_data, conf_data, prior_data):
    # def forward(self, loc_data: torch.Tensor, conf_data: torch.Tensor, prior_data: torch.Tensor):   #-> torch.Tensor
    def __call__(self, loc_data: torch.Tensor, conf_data: torch.Tensor, prior_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        # top_k is 200 by default, num is 1 when testing because image input one by one
        output = torch.zeros(num, self.num_classes, self.max_per_image, 5)
        # for each sample, every prior box will have #num_classes conf. scores for it
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        # Decode predictions into bboxes based on loc_data(offsets) and prior_data
        for i in range(num): # for each batch
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms to get bbox for this class
            conf_scores = conf_preds[i].clone() # conf_scores = (num_classes, num_priors)

            for cl in range(1, self.num_classes):
            # for _,cl in enumerate([i for i in range(self.num_classes)]):
                # for particular class, keep those boxes with score greater than threshold
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # scores are those higher than threshold
                scores = conf_scores[cl][c_mask]
                # for this class, no object of this class exists in this image, because all scores < threshold
                # if scores.dim() == 0:
                if scores.size(0) == 0:
                    continue
                #if scores.size(0) == 0:
                #    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                #step 1. remaining boxes for reasonable classes's objects, boxes being kept after conf_thresh
                boxes = decoded_boxes[l_mask].view(-1, 4)
                #step 2. use NMS to remove redundant boxes bounding the same class's object
                # idx of highest scoring and non-overlapping boxes per class
                # ids, count = nms(boxes, scores, self.nms_thresh, self.max_per_image)
                with torch.no_grad():    
                    ids, scores, count = nms(boxes, scores, self.nms_thresh, self.max_per_image)
                    output[i, cl, :count] = \
                    torch.cat((scores.unsqueeze(1), boxes[ids]), 1)
                    # torch.cat((scores[ids[:count]].unsqueeze(1),
                            #    boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.max_per_image).unsqueeze(-1).expand_as(flt)].fill_(0) # this code has no effect
        return output # before
        #return flt # after



# class Detect: #(Function)
#     """
#     At test time, Detect is the final layer of SSD.  Decode location preds,
#     apply non-maximum suppression to location predictions based on conf
#     scores and threshold to a top_k/max_per_image number of output predictions for both
#     confidence score and locations.
#     """
#     def __init__(self, num_classes, bkg_label, cfg, max_per_image, conf_thresh, nms_thresh):
#         self.num_classes = num_classes
#         self.background_label = bkg_label
#         self.max_per_image = max_per_image
#         # Parameters used in nms.
#         self.nms_thresh = nms_thresh
#         if nms_thresh <= 0:
#             raise ValueError('nms_threshold must be non negative.')
#         self.conf_thresh = conf_thresh
#         self.variance = cfg['variance']

#     def forward(self, loc_data, conf_data, prior_data):
#         """
#         Args:
#             loc_data: (tensor) Loc preds from loc layers
#                 Shape: [batch,num_priors*4]
#             conf_data: (tensor) Shape: Conf preds from conf layers
#                 Shape: [batch*num_priors,num_classes]
#             prior_data: (tensor) Prior boxes and variances from priorbox layers
#                 Shape: [1,num_priors,4]
#         """
#         num = loc_data.size(0)  # batch size
#         num_priors = prior_data.size(0)
#         # top_k is 200 by default, num is 1 when testing because image input one by one
#         output = torch.zeros(num, self.num_classes, self.max_per_image, 5)
#         # for each sample, every prior box will have #num_classes conf. scores for it
#         conf_preds = conf_data.view(num, num_priors,
#                                     self.num_classes).transpose(2, 1)
#         # Decode predictions into bboxes based on loc_data(offsets) and prior_data
#         for i in range(num): # for each batch
#             decoded_boxes = decode(loc_data[i], prior_data, self.variance)
#             # For each class, perform nms to get bbox for this class
#             conf_scores = conf_preds[i].clone() # conf_scores = (num_classes, num_priors)

#             for cl in range(1, self.num_classes):
#                 # for particular class, keep those boxes with score greater than threshold
#                 c_mask = conf_scores[cl].gt(self.conf_thresh)
#                 # scores are those higher than threshold
#                 scores = conf_scores[cl][c_mask]
#                 # for this class, no object of this class exists in this image, because all scores < threshold
#                 # if scores.dim() == 0:
#                 if scores.size(0) == 0:
#                     continue
#                 #if scores.size(0) == 0:
#                 #    continue
#                 l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
#                 #step 1. remaining boxes for reasonable classes's objects, boxes being kept after conf_thresh
#                 boxes = decoded_boxes[l_mask].view(-1, 4)
#                 #step 2. use NMS to remove redundant boxes bounding the same class's object
#                 # idx of highest scoring and non-overlapping boxes per class
#                 ids, count = nms(boxes, scores, self.nms_thresh, self.max_per_image)
                
#                 output[i, cl, :count] = \
#                     torch.cat((scores[ids[:count]].unsqueeze(1),
#                                boxes[ids[:count]]), 1)
#         flt = output.contiguous().view(num, -1, 5)
#         _, idx = flt[:, :, 0].sort(1, descending=True)
#         _, rank = idx.sort(1)
#         flt[(rank < self.max_per_image).unsqueeze(-1).expand_as(flt)].fill_(0) # this code has no effect
#         return output # before
#         #return flt # after