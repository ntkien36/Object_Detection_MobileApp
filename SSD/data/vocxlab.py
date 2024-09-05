"""VOC Dataset Classes
    For X-Lab dataset mimicing VOC dataset
"""
from .config import HOME
import pickle
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
cv2.setNumThreads(0) # pytorch issue 1355: possible deadlock in DataLoader
# OpenCL may be enabled by default in OpenCV3;
# disable it because it because it's not thread safe and causes unwanted GPU memory allocations
cv2.ocl.setUseOpenCL(False)
import numpy as np
from .voc_eval import voc_eval

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

XL_CLASSES = ('none_of_the_above', # always index 0
'balishui', 'quecaonatie', 'hongniu', 'nongfunfc',
'jiaduobaoguan', 'quchenshiyuanwei', 'beibingyangjuqi', 'maidongqingningkouwei',
'xingbakemokawei', 'kalabaoweishengsu', 'sanyuanchunnai',
'lingdukele', 'kangshifubinghongcha', 'kuerlexiangli', 'yantaifushipingguo',
'tianranjiaomumianbaonyw', 'laoxiangyangshougongguobawxw', 'tangdarenrishitungulamian',
'haoliyoutilamisuliumeizhuang', 'junzaijuanmianjinxianglawei170g',
'tongyilaotansuancaimian', 'lizhi', 'shiliulvcha', 'kongshou')

#take in target and transform it into list of [bbox coords, label]
#take in ElementTree-type "target" transformed from xml and extract useful annotation information from this "target" object
class XLAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    Returns:
        a list containing lists of bounding boxes  [bbox coords, class idx]
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(XL_CLASSES, range(len(XL_CLASSES))))# make own look up dict if None
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):# in VOC, target is from xml tree
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width for vggSSD and resnetSSD but NOT for refineDet if use preproc()
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

#inheritance of data.Dataset
#For a dataset object, you should have functions to get an item/image for use (from a directory in "root")
class XLDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOC_xlab_products folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=['trainval'], # or 'test'
                 transform=None, target_transform=XLAnnotationTransform(),
                 dataset_name='VOC_XLab'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations_24class', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list() # store the names for each image
        for name in image_sets:#trainval
            rootpath = self.root
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip())) # a tuple of (rootpath, img_name)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        good = False # to skip error data

        while good is not True:
            img_id = self.ids[index]
            target = ET.parse(self._annopath % img_id).getroot()
            img = cv2.imread(self._imgpath % img_id)

            if img is not None: # the image is correct
                height, width, channels = img.shape
                if len(np.array(self.target_transform(target, width, height)).shape) == 2:
                    good = True
            index += 1

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # height, width, channels = img.shape #DONT update height and width after transform
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality. i.e. Do NOT use inherited function

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        #target_transfrom will transform a target "anno" to [(label, bbox coords)
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        #returns a new tensor with a dimension of size 1 inserted at the specified position
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        # write down the detection results
        self._write_voc_results_file(all_boxes)
        # after getting the result file, do evaluation and store in output_dir
        aps, map = self._do_python_eval(output_dir)
        return aps, map

    def _get_voc_results_file_template(self):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(self.root, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(XL_CLASSES):
            cls_ind = cls_ind
            if cls == 'none_of_the_above':
                continue
            print('Writing {} XL results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        # for a class in an image: {image_id} {score} {xcor} {xcor} {ycor} {ycor}
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        rootpath = self.root
        name = self.image_set[0]
        annopath = os.path.join(
            rootpath,
            'Annotations_24class',
            '%s.xml')
        imagesetfile = os.path.join(
            rootpath,
            'ImageSets',
            'Main',
            name + '.txt')
        cachedir = os.path.join(self.root, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True # if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(XL_CLASSES):

            if cls == 'none_of_the_above':
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            # AP = AVG(Precision for each of 11 Recalls's precision)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        # MAP = AVG(AP for each object class)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        return aps, np.mean(aps)
