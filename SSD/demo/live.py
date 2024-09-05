from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
cv2.setNumThreads(0) # pytorch issue 1355: possible deadlock in DataLoader
# OpenCL may be enabled by default in OpenCV3;
# disable it because it because it's not thread safe and causes unwanted GPU memory allocations
cv2.ocl.setUseOpenCL(False)
import time
from imutils.video import FPS, WebcamVideoStream
import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y[0]
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i], (int(pt[0]), int(pt[1])),          #- 1
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = WebcamVideoStream(src=0).start()  # default camera
    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        frame = stream.read()
        key = cv2.waitKey(1) & 0xFF

        # update FPS counter
        fps.update()
        frame = predict(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break


if __name__ == '__main__':
    import sys
    import csv
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from models.SSD_vggres import build_ssd
    from data import *

    # net = build_ssd('test', 300, 21, base='vgg')    # initialize SSD
    # net.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
    net = torch.load(args.weights, map_location=torch.device('cpu'))
    
    # cfg = voc
    # net = build_ssd('test', cfg, 300, 21, base='vgg', max_per_image = 200)    # initialize SSD
    # state_dict = torch.load(args.weights, map_location=torch.device('cpu'))
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     head = k[:7] # head = k[:4]
    #     if head == 'module.': # head == 'vgg.', module. is due to DataParellel
    #         name = k[7:]  # name = 'base.' + k[4:]
    #     else:
    #         name = k
    #     new_state_dict[name] = v
    # net.load_state_dict(new_state_dict)

    transform = BaseTransform(300, (104/256.0, 117/256.0, 123/256.0))   #net.size

    fps = FPS().start()
    cv2_demo(net.eval(), transform)
    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    with open('FPS.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([fps.elapsed(), fps.fps()])

    # cleanup
    cv2.destroyAllWindows()
    stream.stop()
