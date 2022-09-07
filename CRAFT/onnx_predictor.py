from CRAFT.craft import CRAFT
from CRAFT.utils import get_det_boxes, adjust_result_coordinates
from common_utils.imgproc import reformat_input_image, resize_aspect_ratio, normalize_mean_variance
from common_utils.utils import copy_state_dict, diff, group_text_box, get_image_list
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from torch.autograd import Variable

CRAFT_ROTATE_RATIO = 0.65
LINE_THRESHOLD = 0.0375


class CRAFTDetector:
    def __init__(self, trained_model, device='cpu', img_h=64, min_size=20, text_threshold=0.7, low_text=0.4,
                 link_threshold=0.4, poly=False, canvas_size=2560, mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5,
                 height_ths=0.5, width_ths=0.5, add_margin=0.1, rotate_ratio=CRAFT_ROTATE_RATIO,
                 num_torch_threads=None):
        self.device = device
        self.img_h = img_h
        self.poly = poly
        self.min_size = min_size
        self.text_threshold = text_threshold  # text confidence threshold
        self.low_text = low_text  # text low-bound score
        self.link_threshold = link_threshold  # link confidence threshold
        self.canvas_size = canvas_size  # max image size for inference
        self.mag_ratio = mag_ratio  # image magnification ratio
        self.slope_ths = slope_ths
        self.ycenter_ths = ycenter_ths
        self.height_ths = height_ths
        self.width_ths = width_ths
        self.add_margin = add_margin
        self.rotate_ratio = rotate_ratio
        self.num_torch_threads = num_torch_threads

        if self.num_torch_threads is not None:
            torch.set_num_threads(self.num_torch_threads)
        net = CRAFT()
        weights = copy_state_dict(torch.load(trained_model, map_location=device))

        if device == 'cpu':
            net.load_state_dict(weights)
        else:
            net.load_state_dict(weights)
            net = torch.nn.DataParallel(net).to(device)
            cudnn.benchmark = False
        net.eval()

        self.net = net

    def detect(self, image):
        img_cv_grey = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

        text_box = []
        bboxes, polys = self.fit(image)

        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            text_box.append(poly)

        horizontal_list, free_list = group_text_box(text_box, self.slope_ths,
                                                    self.ycenter_ths, self.height_ths,
                                                    self.width_ths, self.add_margin)

        if self.min_size:
            horizontal_list = [i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > self.min_size]
            free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > self.min_size]

        image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height=self.img_h,
                                               rotate_ratio=self.rotate_ratio)
        return image_list

    def detect_image(self, image, up_red=True, size=600):
        img_grey = reformat_input_image(image, up_red, size)
        img = cv2.cvtColor(img_grey, cv2.COLOR_GRAY2BGR)
        text_box = []
        bboxes, polys = self.fit(img)

        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            text_box.append(poly)

        horizontal_list, free_list = group_text_box(text_box, self.slope_ths,
                                                    self.ycenter_ths, self.height_ths,
                                                    self.width_ths, self.add_margin)

        if self.min_size:
            horizontal_list = [i for i in horizontal_list if max(i[1] - i[0], i[3] - i[2]) > self.min_size]
            free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > self.min_size]

        image_list, max_width = get_image_list(horizontal_list, free_list, img_grey, model_height=self.img_h,
                                               rotate_ratio=self.rotate_ratio)
        return image_list, img

    def fit(self, image):
        # resize
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, self.canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = normalize_mean_variance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        x = x.to(self.device)

        # forward pass
        with torch.no_grad():
            if self.num_torch_threads is not None:
                torch.set_num_threads(self.num_torch_threads)
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys = get_det_boxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text,
                                     self.poly)

        # coordinate adjustment
        boxes = adjust_result_coordinates(boxes, ratio_w, ratio_h)
        polys = adjust_result_coordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        return boxes, polys
