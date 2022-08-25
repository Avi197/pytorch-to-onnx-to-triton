import cv2
import onnxruntime
import torch
import numpy as np
import torch.nn.functional as F

from ctpn import CTPN_Model
from utils import gen_anchor, transform_bbox, clip_bbox, filter_bbox, nms, TextProposalConnectorOriented


def detect_resize(self, img):
    rs_img, (rh, rw) = resize_image(img)
    text_boxes = self.detect(rs_img)
    boxes = text_boxes[:, :8]
    boxes[:, ::2] /= rw
    boxes[:, 1::2] /= rh
    return boxes.astype(dtype=int)


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(IMG_SCALE) / float(im_size_min)
    if np.round(im_scale * im_size_max) > IMG_SCALE * 2:
        im_scale = float(IMG_SCALE * 2) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def pth_to_pt(pth_path, pt_path):
    model = CTPN_Model().to(device)
    model.load_state_dict(torch.load(pth_path, map_location='cpu')['model_state_dict'])
    torch.save(model, pt_path)
    print('success')


def preprocess_img(image):
    h, w = image.shape[:2]
    rescale_fac = max(h, w) / 1000
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
        image = cv2.resize(image, (w, h))
    image = image.astype(np.float32) - IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    return image


def cptn_to_onnx(weight_path, img):
    model = CTPN_Model().to(device)
    model.load_state_dict(torch.load(weight_path, map_location='cpu')['model_state_dict'])
    model.eval()

    # set the right shape for input
    # mine is (603, 1000, 3)
    rs_img, (rh, rw) = resize_image(img)
    dummy_input = preprocess_img(rs_img)
    dummy_output = model(dummy_input)

    if dummy_output:
        print('valid input')
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "CTPN.onnx",  # where to save the model
                      # export_params=True,  # store the trained parameter weights inside the model file
                      # do_constant_folding=True,  # whether to execute constant folding for optimization
                      opset_version=11,  # the ONNX version to export the model to
                      verbose=True,
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    print('finish convert onnx')


def preprocess_output(output, h, w):
    prob_thresh = 0.5
    cls, regr = output

    cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
    regr = regr.cpu().numpy()
    anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
    bbox = transform_bbox(anchor, regr)
    bbox = clip_bbox(bbox, [h, w])

    fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
    select_anchor = bbox[fg, :]
    select_score = cls_prob[0, fg, 1]
    select_anchor = select_anchor.astype(np.int32)
    keep_index = filter_bbox(select_anchor, 16)

    select_anchor = select_anchor[keep_index]
    select_score = select_score[keep_index]
    select_score = np.reshape(select_score, (select_score.shape[0], 1))
    nmsbox = np.hstack((select_anchor, select_score))
    keep = nms(nmsbox, 0.3)
    select_anchor = select_anchor[keep]
    select_score = select_score[keep]

    textConn = TextProposalConnectorOriented()
    text_boxes = textConn.get_text_lines(select_anchor, select_score, [h, w])
    return text_boxes


def test_onnx(image):
    providers = ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession('ctpn.onnx', providers=providers)
    input_name = session.get_inputs()[0].name
    image = cv2.imread(image)

    image = preprocess_img(image)
    h, w = image.shape[:2]
    output = session.run(None, {input_name: image})
    text_boxes = preprocess_output(output, h, w)
    return text_boxes


def detect_onnx(image):
    session = onnxruntime.InferenceSession('ctpn.onnx')
    prob_thresh = 0.5
    h, w = image.shape[:2]
    rescale_fac = max(h, w) / 1000
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
        image = cv2.resize(image, (w, h))
        h, w = image.shape[:2]

    image = image.astype(np.float32) - IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    image = to_numpy(image)

    output = session.run(None, {'input': image})
    cls, regr = torch.tensor(output[0]), torch.tensor(output[1])
    cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
    regr = regr.cpu().numpy()
    anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
    bbox = transform_bbox(anchor, regr)
    bbox = clip_bbox(bbox, [h, w])

    fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
    select_anchor = bbox[fg, :]
    select_score = cls_prob[0, fg, 1]
    select_anchor = select_anchor.astype(np.int32)
    keep_index = filter_bbox(select_anchor, 16)

    select_anchor = select_anchor[keep_index]
    select_score = select_score[keep_index]
    select_score = np.reshape(select_score, (select_score.shape[0], 1))
    nmsbox = np.hstack((select_anchor, select_score))
    keep = nms(nmsbox, 0.3)
    select_anchor = select_anchor[keep]
    select_score = select_score[keep]

    textConn = TextProposalConnectorOriented()
    text_boxes = textConn.get_text_lines(select_anchor, select_score, [h, w])

    return text_boxes


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def detect_resize(self, img):
    rs_img, (rh, rw) = resize_image(img)
    text_boxes = self.detect(rs_img)
    boxes = text_boxes[:, :8]
    boxes[:, ::2] /= rw
    boxes[:, 1::2] /= rh
    return boxes.astype(dtype=int)


def draw_box(img, boxes):
    for box in boxes:
        x_min = int(min(box[:8][::2]))
        x_max = int(max(box[:8][::2]))
        y_min = int(min(box[:8][1::2]))
        y_max = int(max(box[:8][1::2]))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow('image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    device = 'cpu'
    IMAGE_MEAN = [123.68, 116.779, 103.939]
    weight_path = '/home/phamson/gitlab/ekyc-people-id/data/model/ctpn/weights.pth'
    pt_path = 'ctpn.pt'
    IMG_SCALE = 600
    img = '/home/phamson/data/EKYC/CMT_data/CMT_real/CMT_real_3/16_04_04.jpg'
    img = cv2.imread(img)
    # boxes = detect_onnx(img)
    # draw_box(img, boxes)
    cptn_to_onnx(weight_path, img)

    # test_onnx(img)
    # detect_ctpn(weight_path, img)
    # model = torch.load(pt_path)
    # img = '/home/phamson/data/EKYC/CMT_data/CMT_real/CMT_real_3/16_04_04.jpg'
    # dummy_input = get_dummy_input(img)
    # dummy_output = model(dummy_input)
    # torch.onnx.export(model, dummy_input, 'CTPN.onnx')
