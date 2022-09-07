import cv2
import torch
import numpy as np

from craft import CRAFT
from collections import OrderedDict
from torch.autograd import Variable
from utils import copy_state_dict, resize_img, preprocess


def craft_to_onnx(weight_path, img):
    model = CRAFT()
    weights = copy_state_dict(torch.load(weight_path, map_location='cpu'))
    model.load_state_dict(weights)
    model.eval()

    img = cv2.imread(img)
    img = resize_img(600 / int(img.shape[1]), img)

    # set the right shape for input
    # mine is (360, 600, 3)

    dummy_input = preprocess(img)
    dummy_output = model(dummy_input)
    if dummy_output:
        print('valid input')
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "craft.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      verbose=True,
                      # do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    print('finish convert onnx')


def craft_to_onnx_dynamic_axes(weight_path, img):
    model = CRAFT()
    weights = copy_state_dict(torch.load(weight_path, map_location='cpu'))
    model.load_state_dict(weights)
    model.eval()

    img = cv2.imread(img)
    img = resize_img(600 / int(img.shape[1]), img)

    dummy_input = preprocess(img)
    dummy_output = model(dummy_input)
    if dummy_output:
        print('valid input')

    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "craft.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      verbose=True,
                      # do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}}
                      )
    print('finish convert onnx')


if __name__ == '__main__':
    canvas_size = 2560
    mag_ratio = 1.
    device = 'cpu'

    weight_path = '/home/phamson/gitlab/ekyc-people-id/data/model/craft_mlt_25k.pth'
    img = '/home/phamson/data/EKYC/CMT_data/CMT_real/CMT_real_3/16_04_04.jpg'

    # crop_img = resize_img(600 / int(img.shape[1]), img)
    # print(crop_img.shape)
    craft_to_onnx(weight_path, img)
    # preprocess(resize_img(600, cv2.imread(img))).shape
