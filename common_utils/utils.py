from collections import OrderedDict

import numpy as np
import math
import cv2
from PIL import Image

from common_utils.object_model import ResultObject


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def diff(input_list):
    return max(input_list) - min(input_list)


def group_text_box(polys, slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=1.0, add_margin=0.05):
    # poly top-left, top-right, low-right, low-left
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    for poly in polys:
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            horizontal_list.append([x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min])
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            margin = int(1.44 * add_margin * height)

            theta13 = abs(np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))))
            theta24 = abs(np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))))
            # do I need to clip minimum, maximum value here?
            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin

            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box = []
    for poly in horizontal_list:

        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths*height
            if (abs(np.mean(b_height) - poly[5]) < height_ths * np.mean(b_height)) and (
                    abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height)):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1:  # one box per line
            box = boxes[0]
            margin = int(add_margin * box[5])
            merged_list.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])
        else:  # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if abs(box[0] - x_max) < width_ths * (box[3] - box[2]):  # merge boxes
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # adjacent box in same line
                    # do I need to add margin here?
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    margin = int(add_margin * (y_max - y_min))

                    merged_list.append([x_min - margin, x_max + margin, y_min - margin, y_max + margin])
                else:  # non adjacent box in same line
                    box = mbox[0]

                    margin = int(add_margin * (box[3] - box[2]))
                    merged_list.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])
    # may need to check if box is really in image
    return merged_list, free_list


def get_image_list(horizontal_list, free_list, img, model_height=64, rotate_ratio=None):
    image_list = []
    maximum_y, maximum_x = img.shape

    max_ratio_hori, max_ratio_free = 1, 1
    for box in free_list:
        rect = np.array(box, dtype="float32")
        transformed_img = four_point_transform(img, rect)
        ratio = transformed_img.shape[1] / transformed_img.shape[0]
        if rotate_ratio is not None and ratio < rotate_ratio:
            transformed_img = cv2.rotate(transformed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ratio = 1 / ratio
        crop_img = cv2.resize(transformed_img, (int(model_height * ratio), model_height), interpolation=Image.ANTIALIAS)
        image_list.append(ResultObject(box, crop_img))  # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        max_ratio_free = max(ratio, max_ratio_free)

    max_ratio_free = math.ceil(max_ratio_free)

    for box in horizontal_list:
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        crop_img = img[y_min: y_max, x_min:x_max]
        width = x_max - x_min
        height = y_max - y_min
        if height == 0 or width == 0:
            continue
        ratio = width / height
        if rotate_ratio is not None and ratio < rotate_ratio:
            crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ratio = 1 / ratio
        crop_img = cv2.resize(crop_img, (int(model_height * ratio), model_height), interpolation=Image.ANTIALIAS)
        image_list.append(ResultObject([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], crop_img))
        max_ratio_hori = max(ratio, max_ratio_hori)

    max_ratio_hori = math.ceil(max_ratio_hori)
    max_ratio = max(max_ratio_hori, max_ratio_free)
    max_width = math.ceil(max_ratio) * model_height

    image_list = sorted(image_list, key=lambda item: item.get_coord()[0][1])  # sort by vertical position
    return image_list, max_width


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def get_paragraph(raw_result, x_ths=1, y_ths=0.5):
    # create basic attributes
    box_group = []
    for box in raw_result:
        all_x, min_x, max_x = box.get_all_x()
        all_y, min_y, max_y = box.get_all_y()
        height = max_y - min_y
        # last element indicates group
        box_group.append(
            [box.get_label(), min_x, max_x, min_y, max_y, height, 0.5 * (min_y + max_y), 0])

    # cluster boxes into paragraph
    current_group = 1
    while len([box for box in box_group if box[7] == 0]) > 0:
        box_group0 = [box for box in box_group if box[7] == 0]  # group0 = non-group
        # new group
        if len([box for box in box_group if box[7] == current_group]) == 0:
            box_group0[0][7] = current_group  # assign first box to form new group
        # try to add group
        else:
            current_box_group = [box for box in box_group if box[7] == current_group]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group]) - x_ths * mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths * mean_height
            min_gy = min([box[3] for box in current_box_group]) - y_ths * mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths * mean_height
            add_box = False
            for box in box_group0:
                same_horizontal_level = (min_gx <= box[1] <= max_gx) or (min_gx <= box[2] <= max_gx)
                same_vertical_level = (min_gy <= box[3] <= max_gy) or (min_gy <= box[4] <= max_gy)
                if same_horizontal_level and same_vertical_level:
                    box[7] = current_group
                    add_box = True
                    break
            # cannot add more box, go to next group
            if add_box == False:
                current_group += 1
    # arrage order in paragraph
    result = []
    for i in set(box[7] for box in box_group):
        current_box_group = [box for box in box_group if box[7] == i]
        mean_height = np.mean([box[5] for box in current_box_group])
        min_gx = min([box[1] for box in current_box_group])
        max_gx = max([box[2] for box in current_box_group])
        min_gy = min([box[3] for box in current_box_group])
        max_gy = max([box[4] for box in current_box_group])

        text = ''
        while len(current_box_group) > 0:
            highest = min([box[6] for box in current_box_group])
            candidates = [box for box in current_box_group if box[6] < highest + 0.4 * mean_height]
            # get the far left
            most_left = min([box[1] for box in candidates])
            for box in candidates:
                if box[1] == most_left:
                    best_box = box
            text += ' ' + best_box[0]
            current_box_group.remove(best_box)

        result.append([[[min_gx, min_gy], [max_gx, min_gy], [max_gx, max_gy], [min_gx, max_gy]], text[1:]])
    return result


def get_rec_box(img, box, maximum_y, maximum_x, model_height, rotate_ratio=None):
    # [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
    if len(box) == 4:
        x_min = max(0, int(box[0]))
        x_max = min(int(box[1]), maximum_x)
        y_min = max(0, int(box[2]))
        y_max = min(int(box[3]), maximum_y)
    else:
        x_min = max(0, box[0])
        x_max = min(box[2], maximum_x)
        y_min = max(0, box[1])
        y_max = min(box[5], maximum_y)
    crop_img = img[y_min: y_max, x_min:x_max]
    width = x_max - x_min
    height = y_max - y_min
    ratio = width / height
    if rotate_ratio is not None and ratio < rotate_ratio:
        crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ratio = 1 / ratio
    crop_img = cv2.resize(crop_img, (int(model_height * ratio), model_height), interpolation=Image.ANTIALIAS)
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], crop_img, ratio


def get_poly_box(img, box, model_height, rotate_ratio=None):
    rect = np.array(box, dtype="float32")
    rect = rect.reshape(4, 2)
    transformed_img = four_point_transform(img, rect)
    ratio = transformed_img.shape[1] / transformed_img.shape[0]
    if rotate_ratio is not None and ratio < rotate_ratio:
        transformed_img = cv2.rotate(transformed_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ratio = 1 / ratio
    crop_img = cv2.resize(transformed_img, (int(model_height * ratio), model_height), interpolation=Image.ANTIALIAS)
    return rect.tolist(), crop_img, ratio


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
