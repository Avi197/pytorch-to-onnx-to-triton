from skimage import io
from deskew import determine_skew
from typing import Tuple, Union
import numpy as np
import cv2
import math
from skimage.transform import rotate


def cv_rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def resize_img(scale_percent, img):
    width = int(img.shape[1] * scale_percent)
    if width > 600 or width < 600:
        width = 600
    height = int(img.shape[0] * scale_percent)
    if height > 360 or height < 360:
        height = 360
    dsize = (width, height)
    return cv2.resize(img, dsize)


def reformat_input(image_path, min_skew_angle=5.0, resize=True, size=600):
    if type(image_path) is str:
        img = cv2.imread(image_path)
    else:
        img = image_path
    if resize:
        img = resize_img(size / int(img.shape[1]), img)
    img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(img_cv_grey)
    if angle < -45.0:
        angle += 90.0
    if angle != 0 and abs(angle) <= min_skew_angle:
        img_cv_grey = cv_rotate(img_cv_grey, angle, (0, 0, 0))
        return cv_rotate(img, angle, (0, 0, 0)), img_cv_grey
    return img, img_cv_grey


def deskew(image_path, min_skew_angle=5.0):
    if type(image_path) is str:
        img = cv2.imread(image_path)
    elif type(image_path) is np.ndarray:
        img = image_path
    else:
        return None
    img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(img_cv_grey)
    if angle < -45.0:
        angle = -90.0
    elif angle > 45.0:
        angle = 90.0
    elif abs(angle) >= min_skew_angle:
        angle = 0.0
    if angle != 0:
        return cv_rotate(img, angle, (0, 0, 0))
    return img


def load_image(img_file, angle=0.0):
    img = io.imread(img_file)  # RGB order
    if angle != 0:
        img = rotate(img, angle, resize=True) * 255
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def cvt2_heatmap_img(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


# -> img: array, upper do/tim -> convert thanh gray_scale theo ti le uu tien/ mau do/tim
def mul_channel(channel, scalar):
    update_channel = scalar * channel
    update_channel[update_channel >= 255] = 255
    return update_channel


def convert_to_gray_red_2(img, up_red):
    hsv_enhance_red = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_enhance_red = np.copy(img)

    b_point = 0.114 * img_enhance_red[:, :, 0]
    g_point = 0.587 * img_enhance_red[:, :, 1]
    r_point = 0.299 * img_enhance_red[:, :, 2]
    temp = b_point + g_point + r_point
    temp = np.uint8(temp)
    img_enhance_red = np.repeat(temp[:, :, np.newaxis], 3, axis=2)

    h_channel = hsv_enhance_red[:, :, 0]
    s_channel = hsv_enhance_red[:, :, 1]
    v_channel = hsv_enhance_red[:, :, 2]

    h_mask = ((h_channel >= 165) & (h_channel <= 179)) | ((h_channel >= 0) & (h_channel <= 15))
    s_mask = s_channel > 35
    v_mask = v_channel > 20
    mask = h_mask & s_mask & v_mask

    s_update = mul_channel(s_channel, 2)

    hsv_upper_red_point = np.uint8(np.dstack((h_channel, s_update, v_channel)))
    bgr_point = cv2.cvtColor(hsv_upper_red_point, cv2.COLOR_HSV2BGR)
    if up_red:
        b_point = 0.064 * bgr_point[:, :, 0] * 9 / 10
        g_point = 0.787 * bgr_point[:, :, 1]
        r_point = 0.149 * bgr_point[:, :, 2] * 6 / 10
    else:
        b_point = 2 * 0.114 * bgr_point[:, :, 0]
        g_point = 0.587 * bgr_point[:, :, 1]
        r_point = 3 * 0.299 * bgr_point[:, :, 2]
    temp = b_point + g_point + r_point
    temp = np.round(temp)
    temp[temp >= 255] = 255
    temp = temp.astype(np.uint8)
    for i, j in zip(*np.where(mask)):
        img_enhance_red[i][j] = temp[i][j]

    h_mask = (h_channel >= 135) & (h_channel <= 164)
    s_mask = s_channel > 35
    v_mask = v_channel > 20
    mask = h_mask & s_mask & v_mask

    s_update = mul_channel(s_channel, 2.5)
    v_update = mul_channel(v_channel, 1.3)

    hsv_upper_red_point = np.uint8(np.dstack((h_channel, s_update, v_update)))
    bgr_point = cv2.cvtColor(hsv_upper_red_point, cv2.COLOR_HSV2BGR)
    if up_red:
        b_point = 0.064 * bgr_point[:, :, 0] * 9 / 10
        g_point = 0.787 * bgr_point[:, :, 1]
        r_point = 0.149 * bgr_point[:, :, 2] * 8 / 10
    else:
        b_point = 1 * 0.114 * bgr_point[:, :, 0]
        g_point = 0.587 * bgr_point[:, :, 1]
        r_point = 1 * 0.299 * bgr_point[:, :, 2]
    temp = b_point + g_point + r_point
    temp = np.round(temp)
    temp[temp >= 255] = 255
    temp = temp.astype(np.uint8)
    for i, j in zip(*np.where(mask)):
        img_enhance_red[i][j] = temp[i][j]

    return cv2.cvtColor(img_enhance_red, cv2.COLOR_BGR2GRAY)


def reformat_input_image(image, up_red, size):
    img_enhance_red = convert_to_gray_red_2(resize_img(size / int(image.shape[1]), image), up_red)
    if up_red:
        return cv2.convertScaleAbs(img_enhance_red, alpha=1.4, beta=0)
    else:
        return cv2.convertScaleAbs(img_enhance_red, alpha=1.2, beta=10)
