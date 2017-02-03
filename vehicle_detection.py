import cv2
import numpy as np
import matplotlib.image as mpimg

def draw_boxes(img, bboxes, color=(0, 0, 255), thickness=6):
    '''
    Draw bounding boxes over an image.

    :param img: the original image
    :param bboxes: coordinates in the form ((x1, y1), (x2, y2))
    :param color: color for the bounding box
    :param thickness: line thickness
    :return: original image with the bounding boxes drawn on
    '''
    draw_img = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thickness)
    return draw_img


def data_look(car_list, notcar_list):
    '''
    Gets basic information about the images in the dataset
    
    :param car_list:
    :param notcar_list:
    :return:
    '''
    data_dict = {}

    data_dict["n_cars"] = len(car_list)
    data_dict["n_notcars"] = len(notcar_list)

    example_img = mpimg.imread(car_list[0])
    data_dict["image_shape"] = example_img.shape
    data_dict["data_type"] = example_img.dtype
    return data_dict