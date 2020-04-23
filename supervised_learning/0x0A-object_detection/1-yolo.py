#!/usr/bin/env python3
"""
Yolo Object Detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """Class constructor"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to Darknet Keras model
        classes_path: path to list of class names for Darnet model
                      list in order of index
        class_t: float = box score threshold for inital filtering
        nms_t: float for IoU threshold for non-max suppression
        anchors: numpy.darray shape(outputs, anchor_boxes, 2)
                 contains all anchor boxes
        ouputs: num Darknet Model outputs
        anchor_boxes: num anchor boxes used for each prediction
        2 -> [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classFile:
            self.class_names = [line.strip() for line in classFile]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors


def process_outputs(self, outputs, image_size):
    """
    Process Outputs Darknet
    outputs: list of numpy.ndarray s; contains Darknet predictions from image
            shape(grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
        grid_height: height of grid used for out
        grid_width: width of grid used for out
        anchor_boxes: num of anchor boxes used
        4: (t_x, t_y, t_w, t_h)
        1: box_confidence
        classes: classes probabilites for all classes
    image_size: numpy.ndarray containing image’s original size
        [image_height, image_width]
    Returns: (boxes, box_confidences, box_class_probs)
        boxes: list numpy.ndarrays
               shape(grid_height, grid_width, anchor_boxes, 4)
               containing the processed boundary boxes for each out
        box_class_probs: list numpy.ndarrays
               shape(grid_height, grid_width, anchor_boxes, classes)
               containing box’s class probabilities for each
        box_confidences: list of numpy.ndarrays
               shape (grid_height, grid_width, anchor_boxes, 1)
               containing box confidences for each out
    """

    boxes = []
    box_class_probs = []
    box_confidences = []

    for out in range(len(outputs)):
        grid_h = outputs[out].shape[0]
        grid_w = outputs[out].shape[1]
        anchor_boxes = outputs[out].shape[2]
        _ = outputs[out].shape[3]

        # box_p, box_c: bounding box parameters, sigmoid
        box_c = 1 / (1 + np.exp(-(outputs[out][:, :, :, 4:5])))
        box_confidences.append(box_c)
        box_p = 1 / (1 + np.exp(-(outputs[out][:, :, :, 5:])))
        box_class_probs.append(box_p)

        # grid boxes
        xybox = 1 / (1 + np.exp(-(outputs[out][:, :, :, :2])))
        whbox = np.exp(outputs[out][:, :, :, 2:4])
        anchor_tensor = self.anchors.reshape(1, 1, self.anchors.shape[0],
                                             anchor_boxes, 2)
        whbox = whbox * anchor_tensor[:, :, out, :, :]

        # grid image tile arange reshape
        imcol = np.tile(np.arange(0, grid_w),
                        grid_h).reshape(grid_h, grid_w)
        imrow = np.tile(np.arange(0, grid_h),
                        grid_w).reshape(grid_w, grid_h).T
        row = imrow.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
        col = imcol.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=2)
        imgrid = np.concatenate((col, row), axis=3)

        # build boxes
        xybox = (xybox * imgrid)
        xybox /= (grid_w, grid_h)
        model_inputH = self.model.input.shape[2].value
        model_inputW = self.model.input.shape[1].value

        whbox = (whbox / (model_inputW, model_inputH))
        xybox -= (whbox / 2)
        xybox2 = xybox + whbox
        box = np.concatenate((xybox, xybox2), axis=-1)

        box[..., 0] *= image_size[1]
        box[..., 2] *= image_size[1]
        box[..., 1] *= image_size[0]
        box[..., 3] *= image_size[0]
        boxes.append(box)

    return (boxes, box_confidences, box_class_probs)
