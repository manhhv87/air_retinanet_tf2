"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications copyright (c) 2020-2021 Accenture
"""

# Modified by Pasi Pyrrö 16.7.2020
# Did Weights & Biases integration
# Implemented merge_overlapping_boxes

import os
import sys

from PIL.Image import merge

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import keras_retinanet.utils
    __package__ = "keras_retinanet.utils"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "airutils"))
import imtools
from mob import merge_boxes_per_label

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compute_overlap import compute_overlap
from .transform import clip_aabb_array
from .image import resize_image as resize_func
from .visualization import draw_detections, draw_annotations
from ..preprocessing.pascal_voc import voc_classes, PascalVocGenerator
from ..preprocessing.inference import InferenceGenerator

class_id_to_voc_label = {v: k for k, v in voc_classes.items()}

from tensorflow import keras
import numpy as np
import time

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

MAX_NUM_UPLOAD_IMAGES = 10
DISABLE_IMG_UPLOADS = False
HERIDAL_VIS_SETTINGS = True
TILING_OVERLAP = 100
TRAIN_OVERLAP = 50


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def run_inference_on_image(
    model,
    image,
    resize_image=None,
    score_threshold=0.01, 
    max_detections=1000,
    tiling_dim=1,
    top_k=50,
    nms_threshold=0.5,
    nms_mode="argmax",
    mob_iterations=3,
    max_inflation_factor=100,
    tiling_overlap=TILING_OVERLAP,
):
    orig_img_shape = image.shape

    
    tile_size = imtools.grid_dim_to_tile_size(image, tiling_dim, overlap=tiling_overlap)
    image_parts, offsets = imtools.create_overlapping_tiled_mosaic(image, tile_size=tile_size,
                                                                    overlap=tiling_overlap, no_fill=True)
    
    all_boxes, all_scores, all_labels = [], [], []

    for image, offset in zip(image_parts, offsets):
        offset = np.tile(offset[::-1], 2)
        if resize_image is not None:
            image, scale = resize_image(image)
        else:
            scale = 1

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale and translate subtiles to whole 
        # image coordinate frame
        boxes /= scale
        boxes += offset

        all_boxes.append(clip_aabb_array(orig_img_shape, boxes))
        all_scores.append(scores)
        all_labels.append(labels)

    boxes = np.concatenate(all_boxes, axis=1)
    scores = np.concatenate(all_scores, axis=1)
    labels = np.concatenate(all_labels, axis=1)

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0, indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes      = boxes[0, indices[scores_sort], :].astype(np.float64)
    image_scores     = scores[scores_sort]
    image_labels     = labels[0, indices[scores_sort]]

    if nms_mode != "none":
        if nms_mode != "argmax":
            image_boxes, image_scores, image_labels = merge_boxes_per_label(image_boxes, image_scores, image_labels, 
                                                                            max_iterations=mob_iterations, top_k=top_k,
                                                                            iou_threshold=nms_threshold, merge_mode=nms_mode,
                                                                            max_inflation_factor=max_inflation_factor)
        elif tiling_dim > 1: # we need to perform another NMS for tiling duplicates
            # not using MOB to ensure reproducibility of earlier results, although it should be the same thing
            # for performance sake, using MOB now
            image_boxes, image_scores, image_labels = merge_boxes_per_label(image_boxes, image_scores, image_labels,
                                                                            merge_mode="enclose", max_iterations=1,
                                                                            iou_threshold=nms_threshold, top_k=1)
    
    # find the order with which to sort the scores
    scores_sort = np.argsort(-image_scores)

    image_boxes      = image_boxes[scores_sort, :]
    image_scores     = image_scores[scores_sort]
    image_labels     = image_labels[scores_sort]

    return image_boxes, image_scores, image_labels


def _get_detections(
        generator, 
        model, 
        score_threshold=0.01, 
        max_detections=1000, 
        save_path=None,
        tiling_dim=1,
        top_k=50,
        nms_threshold=0.5,
        max_inflation_factor=100,
        nms_mode="argmax",
        wandb_logging=False,
        profile=False
):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
        tiling_dim      : Divide image into a square tile grid of this dimension
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    if profile:
        import cProfile, pstats
        pr = cProfile.Profile(builtins=False)
        pr.enable()
        print("Execution profiler is turned ON")

    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
    all_inference_times = [None for i in range(generator.size())]

    mob_iterations = 1
    if nms_mode == "enclose":
        # make sure we merge everything in this mode
        mob_iterations = 3
        nms_threshold = 0

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        # this is probably computational bottleneck, that can be alleviated with async queues
        # do not include in perf measurement
        image_name, image_ext = os.path.splitext(os.path.basename(generator.image_path(i)))
        # raw_image    = generator.load_image(i)
        raw_image = generator.load_image(i)
        # image, scale = generator.resize_image(raw_image.copy())
        image = raw_image.copy()
        start = time.perf_counter()     # them

        ############ Performance measurement per image start point ############
        image = generator.preprocess_image(image)
        image_boxes, image_scores, image_labels = run_inference_on_image(model, image, generator.resize_image,
            score_threshold=score_threshold, 
            max_detections=max_detections,
            tiling_dim=tiling_dim,
            top_k=top_k,
            nms_threshold=nms_threshold,
            nms_mode=nms_mode,
            mob_iterations=mob_iterations,
            max_inflation_factor=max_inflation_factor
        )
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        ############ Performance measurement per image end point ############
        inference_time = time.perf_counter() - start

        # if keras.backend.image_data_format() == 'channels_first':
        #     image = image.transpose((2, 0, 1))

        # # run network
        # start = time.time()
        # boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        # inference_time = time.time() - start

        # # correct boxes for image scale
        # boxes /= scale

        # # select indices which have a score above the threshold
        # indices = np.where(scores[0, :] > score_threshold)[0]

        # # select those scores
        # scores = scores[0][indices]

        # # find the order with which to sort the scores
        # scores_sort = np.argsort(-scores)[:max_detections]

        # # select detections
        # image_boxes      = boxes[0, indices[scores_sort], :]
        # image_scores     = scores[scores_sort]
        # image_labels     = labels[0, indices[scores_sort]]
        # image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            if HERIDAL_VIS_SETTINGS:
                ann_color = [255, 0, 0] # blue in BGR
                det_color = [0, 0, 255] # red in BGR
                draw_labels = False
            else:
                ann_color = None
                det_color = None
                draw_labels = True

            save_image = raw_image.copy()
            if not isinstance(generator, InferenceGenerator):
                save_image = draw_annotations(save_image, generator.load_annotations(i), color=ann_color, 
                                              label_to_name=generator.label_to_name, draw_labels=draw_labels)
            save_image = draw_detections(save_image, image_boxes, image_scores, image_labels, color=det_color, 
                                         label_to_name=generator.label_to_name, score_threshold=score_threshold,
                                         draw_labels=draw_labels)

            cv2.imwrite(os.path.join(save_path, f'{image_name}-idx-{i}{image_ext}'), save_image)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        all_inference_times[i] = inference_time

    if profile:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.sort_stats("cumulative")
        # ps.reverse_order()
        ps.print_stats(50)

    return all_detections, all_inference_times


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    top_k=50,
    nms_threshold=0.5,
    nms_mode="argmax",
    max_detections=100,
    save_path=None,
    mode="voc2012",
    tiling_dim=0,
    profile=False,    
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
        mode            : The way to perform evaluation, accepts "voc2012" (default) and "sar-apd"
    # Returns
        A dict mapping class names to mAP scores.
    """
    if mode == "sar-apd":
        # it's not meaningful to run SAR-APD mode with large iou_thres
        # using this iou threshold should prevent getting perfect score 
        # by bboxing the whole image in HERIDAL
        # but allows sufficiently large bbox groups
        # this corresponds to ~1200x1200 pixel location accuracy requirement (~20x20 m in HERIDAL) 
        iou_threshold = 0.0025
        if nms_mode != "argmax": # don't eliminate any boxes in a cluster
            top_k = -1
        
    # gather all detections and annotations
    all_detections, all_inference_times = _get_detections(generator, model, 
                                                          score_threshold=score_threshold, max_detections=max_detections, 
                                                          save_path=save_path, tiling_dim=tiling_dim, nms_threshold=nms_threshold,
                                                          nms_mode=nms_mode, top_k=top_k, profile=profile)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}

    set_name = generator.set_name

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    # iterate over classes
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        # iterate over images
        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            # detected_annotations = []
            detected_annotations = set([])
            
            # them
            all_overlaps = compute_overlap(detections, annotations)
            
            # iterate over detection candidates
            # for d in detections:
            for j, d in enumerate(detections):
                # scores = np.append(scores, d[4])
                something_detected = False

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue
                
                # overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                overlaps = all_overlaps[j]  # moi

                # them
                if mode == "voc2012":
                    start_idx = np.argmax(overlaps)
                    end_idx = start_idx + 1
                else: # SAR-APD evaluation checks all GT labels
                    start_idx = 0
                    end_idx = overlaps.shape[0]

                # assigned_annotation = np.argmax(overlaps, axis=1)
                # max_overlap         = overlaps[0, assigned_annotation]

                # if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                #     false_positives = np.append(false_positives, 0)
                #     true_positives  = np.append(true_positives, 1)
                #     detected_annotations.append(assigned_annotation)
                # else:
                #     false_positives = np.append(false_positives, 1)
                #     true_positives  = np.append(true_positives, 0)

                # iterate over ground truth labels
                for annotation in range(start_idx, end_idx):
                    if overlaps[annotation] >= iou_threshold and annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.add(annotation)
                        scores = np.append(scores, d[4])
                        something_detected = True

                if not something_detected:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    scores = np.append(scores, d[4])

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        try:
            overall_recall = recall[-1]
            overall_precision = precision[-1]
        except IndexError:
            overall_recall = 0
            overall_precision = 0
        
        if isinstance(generator, PascalVocGenerator):
            class_name = class_id_to_voc_label[int(label)]
        else:
            class_name = str(int(label))

        try:
            num_tp = true_positives[-1]
        except IndexError:
            num_tp = 0
        
        try:
            num_fp = false_positives[-1]
        except IndexError:
            num_fp = 0

        # compute average precision
        # average_precision  = _compute_ap(recall, precision)        
        average_precision  = _compute_ap(recall, precision)

        print(
            f"\n{set_name}_num_tp_{class_name}: {num_tp}\n"
            f"{set_name}_num_fp_{class_name}: {num_fp}\n" 
            f"{set_name}_num_anns_{class_name}: {num_annotations}\n"
            f"{set_name}_recall_{class_name}: {overall_recall}\n"
            f"{set_name}_precision_{class_name}: {overall_precision}\n"
            f"{set_name}_mAP_{class_name}: {average_precision}\n"
        )
        average_precisions[label] = average_precision, num_annotations

    # inference time
    mean_inference_time = np.sum(all_inference_times) / generator.size()

    return average_precisions, mean_inference_time


if __name__ == "__main__":
    ''' test something here '''

    from time import perf_counter

    bbox0 = np.array([10,20,20,30])
    bbox1 = np.array([100,200,200,300])
    bbox2 = np.array([1000,2000,2000,3000])
    bbox3 = np.array([1100,2100,2000,3000])

    bboxes0 = np.tile(bbox0, (100, 1)).astype(float) + np.random.normal(scale=2, size=(100,4))    
    bboxes1 = np.tile(bbox1, (100, 1)).astype(float) + np.random.normal(scale=10, size=(100,4))
    bboxes2 = np.tile(bbox2, (500, 1)).astype(float) + np.random.normal(scale=10, size=(500,4))
    bboxes3 = np.tile(bbox3, (200, 1)).astype(float) + np.random.normal(scale=10, size=(200,4))
    boxes = np.concatenate((bboxes0, bboxes1, bboxes2, bboxes3), axis=0) #[:, [0,2,1,3]]
    
    scores = np.random.rand(len(boxes))

    labels = np.random.randint(0, 10, len(boxes))

    merged_labels = []

    t1 = perf_counter()
    merged_boxes, merged_scores, merged_labels = merge_boxes_per_label(boxes, scores, labels)
    # merged_boxes, merged_scores = merge_overlapping_boxes(boxes, scores, iou_threshold=-100.001)

    # merged_boxes, merged_scores, merged_labels = merge_boxes_per_label(*[np.array([])]*3)
    # merged_boxes, merged_scores = merge_overlapping_boxes(np.array([]), np.array([]), iou_threshold=-100.001)

    print("Time elapsed", perf_counter()-t1)
    print("==boxes==")
    print(merged_boxes)
    print("==scores==")
    print(merged_scores)
    print("==labels==")
    print(merged_labels)
    print("==merged-length==")
    print(len(merged_boxes))