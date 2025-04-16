#!/usr/bin/env python

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
"""

import argparse
import os
import sys

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.kitti import KittiGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import evaluate
from ..utils.image import preprocess_image_caffe_fast
from ..utils.gpu import setup_gpu
from ..utils.tf_version import check_tf_version

from ..utils.dataset import compute_dataset_metadata
from ..utils import seed, optimize_tf_parallel_processing

import multiprocessing

NUM_PARALLEL_EXEC_UNITS = multiprocessing.cpu_count()


def create_generator(args, preprocess_image, set_name=None):
    """ Create generators for evaluation.
    """
    common_args = {
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'no_resize'        : args.no_resize,
        'preprocess_image' : preprocess_image,
        'shuffle_groups'   : False
    }

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017' if set_name is None else set_name,
            **common_args
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test' if set_name is None else set_name,
            image_extension=args.image_extension,
            **common_args
        )
    elif args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            'test' if set_name is None else set_name,
            **common_args
        )
    elif args.dataset_type == 'kitti':
        validation_generator = KittiGenerator(
            args.kitti_path,
            'val' if set_name is None else set_name,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for the AIR detector.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--image-extension',   help='Declares the dataset images\' extension.', default='.jpg')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    kitti_parser=subparsers.add_parser('kitti')
    kitti_parser.add_argument('--kitti_path', help='Path to dataset directory') 
    
    def csv_list(string):
        return string.split(',')

    def bool_str(string):
        return string.lower() in {"true", "yes", "y", "on"}
    
    parser.add_argument('--model',            help='Path to RetinaNet model.', required=True)
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', type=bool_str, default=False)
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', type=int)
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--iou-threshold',    help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--nms_threshold',    help='NMS Threshold for two overlapping detections (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--nms_mode',         help='How to merge two overlapping detections in NMS (defaults to "argmax").', default="argmax")
    parser.add_argument('--top_k',            help='Number of top scoring bboxes to keep in merge cluster when nms_mode is not "argmax"', type=int, default=-1)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--anchor_scale',     help='Scale the anchor boxes by this constant (e.g. if your objects are very small)', type=float, default=1.0)
    parser.add_argument('--set_name',         help='Which partition of the dataset to evaluate on? (default: "test" or "val2017")')
    parser.add_argument('--eval_mode',        help='The way to perform evaluation, accepts "voc2012" (default) and "sar-apd"', 
                                              default="voc2012", const="voc2012", nargs="?", choices=("sar-apd", "voc2012"))
    parser.add_argument('--profile',          help="ADVANCED: Enable execution profiling to find bottlenecks in the program performance", 
                                              default=False, action="store_true")
    
    # Misc
    parser.add_argument("--seed", help="Fix various random seeds for more reproducible results.", type=int)
    
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.backbone == "infer":
        import re
        base_name = os.path.basename(args.model)
        try:
            args.backbone = re.search(r"\_([A-Za-z]*[^\-]+[A-Za-z]+\d+?)\_", base_name).group(1)
        except AttributeError:
            raise ValueError("Cannot infer backbone from model name: " + base_name)

    if "pascal_path" in vars(args):
        path_key = "pascal_path"
    else:
        path_key = "coco_path"

    arg_dict = vars(args)

    if "SLURM_JOB_ID" in os.environ:
        print("Slurm Job ID is", os.environ["SLURM_JOB_ID"])
        arg_dict[path_key] = arg_dict[path_key].replace("UNIQUE_JOB_FOLDER", os.environ["SLURM_JOB_ID"])
    else:
        arg_dict[path_key] = arg_dict[path_key].replace("UNIQUE_JOB_FOLDER", "sar-uav-cv")

    # optionally seed random number generators
    if args.seed:
        seed(args.seed)    

    # make sure tensorflow is the minimum required version
    check_tf_version()

    # optionally choose specific GPU
    if args.gpu:
        setup_gpu(args.gpu)

    # optimize parallel thread usage on CPU and configure GPU
    optimize_tf_parallel_processing(NUM_PARALLEL_EXEC_UNITS, gpu_id=args.gpu)

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    backbone = models.backbone(args.backbone)
    generator = create_generator(args, backbone.preprocess_image, args.set_name)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)
        anchor_params.scales *= args.anchor_scale
    
    # load the model
    print('Loading model, this may take a second...')
    # using compile=False might fix some model loading problems (especially with seresnext101 backbone)
    # but inference performance may drop
    model = models.load_model(args.model, backbone_name=args.backbone, compile=True)
    generator.compute_shapes = make_shapes_callback(model)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(model, 
                                     nms=args.nms_mode == "argmax",
                                     anchor_params=anchor_params,                                      
                                     nms_threshold=args.nms_threshold,
                                     max_detections=args.max_detections     # // 4 if args.image_tiling_dim else args.max_detections
                                    )

    # print model summary
    # print(model.summary())

    # start evaluation
    if args.dataset_type == 'coco':
        from ..utils.coco_eval import evaluate_coco
        evaluate_coco(generator, model, args.score_threshold)
    else:
        average_precisions, inference_time = evaluate(
            generator,
            model,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            top_k=args.top_k,
            nms_threshold=args.nms_threshold,
            nms_mode=args.nms_mode,
            max_detections=args.max_detections,
            save_path=args.save_path,
            mode=args.eval_mode,
            tiling_dim=args.image_tiling_dim,
            profile=args.profile
        )

        # print evaluation
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            print('{:.0f} instances of class'.format(num_annotations),
                  generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        # print('Inference time for {:.0f} images: {:.4f}'.format(generator.size(), inference_time))
        print(f'Inferenced {generator.size():.0f} images')
        print(f'Average inference time per image: {inference_time:.4f} s')
        
        print('mAP using the weighted average of precisions among classes: {:.4f}'.format(sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
        print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))


if __name__ == '__main__':
    main()
