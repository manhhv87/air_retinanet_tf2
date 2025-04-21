'''
Handles parsing PASCAL VOC dataset annotations and computes statistics 
about the dataset folder, such as its hash sum, which can be used to verify
the dataset version later.

Modifications copyright (c) 2021 Accenture
'''

import os
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import keras_retinanet.utils
    __package__ = "keras_retinanet.utils"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "airutils"))

import hashlib
import numpy as np
import xml.etree.cElementTree as ET


def compute_dir_metadata(path, human_readable=True):
    ''' Credits for hashing go to: https://stackoverflow.com/a/49701019 
        
        Computes metadata from directory
        returns dir_name, sha1_hash, num_files, dir_size
    '''
    if not os.path.isdir(path):
        raise TypeError(f"'{path}' is not a directory!")
    
    dir_name = os.path.basename(path)
    digest = hashlib.sha1()
    num_files = 0
    dir_size = 0

    hash_files = []

    for root, _, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.isfile(file_path) and not name.startswith("."):
                hash_files.append((name, file_path))
                num_files += 1
                dir_size += os.path.getsize(file_path)
    
    # make sure we always hash in the same order as it affects the overall hash!
    for name, file_path in sorted(hash_files, key=lambda f: f[0]):
        # hash the file name
        digest.update(name.encode())

        # Hash the path and add to the digest to account for empty files/directories
        # This is no good as the path might change on every training run!
        # digest.update(hashlib.sha1(file_path[len(path):].encode()).digest())

        # hash file contents
        with open(file_path, 'rb') as f_obj:
            while True:
                buf = f_obj.read(1024 * 1024)
                if not buf:
                    break
                digest.update(buf)

    if human_readable:
        dir_size = f"{dir_size/1e6:.1f}M"
    else:
        dir_size = f"{dir_size}B"

    return dir_name, digest.hexdigest(), num_files, dir_size


def compute_dataset_metadata(path, human_readable=True):
    """ Wraps dir metadata in a dictionary that can be directly fed to wandb """
    dir_name, sha1_hash, num_files, dir_size = compute_dir_metadata(path, human_readable)
    return {
        "dataset_name" : dir_name,
        "dataset_checksum" : sha1_hash,
        "dataset_num_files" : num_files,
        "dataset_size" : dir_size
    }


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise ValueError('illegal value for \'{}\': {}'.format(debug_name, e))
    return result
    

def parse_voc_annotations(filename):
    def parse_annotation(element):
        """ Parse an annotation given an XML element.
        """
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name != "person":
            raise ValueError('class name \'{}\' not understood')

        box = np.zeros((4,))
        label = 14

        bndbox    = _findNode(element, 'bndbox')
        box[0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float)
        box[1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float)
        box[2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float)
        box[3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float)

        return truncated, difficult, box, label

    xml_root = ET.parse(filename).getroot()
    """ Parse all annotations under the xml_root.
    """
    num_objects = len(xml_root.findall('object'))
    annotations = {'labels': np.empty((num_objects,)), 'bboxes': np.empty((num_objects, 4))}
    for i, element in enumerate(xml_root.iter('object')):
        try:
            _, _, box, label = parse_annotation(element)
        except ValueError as e:
            raise ValueError('could not parse object #{}: {}'.format(i, e))

        annotations['bboxes'][i, :] = box
        annotations['labels'][i] = label

    return annotations