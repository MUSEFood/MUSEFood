#coding=utf-8
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
 
import TensorflowUtils as utils
 
# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
 
 
def read_dataset(data_dir,data_name):
    pickle_filename = "MITSceneParsing.pickle"    
    pickle_filepath = os.path.join(data_dir, pickle_filename)
 
    if not os.path.exists(pickle_filepath): 
          
        result = create_image_lists(os.path.join(data_dir, data_name))
 
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")
 
    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result
 
    return training_records, validation_records
 
'''
  返回一个字典:
  image_list{ 
           "training":[{'image': image_full_name, 'annotation': annotation_file, 'image_filename': },......],
           "validation":[{'image': image_full_name, 'annotation': annotation_file, 'filename': filename},......]
           }
'''
def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}
 
    for directory in directories:
        file_list = []
        image_list[directory] = []
 
        # 获取images目录下所有的图片名
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'png')
        file_list.extend(glob.glob(file_glob))             
 
        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                # 注意注意，下面的分割符号，在window上为：\\,在Linux撒花姑娘为 : /
                filename = os.path.splitext(f.split("\\")[-1])[0]  # 图片名前缀
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
 
        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))
 
    return image_list

'''
  返回一个字典:
  image_list{ 
           "training":[{'image': image_full_name, 'annotation': annotation_file, 'image_filename': },......],
           "validation":[{'image': image_full_name, 'annotation': annotation_file, 'filename': filename},......]
           }
'''
def create_demo_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory " + image_dir + " not found.")
        return None
 
    file_list = []
    image_list = []

    # 获取demos目录下所有的图片名
    file_glob = os.path.join(image_dir, "demos", '*.' + 'png')
    file_list.extend(glob.glob(file_glob))             

    if not file_list:
        print('No files found')
    else:
        for f in file_list:
            # 注意注意，下面的分割符号，在window上为：\\,在Linux撒花姑娘为 : /
            filename = os.path.splitext(f.split("\\")[-1])[0]  # 图片名前缀
            record = {'image': f, 'filename': filename}
            image_list.append(record)
    return image_list
