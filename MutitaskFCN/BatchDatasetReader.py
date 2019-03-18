#coding=utf-8
import numpy as np
import scipy.misc as misc
 
# 批量读取数据集的类
class BatchDatset:
    files = []
    images = []
    annotations = []
    c_gt = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    c_gt_dir = ".\\Data_zoo\\MIT_SceneParsing\\label.txt"
 
    def __init__(self, records_list, image_options={}):
        """
          Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
          sample record:
           {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
          Available options:
            resize = True/ False
            resize_size = #size of output image - does bilinear resize
            color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()
        #self._read_demo_images()
 
    def _read_images(self):
        self.__channels = True
 
        # 读取训练集图像
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        
        # 读取label的图像，由于label图像是二维的，这里需要扩展为三维
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        #读取碗盘标签
        self.c_gt = np.array([self._c_gt_transform(filename['filename']) for filename in self.files])

        print (self.images.shape)
        print (self.annotations.shape)
        print (self.c_gt.shape)

    # 把碗盘标签转为 numpy数组
    def _c_gt_transform(self, filename):
        plate = 0
        bowl = 0
        with open(self.c_gt_dir,'r') as f:
            for line in f:
                label = line.split(" ")
                if label[0] == filename:
                    bowl = int(label[1])
                    plate = int(label[2])
                    break
        return np.array([plate, bowl])

    # 把图像转为 numpy数组
    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])
 
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,[resize_size, resize_size], interp='nearest')
        else:
            resize_image = image
 
        return np.array(resize_image)
 
    def get_records(self):
        return self.images, self.annotations, self.c_gt
 
    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset
 
    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm] 
            self.c_gt = self.c_gt[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
 
        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end], self.c_gt[start:end]
		
    def next_valid_batch(self,point, batch_size):
        start = point
        end = point + batch_size
        return self.images[start:end], self.annotations[start:end], self.c_gt[start:end]		

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
