from __future__ import print_function
import tensorflow as tf
import numpy as np
 
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatasetReader as dataset
from six.moves import xrange
 
batch_size=32                            # batch 大小
logs_dir="logs/"
data_dir= "Data_zoo/MIT_SceneParsing/"   # 存放数据集的路径，需要提前下载
#data_name="ADEChallengeData2016"
data_name="FoodData"
learning_rate=1e-5                                           # 学习率
model_path="Model_zoo/imagenet-vgg-verydeep-19.mat"          # VGG网络参数文件，需要提前下载
debug= False


MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat' #训练好的VGGNet参数
 
MAX_ITERATION = int(5e4 + 1)   # 最大迭代次数
NUM_OF_CLASSESS = 2          # 类的个数
IMAGE_SIZE = 224               # 图像尺寸
 
# 根据载入的权重建立原始的 VGGNet 的网络
def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3','relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3','relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3','relu5_3', 'conv5_4', 'relu5_4'
    )
 
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
            print ("当前形状：",np.shape(current))
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
            print ("当前形状：",np.shape(current))
        net[name] = current
 
    return net
 
 
# FCN的网络结构定义，网络中用到的参数是迁移VGG训练好的参数
def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    # 加载模型数据    
    print ("原始图像：",np.shape(image))
    model_data = utils.get_model_data(model_path)
 
    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
 
    weights = np.squeeze(model_data['layers'])
 
    # 图像预处理
    processed_image = utils.process_image(image, mean_pixel)
    print ("预处理后的图像:",np.shape(processed_image))
 
    with tf.variable_scope("inference"):
        # 建立原始的VGGNet-19网络
 
        print ("开始建立VGG网络：")
        image_net = vgg_net(weights, processed_image)
 
        # 在VGGNet-19之后添加 一个池化层和三个卷积层
        conv_final_layer = image_net["conv5_3"]
        print ("VGG处理后的图像：",np.shape(conv_final_layer))

        #盘子、碗分类
        process_data = tf.layers.flatten(conv_final_layer)
        print("process_data", np.shape(process_data))
        fc_category1 = tf.layers.dense(process_data, 128, activation='relu')
        print("fc_category1", np.shape(fc_category1))
        fc_output = tf.layers.dense(fc_category1, 2, activation='softmax')
        print("fc_output", np.shape(fc_output))
		
        pool5 = utils.max_pool_2x2(conv_final_layer)
 
        print ("pool5：",np.shape(pool5))
 
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
 
        print ("conv6:",np.shape(relu_dropout6))
 
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
 
        print ("conv7:",np.shape(relu_dropout7))
 
        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
 
        print ("conv8:",np.shape(conv8))
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
 
        # 对卷积后的结果进行反卷积操作
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
 
        print ("pool4 and de_conv8 ==> fuse1:",np.shape(fuse_1)) # (14, 14, 512)
 
        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2") 
 
        print ("pool3 and deconv_fuse1 ==> fuse2:",np.shape(fuse_2)) # (28, 28, 256)
 
        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
 
        print ("conv_t3:",[np.shape(image)[1],np.shape(image)[2],NUM_OF_CLASSESS]) # (224,224,2)
 
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")  # (224,224,1)
        
 
    return tf.expand_dims(annotation_pred, dim=3), conv_t3, mean_pixel, fc_output
 
 
# 返回优化器
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)
 
# 主函数,返回优化器的操作步骤
def main(argv=None):
   
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    c_gt = tf.placeholder(tf.int32, shape=[None, 2], name="c_groundtruth")
 
    print("setting up vgg initialized conv layers ...")
 
    # 定义好FCN的网络模型
    pred_annotation, logits, mean_pixel, c_pred = inference(image, keep_probability)
	
    # 定义损失函数，这里使用交叉熵的平均值作为损失函数
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                         name="entropy"))) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_pred, labels=c_gt, name='c_entropy'))
    print("loss", np.shape(loss))
	# 定义优化器 
    trainable_var = tf.trainable_variables()
 
    if debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)
 
    # 加载数据集
    print("Setting up image reader...")
 
    train_records, valid_records = scene_parsing.read_dataset(data_dir,data_name)
 
    print("训练集的大小:",len(train_records))
    print("验证集的大小:",len(valid_records))
 
    print("Setting up dataset reader")
 
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
 
    min_loss = 10000
    ep = 0
    # 开始训练模型
    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()
 
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
 
    for itr in xrange(MAX_ITERATION):
        train_images, train_annotations,train_c_gt = train_dataset_reader.next_batch(batch_size)
        #print (np.shape(train_images),np.shape(train_annotations),np.shape(train_c_gt))
        feed_dict = {image: train_images, annotation: train_annotations, c_gt: train_c_gt, keep_probability: 0.85}
        sess.run(train_op, feed_dict=feed_dict)
        #print ("step:",itr)

        if itr % 10 == 0:
            train_loss = sess.run(loss, feed_dict=feed_dict)
            print ("step:",itr)				
            print("Step: %d, Train_loss:%g" % (itr, train_loss))
            print("Best Step: %d, Valid_loss:%g" % (ep, min_loss))
			

        if itr % 200 == 0:
            valid_images, valid_annotations, valid_c_gt = validation_dataset_reader.next_batch(batch_size)
            valid_loss= sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations, c_gt: valid_c_gt,
                                                   keep_probability: 1.0})
            print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
            if min_loss > valid_loss:
                min_loss = valid_loss
                ep = itr					
                saver.save(sess, logs_dir + "model.ckpt", itr)

if __name__ == "__main__":
    tf.app.run()
