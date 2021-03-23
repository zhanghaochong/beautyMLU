# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
os.environ['MLU_VISIBLE_DEVICES']='0'
os.environ['MLU_QUANT_PARAM']='ganquant.txt'
os.environ['MLU_RUNNING_MODE']='1'
os.environ['MLU_STATIC_NODE_FUSION']='true'
os.environ['MLU_OP_PRECISION']='float32'


parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup', 'xfsy_0068.png'), help='path to the no_makeup image')
parser.add_argument('--output_nodes', type=str, default="generator/xs:0", help='something like this: generator/xs:0,generator/concat:0')
args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2
    #return img

def deprocess(img):
    return (img + 1) / 2

batch_size = 1
img_size = 256
no_makeup = cv2.resize(imread(args.no_makeup), (img_size, img_size))
#X_img = np.expand_dims(preprocess(no_makeup), 0)
X_img = np.expand_dims(no_makeup, 0)
makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
result[img_size: 2 *  img_size, :img_size] = no_makeup / 255.

input_node="X:0,generator/Relu_5:0"
output_node="generator/xs:0"
output_node_name=["generator/xs"]

relu5=np.load("relu5.npy")
#relu5_1=np.transpose(relu5, (0, 2 ,3 ,1))
print(relu5.shape)
np.savetxt("relu5.nptxt",relu5.flatten(), fmt='%.12f')
#config = tf.ConfigProto(allow_soft_placement=True)
#config = tf.ConfigProto()
#config.mlu_options.core_num = 1
#config.mlu_options.core_version = 'MLU270'

config = tf.ConfigProto(allow_soft_placement=True)
config.mlu_options.core_num = 16
config.mlu_options.core_version = 'MLU270'
config.mlu_options.offline_model_name = "beautygan.cambricon"
config.mlu_options.save_offline_model = True

new_ph = tf.placeholder(dtype=tf.float32, shape=relu5.shape)

session=tf.Session(config=config)
saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'), input_map = {"generator/Relu_5:0":new_ph})
saver.restore(session, tf.train.latest_checkpoint('model'))
#output_graph = tf.graph_util.convert_variables_to_constants(session,
#                                                   tf.get_default_graph().as_graph_def(), output_node_name)
#output_graph = graph_util.extract_sub_graph(output_graph, output_node_name)


input = [tf.get_default_graph().get_tensor_by_name(node) for node in input_node.split(",")]
output = [tf.get_default_graph().get_tensor_by_name(node) for node in output_node.split(",")]


for i in range(len(makeups)):
    #makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
    #Y_img = np.expand_dims(preprocess(makeup), 0)
    #makeup = np.zeros(makeup.shape)
    #Y_img = np.expand_dims(makeup, 0)
    
    #out = session.run(output, feed_dict={input[0]:X_img,input[1]:Y_img})
    #out = session.run(output, feed_dict={input[0]:X_img,input[1]:relu5})
    out = session.run(output, feed_dict={input[0]:X_img,new_ph:relu5})
    for k in range(len(output)):
        print(output[k].name.replace('/',''))
        with open(output[k].name.replace('/','')+"_mlu.txt","w+") as f:
            for j in out[k].reshape(-1):
                f.write("%.5f\n" % j)
    #np.save("relu5", out[1]) 
    out = deprocess(out[0])
    #result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
    result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = out[0]

imsave('result.jpg', result)

