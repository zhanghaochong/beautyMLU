# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
os.environ['MLU_VISIBLE_DEVICES']=''
os.environ['MLU_QUANT_PARAM']='ganquant.txt'
os.environ['MLU_RUNNING_MODE']='0'

parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup', 'xfsy_0068.png'), help='path to the no_makeup image')
parser.add_argument('--output_nodes', type=str, default="generator/xs:0", help='something like this: generator/xs:0,generator/concat:0')
args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

batch_size = 1
img_size = 256
no_makeup = cv2.resize(imread(args.no_makeup), (img_size, img_size))
X_img = np.expand_dims(preprocess(no_makeup), 0)
makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
result[img_size: 2 *  img_size, :img_size] = no_makeup / 255.

#config = tf.ConfigProto(allow_soft_placement=True)
#config.mlu_options.core_num = 16
#config.mlu_options.core_version = 'MLU270'
#with tf.Session(config=config) as session:
tf.reset_default_graph()
#with tf.Session() as session:
session = tf.Session()
saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))

saver.restore(session, tf.train.latest_checkpoint('model'))
input_node="X:0,generator/Relu_5:0"
output_node=args.output_nodes  #"generator/xs:0,generator/concat0"
output_node_name=["generator/xs"]

input = [tf.get_default_graph().get_tensor_by_name(node) for node in input_node.split(",")]
output = [tf.get_default_graph().get_tensor_by_name(node) for node in output_node.split(",")]

#output_name = [tf.get_default_graph().get_tensor_by_name(node) for node in output_node_name.split(",")]
output_graph = tf.graph_util.convert_variables_to_constants(session,
                                                       tf.get_default_graph().as_graph_def(), output_node_name)
output_graph = graph_util.extract_sub_graph(output_graph, output_node_name)
with tf.gfile.GFile("model.pb", "wb") as f:
    f.write(output_graph.SerializeToString())

tf.io.write_graph(
    output_graph, "./", "model.pbtxt", as_text=True
)


for i in range(len(makeups)):
    #makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
    #Y_img = np.expand_dims(preprocess(makeup), 0)
    relu5=np.load("relu5.npy")         

    #out = session.run(output, feed_dict={input[0]:X_img,input[1]:Y_img})
    out = session.run(output, feed_dict={input[0]:X_img,input[1]:relu5})
    #out = deprocess(out)
    for k in range(len(output)):
        print(output[k].name.replace('/',''))
        with open(output[k].name.replace('/','')+"_cpu.txt","w+") as f:
            for j in out[k].reshape(-1):
                f.write("%.5f\n" % j)
    #result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
    #result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = out[0]
session.close()
#imsave('result.jpg', result)
#graph = tf.get_default_graph()
#print(graph.get_operations())
#X = graph.get_tensor_by_name('X:0')
#Y = graph.get_tensor_by_name('Y:0')
#Xs = graph.get_tensor_by_name('generator/xs:0')

#Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    
#imsave('result.jpg', result)
