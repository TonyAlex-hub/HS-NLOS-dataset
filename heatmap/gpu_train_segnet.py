# coding=utf-8
'''''
19/07/03 lpl 15:33
FCN模型，输入数据为24*24*3的遥感影像片，输出为24*24*2的标签地图
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import tensorflow as tf
import numpy as np
import os
import time

sys.path.append("/data1/lpl/merge_roads")
sys.path.append("/data1/lpl/merge_roads/standard")
# import lpl_prepareData_1C as lpl_prepareData
from standard import lpl_prepareData
from standard import lpl_logger
from standard import lpl_tensorboard
from standard import lpl_accuracy as acc
import socket

import segnet as NN

# 创建一个日志记录器s
host_name = socket.gethostname()
acc_logger = lpl_logger.logger('/data1/cmy/logs/rich_road_segnet_1e-7_crossentropy/', 'acc')


# 定义参数
def Create_args():
    parser = argparse.ArgumentParser()
    # Training settings
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--sat_size', type=int, default=128)
    parser.add_argument('--sat_channel', type=int, default=3)
    parser.add_argument('--map_size', type=int, default=128)
    parser.add_argument('--map_channel', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--gpu_index', type=str, default="0")
    parser.add_argument('--gpu_rate', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=2e-7)  # 设置学习率
    parser.add_argument('--lr_decay_rate', type=float, default=0.9)  # 设置衰减率
    parser.add_argument('--decay_steps', type=float, default=100000)
    parser.add_argument('--momentum', type=float, default=0.9)#动量
    parser.add_argument('--image_db', type=str, default=
    '/data1/datasets/roads/train/rich_road_1500_1500_128_128_697048s/lmdb/sat')
    parser.add_argument('--label_db', type=str, default=
    '/data1/datasets/roads/train/rich_road_1500_1500_128_128_697048s/lmdb/map')
    parser.add_argument('--test_image_db', type=str, default=
    '/data1/datasets/roads/valid/rich_road_1500_1500_128_128_697048s_double/lmdb/sat')
    parser.add_argument('--test_label_db', type=str, default=
    '/data1/datasets/roads/valid/rich_road_1500_1500_128_128_697048s_double/lmdb/map')
    parser.add_argument('--image_save_path', type=str, default='./' + host_name + '/image_save/')
    parser.add_argument('--save_path', type=str, default='/data1/cmy/save/rich_road_segnet_1e-7_crossentropy/')
    # parser.add_argument('--model_path', type=str, default=
    # '/data1/cmy/best_save/fcn_6000-201912312036.meta')
    args = parser.parse_args()
    return args


# 参数实例化
args = Create_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index


# 计算tower的损失值
def pixel_ratio(label):
    shape = label.shape
    count_road = np.sum(label[:, :, :, 1])
    ratio_road = np.float(count_road) / (shape[0] * shape[1] * shape[2])
    ratio_bg = 1 - ratio_road
    return ratio_road, ratio_bg


# 计算损失值
def generate_loss(prob, target_tensor, w_ro, w_bg, regu=0):
    res0 = -tf.reduce_sum(tf.multiply(target_tensor[:, :, :, 0], tf.log(prob[:, :, :, 0])))#同样位置的元素相乘，不同的反而为零，相同的为1,loss值不是反而会变大吗？
    res1 = -tf.reduce_sum(tf.multiply(target_tensor[:, :, :, 1], tf.log(prob[:, :, :, 1])))
    res = tf.reduce_sum(w_bg * res0 + w_ro * res1)
    lpl_tensorboard.variable_summaries(prob[:, :, :, 1])

    # res1=-tf.reduce_sum(tf.multiply(target_tensor,tf.log(tf.add(1e-7,prob))))
    # res2=-tf.reduce_sum(tf.multiply(tf.subtract(1.0,target_tensor),tf.log(tf.subtract(1+1e-7,prob))))
    # res = tf.add(tf.multiply(w_ro,res1),tf.multiply(w_bg,res2))
    #res = tf.reduce_sum(tf.square(prob - target_tensor))
    return res

def train(loss, lr):
    tf.summary.scalar('loss_value', loss)
    lr_decay = tf.train.exponential_decay(lr, global_step, args.decay_steps, args.lr_decay_rate, staircase=False)#1.首先使用较大学习率(目的：为快速得到一个比较优的解);
    # 2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);
    # 计算梯度
    #optimizer = tf.train.MomentumOptimizer(learning_rate=lr_decay, momentum=args.momentum, name='optimizer_mom')#这是动量优化法还有梯度下降优化法
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_decay)
    grads = optimizer.compute_gradients(loss)#计算loss的导数
    apply_gradient_op = optimizer.apply_gradients(grads, name='apply_grads')
    lpl_tensorboard.gradient_summaries(grads)
    # 更新梯度
    return apply_gradient_op, lr


# 使用测试集验证模型精度,返回模型输出
def test_model(test_data, probability):
    probability_seg = []
    for i in range(0, len(test_data)):
        test_data_sub = test_data[i]
        feed_dict = {images: test_data_sub, train_mode: False}
        # 模型预测
        probability_val = sess.run(probability, feed_dict=feed_dict)
        probability_seg.append(probability_val)
    return probability_seg


def printf_acc(iter_times, probability_seg, test_name_list, test_label, path=None):
    print('printf acc...')
    pre_sum = 0;
    rec_sum = 0;
    f1_sum = 0
    probability_seg = flat_var(probability_seg)
    pre_labels, names = lpl_prepareData.merge_map_segmentation(probability_seg, test_name_list)
    test_label = flat_var(test_label)
    num = len(test_label)
    labels, names = lpl_prepareData.merge_map_segmentation(test_label, test_name_list)
    for i in range(0, len(pre_labels)):
        # for i in range(0, len(pre_labels)):
        pre_label = pre_labels[i];
        label = labels[i]
        pre, rec = acc.acc_2D(pre_label, label)
        f1 = acc.acc_f1(pre, rec)
        if path != None:
            lpl_prepareData.save_image(pre_label, path + names[i] + '.tif')
        pre_sum = pre_sum + pre;
        rec_sum = rec_sum + rec;
        f1_sum = f1_sum + f1
        print(pre, rec, f1)
        acc_logger.output([iter_times, pre, rec, f1])
        #print(pre_sum/num,rec_sum/num,f1_sum/num)
    #acc_logger.output([iter_times, pre_sum/num,rec_sum/num,f1_sum/num])



'''加载数据，并将标签转化为标准标签'''


def load_data():
    batchs_data, batchs_label, key = lpl_prepareData.GetData(args, args.image_db, args.label_db)#将lmdb中的文件转为tensor，key为数据库中每个条目的索引
    test_data, test_label, test_name_list = lpl_prepareData.GetData(args, args.test_image_db, args.test_label_db)
    test_name_list_flat = []
    # 对影像快名字进行flat
    for i in range(len(test_name_list)):
        for j in range(len(test_name_list[i])):
            test_name_list_flat.append(test_name_list[i][j])
    # print('the length of test_label is ',len(test_label),test_label[2].shape,type(test_label[2]))
    return batchs_data, batchs_label, test_data, test_label, test_name_list_flat


def flat_var(var):
    flat = []
    for i in range(len(var)):
        for j in range(len(var[i])):
            flat.append(var[i][j])
    return flat


if __name__ == '__main__':
    # 创建一个Session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # sess=tf.Session();sess.as_default()
        # 输入变量
        images = tf.placeholder(tf.float32, [None, args.sat_size, args.sat_size, args.sat_channel],
                                name='images')
        # 输出变量
        true_out = tf.placeholder(tf.float32, [args.batchsize, args.map_size, args.map_size, args.map_channel],
                                  name='true_out')
        keep_prob_pl = tf.placeholder(tf.float32, shape=None, name="keep_rate")
        lr = tf.placeholder(tf.float32, name='lr')
        w_ro = tf.placeholder(tf.float32)
        w_bg = tf.placeholder(tf.float32)
        # 模式
        train_mode = tf.placeholder(tf.bool)
        global_step = tf.placeholder(tf.float32)
        ####
        # 实现网络实例
        nn = NN.SegNet()
        nn.build(images)
        #loss = generate_loss(nn.output, true_out, w_ro, w_bg)
        loss = improved_cross_entropy(true_out,nn.output)
        optimizer, lr_decay = train(loss=loss, lr=lr)
        # loss,optimizer,fcn_pro,lr= train(fcn=fcn,input_tensor=images,out_tensor=true_out,sess=sess,train_mode=train_mode)
        # 初始化所有变量
        TIMESTAMP = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        merged = tf.summary.merge_all()  #
        writer = tf.summary.FileWriter('./' + host_name + '/tensorboard/' + TIMESTAMP, sess.graph)  # 将训练日志写入到logs文件夹下
        sess.run(tf.global_variables_initializer())
        # 准备数据
        print('load data...')
        batchs_data, batchs_label, test_data, test_label, test_name_list_flat = load_data()
        # 道路宽度类实例化
        # lw=LW.LW(np.reshape(batchs_label[0],[256,256]),np.reshape(batchs_label[0],[256,256]),np.reshape(batchs_label[0],[256,256]))
        # 开始训练
        print('Start training...')
        tstart = time.time()
        iter_times = 1
        probability_seg = test_model(test_data, nn.output)
        printf_acc(iter_times, probability_seg, test_name_list_flat, test_label)

        for epoch in range(0, 20000):
            start = time.time()
            index = np.arange(len(batchs_data))
            np.random.shuffle(index)
            for i in index:
                train_data = batchs_data[i]
                train_label = batchs_label[i]
                feed_dict = {images: train_data, train_mode: True}
                prob = sess.run(nn.output, feed_dict=feed_dict)
                ratio_ro, ratio_bg = pixel_ratio(train_label)
                if ratio_ro == 0:
                    ratio_ro = 1;
                    ratio_bg = 1
                print(ratio_ro, ratio_bg)
                feed_dict = {images: train_data, true_out: train_label, global_step: iter_times,
                             w_ro: 1*ratio_bg, w_bg: 1*ratio_ro, lr: args.lr}
                _, loss_regu_val, loss_val, rs, lr_val = sess.run([optimizer, loss, loss, merged, lr_decay],
                                                                  feed_dict=feed_dict)
                # duration = (time.time() - start) * 1000 / 1000
                # print('单节点单卡时: %.2fms' % (duration))
                print('epoch=%d,i=%d of %d, loss_regu_val=%f,loss=%f,lr=%f' % (
                epoch, iter_times, len(batchs_data), loss_regu_val, loss_val, lr_val))#输出的六个数
                #writer.add_summary(rs, iter_times)
                iter_times = iter_times + 1
                if iter_times % 5000 == 0:
                    probability_seg = test_model(test_data, nn.output)
                    printf_acc(iter_times, probability_seg, test_name_list_flat, test_label)
                    model_save_path = args.save_path + '/segnet_' + str(iter_times) + '-' + \
                                      time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
                    nn.save_npy(sess, model_save_path)
                    # duration = (time.time() - start) * 1000 / 1000
                    # print('单节点单卡时: %.2fms' % (duration))
            telapse = time.time() - tstart
            print('单节点单卡时: %.2fs' % (telapse))
