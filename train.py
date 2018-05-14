#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import tensorflow as tf

import utils
from model import Model
from utils import read_data

from flags import parse_args
FLAGS, unparsed = parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

# 读入'QuanSongCi.txt',并转换为list
vocabulary = read_data(FLAGS.text)
print('Data size', len(vocabulary))

# 读入预处理好的dictionary和reverse_dictionary
# dictionary: key=word, value=ID
# reverse_dictionary: key=ID, value=word
with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')


model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
model.build()


with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        # 查看output_dir中是否有checkpoint, 
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        # 若有就恢复
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        # 若没有就输出未找到checkpoint
        logging.debug('no check point found....')

    for x in range(1):  #运行一个epoch
        logging.debug('epoch [{0}]....'.format(x))
        state = sess.run(model.state_tensor)
        # 使用生成器get_train_data, 逐步生成所需数据
        for dl in utils.get_train_data(vocabulary, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):

            ##################
            # Your Code here
            ##################
            
            feed_dict = {
            model.X : utils.index_data(dl[0], dictionary),
            model.Y : utils.index_data(dl[1], dictionary),
            model.keep_prob : 0.5
            }
            #-----------------------------------------------------------------
            gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.outputs_state_tensor, model.loss, model.merged_summary_op], feed_dict=feed_dict)
            summary_string_writer.add_summary(summary_string, gs)

            if gs % 10 == 0:  #每10个step输出loss, 并保存该step下相关信息
                logging.debug('step [{0}] loss [{1}]'.format(gs, l))
                save_path = saver.save(sess, os.path.join(
                    FLAGS.output_dir, "model.ckpt"), global_step=gs)
    summary_string_writer.close()
