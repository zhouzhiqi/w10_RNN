
# coding: utf-8

# In[38]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from tempfile import gettempdir

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import json

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

try: 
    from tinyenv.flags import flags
except ImportError:
    # 若在本地运行，则自动生成相同的class
    class flags(object):
        def __init__(self):
            self.iterations=100001
            self.batch_size = 128
            self.output_dir = os.path.join(os.getcwd(), 'output')
            self.data_dir = '../'

#实例化class
FLAGS = flags()


# In[54]:


data_dir = FLAGS.data_dir
print('data_dir is : ',data_dir)
output_dir = FLAGS.output_dir
file_name = 'QuanSongCi.txt'
file_path = os.path.join(data_dir, file_name)
with open(file_path, 'r', encoding='UTF-8') as f:
    vocabulary = list(f.read())


# In[55]:


vocabulary[:10]


# In[56]:



# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 5000

# 建立数据集, words是所有单词列表, n_words是想建的字典中的单词的个数
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]  #将所有低频词设为'UNK', 个数为-1
  count.extend(collections.Counter(words).most_common(n_words - 1))
  #将words集合中的单词按频数排序，将频率最高的前
  #n_words-1个单词以及他们的出现的个数按顺序输出到count中，
  #将频数排在n_words-1之后的单词设为UNK。
  #同时，count的规律为索引越小，单词出现的频率越高
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
    #对count中所有单词进行编号，由0开始，保存在字典dict中
  
  data = list()
  unk_count = 0
   #对原words列表中的单词使用字典中的ID进行编号，
   #即将单词转换成整数，储存在data列表中
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1 # 对UNK进行计数
    data.append(index)
  count[0][1] = unk_count #'UNK'个数
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  #将dictionary中的数据反转，即可以通过ID找到对应的单词
  return data, count, dictionary, reversed_dictionary


# In[57]:



# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.

print('Most common words (+UNK)', count[:5]) 
# 输出频数最高的前5个单词
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


# In[58]:


# 保存dictionary和reverse_dictionary为json
with open(os.path.join(output_dir,'reverse_dictionary.json'),'w') as f:
    json.dump(reverse_dictionary, f)
with open(os.path.join(output_dir,'dictionary.json'),'w') as f:
    json.dump(dictionary, f)


# In[59]:



data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
# 生成batch, 对data中的单词,分别与其前一个和后一个单词生成batch
# batch内容为: [data[1], data[0]]
def generate_batch(batch_size, num_skips, skip_window):
  global data_index  # 全局索引
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  #窗的大小为3，结构为 [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  #建立一个结构为双向队列的缓冲区，大小不超过3
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  #将数据index到index+3的字段赋值给buffer，大小刚好为span
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window] # 生成batch
      labels[i * num_skips + j, 0] = buffer[context_word]  # 生成对应labels
    if data_index == len(data): #是否遍历完data
      buffer = data[:span]
      data_index = span
      # 重新开始, 将data前span位传入buffer, 重置data_index
    else:
      buffer.append(data[data_index])
      data_index += 1
      # 没有遍历结束, 在buffer尾部加入一个单词并挤出buffer中最前单词
      # 相当于span后移一位
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  # 重置data_index, 为避免身后移动apan时超出data
  return batch, labels


# In[60]:



batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


# In[61]:



# Step 4: Build and train a skip-gram model.

batch_size = FLAGS.batch_size # 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 64      # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)



# In[62]:



graph = tf.Graph()

with graph.as_default():

  # Input data.
  # 输入一个batch的训练数据，是当前单词在字典中的索引id
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
   # 输入一个batch的训练数据的标签，是当前单词前一个或者后一个单词在字典中的索引id
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
   #从字典前100个单词，即频率最高的前100个单词中，随机选出16个单词，将它们的id储存起来，作为验证集
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    # 初始化字典中每个单词的embeddings，值为-1到1的均匀分布
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    #找到训练数据对应的embeddings
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    #初始化训练参数
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # 初始化偏置

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

  '''
  算法非常简单，根据词频或者类似词频的概率选出64个负采样v，联同正确的输入w（都是词的id），用它们在nce_weights对应的向量组成一个训练子集mu。
  对于训练子集中各个元素mu(i)，如果是w或者m(i)==w(w这里是输入对应的embedding)，loss(i)=log(sigmoid(w*mu(i)))
  如果是负采样，则loss(i)=log(1-sigmoid(w*mu(i)))
  然后将所有loss加起来作为总的loss，loss越小越相似（余弦定理）
  用总的loss对各个参数求导数，来更新nce_weight以及输入的embedding
  '''

  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  #对embedding进行归一化
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  #找到验证集中的id对应的embedding
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  #判断验证集和整个归一化的embedding的相似性
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()


# In[ ]:



# Step 5: Begin training.
num_steps = FLAGS.iterations

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:  #求移动平均loss
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:  #评估一下验证集和整个embeddings的相似性
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()


# In[50]:


# 保存最终的 word embedding
np.save(os.path.join(output_dir,'embedding.npy'), final_embeddings)


# In[51]:



# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)


# In[52]:



try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(output_dir, 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)

