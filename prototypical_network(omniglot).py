from google.colab import drive
drive.mount ('/gdrive')
%cd /gdrive

%tensorflow_version 1.x

import os
import glob
from PIL import Image

import numpy as np
import tensorflow as tf

print(tf.__version__)

image_name='/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data/images/Japanese_(katakana)/character13/0608_04.png'
image=Image.open(image_name)
print ("image.format=%s,image.size=%s,image.mode=%s" % (image.format,image.size,image.mode))
a=np.array(Image.open(image_name))
print(a.shape) # (105, 105)
print(a.dtype) # bool
print(a[40]) # 有字的點為黑色,也就是False,其餘為白色,也就是True
a=(np.array(a, np.float32,copy=True)) 
print(a[40]) # True轉為1.,False轉為0.
alphabet, character, rotation = 'Sanskrit/character13/rot000'.split('/')
print("alphabet=%s,character=%s,rotation=%s" % (alphabet,character,rotation))
rotation = float(rotation[3:])
print("rotation=%f" % (rotation))
#Image.open('data/images/Japanese_(katakana)/character13/0608_01.png')

image_name='/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data/images/Japanese_(katakana)/character13/0608_04.png'
Image.open(image_name)

image_name='/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data/images/Japanese_(katakana)/character13/0608_02.png'
Image.open(image_name)

Image.open('/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data/images/Sanskrit/character13/0863_09.png')

Image.open('/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data/images/Sanskrit/character13/0863_13.png')

alphabet, character, rotation = 'Sanskrit/character13/rot000'.split('/')
rotation = float(rotation[3:])
print(alphabet, character,rotation)
image_name = '/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data/images/Sanskrit/character13/0863_13.png'
Image.open(image_name)
alphabet, character, rotation = 'Sanskrit/character13/rot090'.split('/')
rotation = float(rotation[3:])
print(alphabet, character,rotation)
Image.open(image_name).rotate(rotation)

image_name = '/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data/images/Sanskrit/character13/0863_13.png'
image=Image.open(image_name)
print ("image.format=%s,image.size=%s,image.mode=%s" % (image.format,image.size,image.mode))
a=(np.array(Image.open(image_name), np.float32,copy=True))
print(a.shape)
a=(np.array(Image.open(image_name).rotate(rotation).resize((28, 28)), np.float32,copy=True))
print(a.shape)
Image.open(image_name).rotate(rotation).resize((28,28))

root_dir='/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data/'

train_split_path = os.path.join(root_dir, 'splits', 'train.txt')

with open(train_split_path, 'r') as train_split:
    train_classes = [line.rstrip() for line in train_split.readlines()]

#number of classes (在train.txt中每一列如'Angelic/character01/rot000'稱為1個class)
no_of_classes = len(train_classes)
print(no_of_classes)

#number of examples
num_examples = 20

#image width
img_width = 28

#image height
img_height = 28
channels = 1

train_dataset = np.zeros([no_of_classes, num_examples, img_height, img_width], dtype=np.float32)
print(train_dataset.shape)

for label, name in enumerate(train_classes): # enumerate() 預設是由0開始
    #if (label>=50):
    #  break;
    alphabet, character, rotation = name.split('/')
    rotation = float(rotation[3:])
    img_dir = os.path.join(root_dir, 'images',alphabet, character)
    #if (label==1999):
    #  print(img_dir)
    #  dir_=os.path.join(img_dir, '*.png')
    #  print(dir_)
    #  print(glob.glob(os.path.join(img_dir, '*.png')))
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if (label%100==0):
      print("label:",label)
    if (label==2999):
      print(img_files)
    for index, img_file in enumerate(img_files):
        if (index>=num_examples):
          break
        values = 1. - np.array(Image.open(img_file).rotate(rotation).resize((img_width, img_height)), np.float32, copy=False)
        train_dataset[label, index] = values
print(train_dataset.shape)
print(train_dataset[0][0])


def convolution_block(inputs, out_channels, name='conv'):

    conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
    conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
    conv = tf.nn.relu(conv)
    conv = tf.contrib.layers.max_pool2d(conv, 2)

    return conv

def get_embeddings(support_set, h_dim, z_dim, reuse=False):

        net = convolution_block(support_set, h_dim)
        net = convolution_block(net, h_dim)
        net = convolution_block(net, h_dim) 
        net = convolution_block(net, z_dim) 
        net = tf.contrib.layers.flatten(net)
        
        return net


#number of classes
num_way = 10 # 60  

#number of examples per class for support set
num_shot = 5  

#number of query points
num_query = 5 

#number of examples
num_examples = 20

h_dim = 64

z_dim = 64

support_set = tf.placeholder(tf.float32, [None, None, img_height, img_width, channels])
query_set = tf.placeholder(tf.float32, [None, None, img_height, img_width, channels])

support_set_shape = tf.shape(support_set)
query_set_shape = tf.shape(query_set)

num_classes, num_support_points = support_set_shape[0], support_set_shape[1]
num_query_points = query_set_shape[1]

y = tf.placeholder(tf.int64, [None, None])

#convert the label to one hot
y_one_hot = tf.one_hot(y, depth=num_classes)

support_set_embeddings = get_embeddings(tf.reshape(support_set, [num_classes * num_support_points, img_height, img_width, channels]), h_dim, z_dim)

support_set_embeddings_shape=tf.shape(support_set_embeddings)
embedding_dimension = tf.shape(support_set_embeddings)[-1]
class_prototype = tf.reduce_mean(tf.reshape(support_set_embeddings, [num_classes, num_support_points, embedding_dimension]), axis=1)
class_prototype_shape=tf.shape(class_prototype)

query_set_embeddings = get_embeddings(tf.reshape(query_set, [num_classes * num_query_points, img_height, img_width, channels]), h_dim, z_dim, reuse=True)
query_set_embeddings_shape=tf.shape(query_set_embeddings)

def euclidean_distance(a, b):

    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)

distance = euclidean_distance(query_set_embeddings,class_prototype)
distance_shape=tf.shape(distance)

predicted_probability = tf.reshape(tf.nn.log_softmax(-distance), [num_classes, num_query_points, -1])
predicted_probability_shape=tf.shape(predicted_probability)

loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, predicted_probability), axis=-1), [-1]))

accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(predicted_probability, axis=-1), y)))

train = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

num_epochs = 20
num_episodes = 100

for epoch in range(num_epochs):
    #if (epoch >=2):
    #  break 
    for episode in range(num_episodes):
        
        # select 60 classes
        episodic_classes = np.random.permutation(no_of_classes)[:num_way]
        
        support = np.zeros([num_way, num_shot, img_height, img_width], dtype=np.float32)#底下利用np.expand_dims()增加1維,使其與support_set匹配
        
        query = np.zeros([num_way, num_query, img_height, img_width], dtype=np.float32)
        
        
        for index, class_ in enumerate(episodic_classes):
            selected = np.random.permutation(num_examples)[:num_shot + num_query]
            support[index] = train_dataset[class_, selected[:num_shot]]
            
            # 5 querypoints per classs
            query[index] = train_dataset[class_, selected[num_shot:]]
            
        support = np.expand_dims(support, axis=-1)
        query = np.expand_dims(query, axis=-1)
        labels = np.tile(np.arange(num_way)[:, np.newaxis], (1, num_query)).astype(np.uint8)
        _, loss_, accuracy_ = sess.run([train, loss, accuracy], feed_dict={support_set: support, query_set: query, y:labels})
        
        #print("support_set_embeddings_shape=",sess.run(support_set_embeddings_shape,feed_dict={support_set: support, query_set: query, y:labels}))
        #print("class_prototype_shape=",sess.run(class_prototype_shape,feed_dict={support_set: support, query_set: query, y:labels}))
        #print("query_set_embeddings_shape=",sess.run(query_set_embeddings_shape,feed_dict={support_set: support, query_set: query, y:labels}))
        #print("distance_shape=",sess.run(distance_shape,feed_dict={support_set: support, query_set: query, y:labels}))        
        #print("predicted_probability_shape=",sess.run(predicted_probability_shape,feed_dict={support_set: support, query_set: query, y:labels}))

        if (episode+1) % 10 == 0:
            print('Epoch {} : Episode {} : Loss: {}, Accuracy: {}'.format(epoch+1, episode+1, loss_, accuracy_))
        
        #if (episode>=2):
        #  break