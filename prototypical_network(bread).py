from google.colab import drive
drive.mount ('/gdrive')
%cd /gdrive

import os
import glob
import cv2
from google.colab.patches import cv2_imshow 
import numpy as np

imhigh_Bread = 640
imwidth_Bread = 640
channels = 4

root_dir='/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data-bread/'
images_original='images-original-10'

def toStandardBreadSize(filepathname,imhgih_Bread=640,imwidth_Bread=640,channels=3,color_expandArea=255):
  value=np.zeros((imhigh_Bread,imwidth_Bread,channels))

  image=cv2.imread(filepathname,cv2.IMREAD_UNCHANGED)
  imhigh_image=np.float32(image.shape[0])
  imwidth_image=np.float32(image.shape[1])
  if (imhigh_image>=imwidth_image):# 相片高大於寬
    f=imhigh_Bread/imhigh_image #原相片需要縮放比率
  elif (imhigh_image<imwidth_image):# 相片高小於寬
    f=imwidth_Bread/imwidth_image #原相片需要放大比率
  
  image=cv2.resize(image,None,fx=f,fy=f,interpolation=cv2.INTER_AREA)
  imhigh_image=image.shape[0]
  imwidth_image=image.shape[1]

  if (imhigh_image>=imwidth_image):# 相片高大於寬,高=imhigh_Bread
    a=np.int32((imwidth_Bread-imwidth_image)/2)
    value[:,:a,:]=255
    value[:,a:(a+imwidth_image),:]=image
    value[:,(a+imwidth_image):,:]=255
  else:
    a=np.int32((imhigh_Bread-imhigh_image)/2)
    value[:a,:,:]=255
    value[a:(a+imhigh_image),:]=image
    value[(a+imhigh_image):,:,:]=255
  return value


if os.path.isdir(root_dir+'images/'): # 若(root_dir+'images/')存在
  print((root_dir+'images/') + " already exist")
else: # 若(root_dir+'images/')不存在,建立(root_dir+'images/')
  os.mkdir(root_dir+'images/') # 建立(root_dir+'images/')

for dirPath1, dirNames1, fileNames1 in os.walk(root_dir+images_original):
    
    for dirNames in dirNames1:
      dirPath=root_dir+'images/'+dirNames
      if os.path.isdir(dirPath): # 若dirPath存在
        print(dirPath + " already exist")
      else:# 若dirPath不存在
        os.mkdir(dirPath) # 建立dirPath
    
    for f1 in fileNames1:
      filepathname1 = os.path.join(dirPath1,f1)
      img=toStandardBreadSize(filepathname1,imhigh_Bread,imwidth_Bread,channels,color_expandArea=255)
      dirPath=dirPath1.replace(images_original,'images') # 將dirPath1中的images_original替換成'images'
      filepathname = os.path.join(dirPath,f1)
      cv2.imwrite(filepathname,img) # 存照片

image_filepathname=root_dir+'images/乳酪條/004.png'
#image_filepathname=root_dir+'images/奶酥葡萄/D122003.png'
#image_filepathname=root_dir+'images/日式紅豆/A114001.png'
#img=toStandardBreadSize(image_filename,imhigh_Bread,imwidth_Bread,channels,color_expandArea=255)

img=cv2.imread(image_filepathname,cv2.IMREAD_UNCHANGED)
print(img.shape)
cv2_imshow(img)

%tensorflow_version 1.x

import os
import glob
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow 

import numpy as np
import tensorflow as tf

print(tf.__version__)

image_name='/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data-bread/images/日式炒麵三明治_YakisobaBread/A113001.png'
image=cv2.imread(image_name,cv2.IMREAD_UNCHANGED)
print(type(image))
#print("image.shape=%s" % (image.shape))
print("image.shape=%s" % str(image.shape))
cv2_imshow(image)

image_name='/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data-bread/images/日式炒麵三明治_YakisobaBread/A113001.png'
image=Image.open(image_name)
print ("image.format=%s,image.size=%s,image.mode=%s" % (image.format,image.size,image.mode))
breadname, rotation = '日式炒麵三明治_YakisobaBread/rot000'.split('/')
print("breadname=%s,rotation=%s" % (breadname,rotation))
rotation = float(rotation[3:])
print("rotation=%f" % (rotation))
Image.open(image_name)

Image.open('/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data-bread/images/奶酥葡萄/001.png')


image_name = 'data/images/Sanskrit/character13/0863_13.png'
alphabet, character, rotation = 'Sanskrit/character13/rot000'.split('/')
rotation = float(rotation[3:])
image_name = '/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data-bread/images/奶酥葡萄/001.png'
breadname, rotation = '奶酥葡萄/rot090'.split('/')
rotation = float(rotation[3:])
print(alphabet, rotation)
print ("image.format=%s,image.size=%s,image.mode=%s" % (image.format,image.size,image.mode))
Image.open(image_name).rotate(rotation)

a=(np.array(Image.open(image_name).rotate(rotation).resize((64, 64)), np.float32,copy=True))
print(a.shape)
Image.open(image_name).rotate(rotation).resize((64,64))


root_dir='/gdrive/My Drive/AI_Project/meta-learning/Hands-On-Meta-Learning-With-Python/03.Prototypical Networks and its Variants/data-bread/'


train_split_path = os.path.join(root_dir, 'splits', 'train.txt')

with open(train_split_path, 'r') as train_split:
    train_classes = [line.rstrip() for line in train_split.readlines()]

#number of classes
no_of_classes = len(train_classes)
print(no_of_classes)


#number of examples
num_examples = 10 # 20

#image width
img_width = 84

#image height
img_height = 84

#image color channels
channels = 3


train_dataset = np.zeros([no_of_classes, num_examples, img_height, img_width, channels], dtype=np.float32)
print(train_dataset.shape)


for label, name in enumerate(train_classes):
    breadname, rotation = name.split('/')
    rotation = int(rotation[3:])
    #print(type(rotation))
    img_dir = os.path.join(root_dir, 'images', breadname)
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))

    for index, img_file in enumerate(img_files):
        if (index>=num_examples):
          break
        #values = 1. - np.array(Image.open(img_file).rotate(rotation).resize((img_width, img_height)), np.float32, copy=False)
        #values=np.array(Image.open(img_file).rotate(rotation).resize((img_width, img_height)), np.float32, copy=False)
        img=cv2.imread(img_file) # 讀取3個顏色通道channels,不讀取4個
        M=cv2.getRotationMatrix2D((img.shape[0]/2,img.shape[1]/2),rotation,1.0)
        img=cv2.warpAffine(img,M,(img.shape[0],img.shape[1]))
        img=cv2.resize(img,(img_width, img_height))
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        train_dataset[label, index] = img

print(train_dataset[19][4][40])
cv2_imshow(train_dataset[19][4])

train_dataset.shape

print(train_dataset[19][4][14])
train_dataset=train_dataset/255.0
print(train_dataset[19][4][14])


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
num_shot = 5 # 5  

#number of query points
num_query = 5 # 5 

#number of examples
num_examples = 10 # 20

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


embedding_dimension = tf.shape(support_set_embeddings)[-1]

class_prototype = tf.reduce_mean(tf.reshape(support_set_embeddings, [num_classes, num_support_points, embedding_dimension]), axis=1)


query_set_embeddings = get_embeddings(tf.reshape(query_set, [num_classes * num_query_points, img_height, img_width, channels]), h_dim, z_dim, reuse=True)


def euclidean_distance(a, b):

    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)


distance = euclidean_distance(class_prototype,query_set_embeddings)


predicted_probability = tf.reshape(tf.nn.log_softmax(-distance), [num_classes, num_query_points, -1])


loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, predicted_probability), axis=-1), [-1]))


accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(predicted_probability, axis=-1), y)))


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

num_epochs = 60
num_episodes = 100

for epoch in range(num_epochs):
    
    for episode in range(num_episodes):
        #print("episode=",episode)
        # select xx classes
        episodic_classes = np.random.permutation(no_of_classes)[:num_way]
        
        support = np.zeros([num_way, num_shot, img_height, img_width, channels], dtype=np.float32)
        
        query = np.zeros([num_way, num_query, img_height, img_width, channels], dtype=np.float32)
        
        for index, class_ in enumerate(episodic_classes):
            selected = np.random.permutation(num_examples)[:num_shot + num_query]
            support[index] = train_dataset[class_, selected[:num_shot]]
            
            # 5 querypoints per classs
            query[index] = train_dataset[class_, selected[num_shot:]]
            
        #support = np.expand_dims(support, axis=-1)
        #query = np.expand_dims(query, axis=-1)
        labels = np.tile(np.arange(num_way)[:, np.newaxis], (1, num_query)).astype(np.uint8)
        _, loss_, accuracy_ = sess.run([train, loss, accuracy], feed_dict={support_set: support, query_set: query, y:labels})
        
        if (episode+1) % 10 == 0:
            print('Epoch {} : Episode {} : Loss: {}, Accuracy: {}'.format(epoch+1, episode+1, loss_, accuracy_))