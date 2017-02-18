#import sys

#sys.path.append('/home/student/anaconda2/lib/python2.7/site-packages')

import os, sys
from PIL import Image

#os.system("ffmpeg -ss 0:0:0 -t 2 -i video-1.avi -f image2 image-%d.png")


import subprocess

import matplotlib.pyplot as plt  # za prikaz slika, grafika, itd.

from sklearn.datasets import fetch_mldata
import numpy as np



from skimage.io import imread
from scipy import ndimage
from skimage.color import rgb2gray
import cv2
from skimage.morphology import square, diamond, disk  # strukturni elementi
from skimage.morphology import dilation, erosion
from skimage.morphology import opening, closing
from skimage.measure import label  # implementacija connected-components labelling postupka
from skimage.measure import regionprops  # da mozemo da dobavimo osobine svakog regiona
#--------------- ANN ------------------
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD


def getLength(filename):
  result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines() if "Duration" in x]


def getFrames(filename,number):
  l=getLength(filename)
  time=l[0][11:22]
  command="ffmpeg -ss 0:0:0 -t " + time + " -i " + filename + " -f image2 image" + number.__str__() + "-%d.png"
  os.system(command)

def getRegionsCount(img):
  dil_image=dilation(img,selem=disk(5))
  labeled_img = label(dil_image)  # rezultat je slika sa obelezenim regionima
  regions = regionprops(labeled_img)
  return len(regions)

def getMask(img):
    dil_image = dilation(img, selem=disk(5))
    dil_image = dilation(dil_image, selem=disk(5))
    dil_image = dilation(dil_image, selem=disk(5))
    #dil_image = dilation(dil_image, selem=disk(5))
    return dil_image

def to_categorical(labels, n):
    retVal = np.zeros((len(labels), n), dtype='int')
    ll = np.array(list(enumerate(labels)))
    retVal[ll[:,0],ll[:,1]] = 1
    return retVal

#MNIST
print 'mnist'
mnist = fetch_mldata('MNIST original')

data   = mnist.data / 255.0
labels = mnist.target.astype('int')

train_rank = 5000
#test_rank = 100
#------- MNIST subset --------------------------
train_subset = np.random.choice(data.shape[0], train_rank)
#test_subset = np.random.choice(data.shape[0], test_rank)

# train dataset
train_data = data[train_subset]
train_labels = labels[train_subset]

# test dataset
#test_data = data[test_subset]
#test_labels = labels[test_subset]

# train and test to categorical
train_out = to_categorical(train_labels, 10)
#test_out = to_categorical(test_labels, 10)
print 'model nn'
# prepare model
model = Sequential()
model.add(Dense(70, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('relu'))

# compile model with optimizer
sgd = SGD(lr=0.1, decay=0.001, momentum=0.7)
model.compile(loss='mean_squared_error', optimizer=sgd)

print 'trening'
# training
training = model.fit(train_data, train_out, nb_epoch=500, batch_size=400, verbose=0)
print training.history['loss'][-1]

# evaluate on test data
#scores = model.evaluate(test_data, test_out, verbose=1)
#print 'test', scores

# evaluate on train data
#scores = model.evaluate(train_data, train_out, verbose=1)
#print 'train', scores


for rbr in range(1,10):
    getFrames("Videos/video-" + rbr.__str__() + ".avi", rbr)

    str_elem = disk(5)  # parametar je poluprecnik diska
    a=[]
    previus_regions_count=0


    for i in range(1,1201):
        string="image" + rbr.__str__() + "-" + i.__str__() + ".png"
        img = imread(string)

        print(i)

        img_bin_numbers = cv2.inRange(img,(160, 160, 160),(255, 255, 255))

        img_bin_line = cv2.inRange(img,(0, 0, 160),(20, 20, 255))

        img_tr_dil = dilation(img_bin_numbers, selem=str_elem)

        #img_bin_close = closing(img_bin_numbers, selem=str_elem)
        #img_bin_intersection=img_bin_line*img_bin_close
        img_bin_intersection=img_bin_line*img_tr_dil

        img_mask=getMask(img_bin_intersection)

        img_numbers_on_line=img_mask*img_bin_numbers

        img_final=getMask(img_numbers_on_line)*img_bin_numbers

        img_num_line_dil=dilation(img_final,selem=str_elem)

        regions_count=getRegionsCount(img_num_line_dil)

        if regions_count != previus_regions_count:
            a.append(img_final)
            previus_regions_count=regions_count

    b=[]
    previus_regions_count=0
    for i in range(0,a.__len__()-1):
        img_dil=dilation(a[i],selem=str_elem)
        regions_count = getRegionsCount(img_dil)
        if regions_count!=0:
            if previus_regions_count!=0:
                if regions_count!=previus_regions_count:
                    pr_img = a[i-1]
                    pr_img_mask = getMask(pr_img)
                    neg_pr_img_mask = 1 - pr_img_mask
                    cr_img = a[i] * neg_pr_img_mask
                    b.append(cr_img)
                    cr_img_dil=dilation(cr_img,selem=str_elem)
                    previus_regions_count=getRegionsCount(cr_img_dil)
            else:
                previus_regions_count=regions_count
                b.append(a[i])
        else:
            previus_regions_count=0

    c=[]
    for img in b:

        dil_image = dilation(img, selem=disk(5))

        labeled_img = label(dil_image)  # rezultat je slika sa obelezenim regionima
        regions = regionprops(labeled_img)

        for reg in regions:
            bbox = reg.bbox

            center_h=(bbox[0]+bbox[2])/2
            center_w=(bbox[1] + bbox[3]) / 2
            img_crop=img[center_h-14:center_h+14,center_w-14:center_w+14]
            c.append(img_crop)

    suma=0
    d=[]
    for img in c:
        plt.imshow(img,'gray')
        plt.show()
        imgB=img.reshape(784)
        imgB=imgB/255
        tt=model.predict(np.array([imgB]), verbose=1)
        rez_t = tt.argmax(axis=1)

        #print tt[0]

        #print rez_t
        suma = suma + rez_t[0]
        #print 'suma %d' % suma

    print 'Video: %d' %rbr
    print 'Suma je: %d' % suma