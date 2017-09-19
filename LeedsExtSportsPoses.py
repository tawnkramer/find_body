
# coding: utf-8

# In[1]:


import os
import dbcollection as dbc
from dbcollection.utils.string_ascii import convert_ascii_to_str as tostr
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# return a loader of the dataset
lsp = dbc.load('leeds_sports_pose_extended')

print(lsp.sets)

set_name = 'test' #'train'

height, width, ch = 400, 400, 3

def get_set(lsp, set_name):
    num_samples = lsp.get(set_name, 'image_filenames').shape[0]
    print('num_samples', num_samples, 'in', set_name)

    data_set = []
    
    # fetch data from the hdf5 metadata file using the object() method
    for iObj in range(num_samples):        
        data = lsp.object(set_name, iObj, True)  # 100th object (True converts the fields indexes to values automatically)
        img_fname =  tostr(data[0])
        keypoints = data[1]
        filename = os.path.join(lsp.data_dir, img_fname)
        data_set.append({'filename' : filename, 'keypoints' : keypoints})
        
    return data_set

def get_keypoint_names(lsp):
    names = (tostr(lsp.get('train', 'keypoint_names')))
    print(names)
    return names

def draw_point(p, img, col=(0, 0, 255)):
    cv2.circle(img, p, 1, col, -1)

def show_sample(s):
    image_path = s['filename']
    points = s['keypoints']
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    r=[255, 0, 0]
    g=[0, 255, 0]
    b=[0, 0, 255]
    
    for p in points:
        col = r
        if p[2] == 0.0:
            col = b
        elif p[2] == -1.0:
            col = g
        pt = (int(p[0]), int(p[1]))
        draw_point(pt, img, col)
        
    print('height, width', height, width)
    rimg = np.zeros((height, width, 3), np.uint8)
    rimg[0:img.shape[0], 0:img.shape[1]] = img

    fig=plt.figure(figsize=(9, 8))
    plt.imshow(rimg)
    plt.show()
    


# In[2]:


train_set = get_set(lsp, 'train')
test_set = get_set(lsp, 'test')


# In[3]:


#show_sample(test_set[1])
keypoint_names = get_keypoint_names(lsp)


# In[4]:

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Lambda, ELU
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Cropping2D, BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import keras
print(keras.__version__)


# In[5]:

def custom_objective(y_true, y_pred):
    return T.sqr((y_true - y_pred))

def get_model(width, height, ch, num_outputs):
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(height, width, ch)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same"))
    #model.add(BatchNormalization())
    model.add(ELU())
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    #model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    #model.add(BatchNormalization())    
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    #model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    #model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    #model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(.5))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    #model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Dropout(.1))
    model.add(ELU())
    model.add(Dense(128))
    model.add(ELU())
    model.add(Dense(num_outputs))

    model.compile(optimizer=Adam(), loss="mse")
    return model

def show_model_summary(model):
    #model.summary()
    for layer in model.layers:
        print(layer.output_shape)
        


# In[9]:


import random

def shuffle(samples):
    '''
    randomly mix a list and return a new list
    '''
    ret_arr = []
    len_samples = len(samples)
    while len_samples > 0:
        iSample = random.randrange(0, len_samples)
        ret_arr.append(samples[iSample])
        del samples[iSample]
        len_samples -= 1
    return ret_arr

def generator(samples, batch_size=32):
    num_samples = len(samples)
    iJoint = 0
    
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            labels = []
            for sample in batch_samples:
                
                filename = sample["filename"]
                keypoints = sample["keypoints"]


                img = cv2.imread(filename)
                image = np.zeros((height, width, 3), np.uint8)
                image[0:img.shape[0], 0:img.shape[1]] = img[0:height, 0:width]


                if image is None:
                    print('failed to open', filename)
                    continue

                images.append(image)

                kp = keypoints[iJoint]
                keypoint = [kp[0], kp[1]]
                labels.append(keypoint)

                

            # final np array to submit to training
            X_train = np.array(images)
            y_train = np.array(labels)
            yield X_train, y_train
            


# In[10]:


batch_size = 64
epochs = 100
train_generator = generator(train_set, batch_size)
valid_generator = generator(test_set, batch_size)
n_train, n_val = len(train_set), len(test_set)
steps_per_epoch = n_train // batch_size
validation_steps = n_val // batch_size

num_outputs = 2
model = get_model(width, height, ch, num_outputs)

#show_model_summary(model)

model_name = 'find_part.h5'

callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True, verbose=0),
    ]

# In[11]:


history = model.fit_generator(train_generator, 
        steps_per_epoch = steps_per_epoch,
        validation_data = valid_generator,
        validation_steps = validation_steps,
        epochs=epochs,
        verbose=1,
        callbacks=None)

