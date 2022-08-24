import numpy as np

#from skimage.io import imread
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions

from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6
# from efficientnet import EfficientNetB0, EfficientNetB3

from PIL import Image
#from efficientnet.preprocessing import center_crop_and_resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import datetime

path=os.path.dirname(os.path.abspath(__file__))
train_path=os.path.join(path, 'train', 'resize_600')
val_path=os.path.join(path, 'test')
save_path=os.path.join(path,'saved_model')
width=[224, 240, 260, 300, 380, 456, 528, 600] 

mckp=ModelCheckpoint(filepath=os.path.join(save_path, 'model.h5'), monitor='val_accuracy', save_best_only=True, verbose=1)
early=EarlyStopping(monitor='val_accuracy', patience=50)

img_arg=Sequential(
[
 preprocessing.RandomRotation(factor=0.15),
 preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
 preprocessing.RandomFlip(),
 preprocessing.RandomContrast(factor=0.1)

])
                     

def get_classes(train_path):
    classes=[]
    dirs=os.listdir(train_path)
    for i in dirs:
        if len(i)<3 : 
            classes.append(i)
            print(f'{i} 클래스 추가')
            

    
    return classes

def build_model(n, num_of_output, dropout) :
    print(f'EfficientNetB{n} 모델 생성 시작')
    print(f'{num_of_output} 클래스 분류 / Drop out = {dropout}')
    
    
    l=width[n]
    inputs=layers.Input(shape=(l,l,3))
    x=img_arg(inputs)
    print(f'Input Shape : ({l}, {l}, 3)')
    
    if n==0:
        model=EfficientNetB0(include_top=False,
                           input_tensor=x,
                           weights='imagenet'
                           )
        
    elif n==1:
        model=EfficientNetB1(include_top=False,
                           input_tensor=x,
                           weights='imagenet'
                           )
        
    elif n==2:
        model=EfficientNetB2(include_top=False,
                           input_tensor=x,
                           weights='imagenet'
                           )
        
    elif n==3:
        model=EfficientNetB3(include_top=False,
                           input_tensor=x,
                           weights='imagenet'
                           )
        
    elif n==4:
        model=EfficientNetB4(include_top=False,
                           input_tensor=x,
                           weights='imagenet'
                           )
        
    elif n==5:
        model=EfficientNetB5(include_top=False,
                           input_tensor=x,
                           weights='imagenet'
                           )
        
    elif n==6:
        model=EfficientNetB6(include_top=False,
                           input_tensor=x,
                           weights='imagenet'
                           )

    elif n==7:
        model=EfficientNetB7(include_top=False,
                           input_tensor=x,
                           weights='imagenet'
                           )
        
    model.trainable=False
    
    
    x=layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    x=layers.BatchNormalization()(x)

    x=layers.Dropout(dropout, name='top_dropout')(x)

    outputs=layers.Dense(num_of_output,
                       activation='softmax',
                       name='pred')(x)
    model=tf.keras.Model(inputs, outputs, name='EfficientNet')
    optimizer=tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model



def make_model_trainable(model, trainable_ratio):
    model_length=len(model.layers)
    trainable_length=int(model_len*ratio)
    for layer in model.layers[trainable_length:]:
        layer.trainable=True
    return model

        

        
def set_data(path, image_size, seed) : 
    
    train_ds=tf.keras.preprocessing.image_dataset_from_dictionary(os.path.join(path, 'train'), 
                                                                  image_size=(image_size, image_size), seed=seed)
    val_ds=tf.keras.preprocessing.image_dataset_from_dictionary(os.path.join(path, 'test'), 
                                                                image_size=(image_size, image_size), seed=seed)
    return train_ds, val_ds

def make_tensorboard_dir(dir_name):
    root_logdir=os.path.join(path, dir_name)
    sub_dir_name=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    return os.path.join(root_logdir, sub_dir_name)


classes=get_classes(train_path)
# n=[0,1,2,3,4,5,6,7]
n=[0,1]
trainable_ratios=[0.2, 0.4, 0.6, 0.8]
dropouts=[0.1, 0.2, 0.3, 0.4]
epochs=500

for i in n:
    for trainable_ratio in trainable_ratios:
        for dropout in dropouts:
            
            dir_name=f'learning_log_{n}_{trainable_ratio}_{dropout}'

            image_size=width[n]
    
            tb_log_dir=make_tensorboard_dir(dir_name)
            tensor_board=tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir)


            model=build_model(n, len(classes), dropout)

            model=make_model_trainable(model, trainable_ratio)

            train_ds, val_ds = set_data(path, image_size, 123)
            h=model.fit(train_ds, 
                        validation_data=val_ds, 
                        epochs=epochs, 
                        callbacks=[mkcp, early, tensor_board])


