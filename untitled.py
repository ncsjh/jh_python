import numpy as np

#from skimage.io import imread
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7,EfficientNetB1,EfficientNetB2,EfficientNetB3,EfficientNetB4,EfficientNetB5,EfficientNetB6
#from efficientnet import EfficientNetB0, EfficientNetB3

from PIL import Image
#from efficientnet.preprocessing import center_crop_and_resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


path=os.path.dirname(os.path.abspath(__file__))
train_path=os.path.join(path, 'train')
val_path=os.path.join(path, 'test')
save_path=os.path.join(path,'saved_model')
width=[224, 240, 260, 300, 380, 456, 528, 600] 

mckp=ModelCheckpoint(filepath=os.path.join(save_path, 'model.hdf'), monitor='val_accuracy', save_best_only=True, verbose=1)
early=EarlyStopping(monitor='val_accuracy', patience=50)
                     

def get_classes(train_path):
    classes=[]
    dirs=os.path.listdir(train_path)
    for i in dirs:
        if len(i<3) : 
            cls.append(i)
            print(f'{i} 클래스 추가')
    return classes

def build_model(n, num_of_output) :
    print(f'EfficientNetB{n} 모델 생성 시작')
    
    img_arg=Sequential(
    [
     preprocessing.RandomRotation(factor=0.15),
     preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
     preprocessing.RandomFlip(),
     preprocessing.RandomContrast(factor=0.1)

    ])
    
    
    l=width[n]
    inputs=layers.Input(shape=(l,l,3))  #B7을 이용할거니까 가로 세로 600
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
    print('Trainable = False')
    
    x=layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    x=layers.BatchNormalization()(x)

    top_dropout_rate=0.2
    x=layers.Dropout(top_dropout_rate, name='top_dropout')(x)

    outputs=layers.Dense(num_of_output,
                       activation='softmax',
                       name='pred')(x)
    model=tf.keras.Model(inputs, outputs, name='EfficientNet')
    optimizer=tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model



def make_model_trainable(model, ratio):
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



n=0
ratio=0.5
image_size=width[n]
classes=get_classes(train_path)
epochs=50

model=build_model(n, len(classes))

model=make_model_trainable(model, ratio)

train_ds, val_ds = set_data(path, image_size, 123)
h=model.fit(train_ds, 
            validation_data=val_ds, 
            epochs=epochs, 
            callbacks=[mkcp, early])


