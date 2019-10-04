# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:28:35 2018

@author: kui
"""

from keras.layers import merge,Conv3D,ZeroPadding3D,Input,BatchNormalization,Activation,MaxPooling3D,UpSampling3D,Deconv3D
from keras.models import Model
from keras.utils.vis_utils import plot_model
from PIL import Image
import numpy as np
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import concatenate,add,Dot,multiply
from keras.layers.core import Dense, Dropout, Reshape
from keras.layers.pooling import AveragePooling3D, MaxPooling3D
from keras.layers.pooling import GlobalAveragePooling3D
from keras.utils.layer_utils import convert_all_kernels_in_model, convert_dense_weights_data_format
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape,decode_predictions
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.core import Lambda
import tensorflow as tf 
#from Augment import Augment_3D
K.set_image_data_format('channels_first')

smooth = 1.0
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


      
def standartize(array):   
    mean=np.mean(array)
    std =np.std(array)
    array -= mean
    array /= std
    return array    


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=5e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=0)(ip)
    x = Activation('relu')(x)


    if bottleneck:
        inter_channel = nb_filter*4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Conv3D(inter_channel, (1,1,1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis)(x)
        x = Activation('relu')(x)

    x = Conv3D(nb_filter, (3,3, 3), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=5e-4,
                  grow_nb_filters=True, return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: flag to decide to allow number of filters to grow
        return_concat_list: return the list of feature maps along with the actual output
    Returns: keras tensor with nb_layers of conv_block appended
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=5e-4,type='down'):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis)(ip)
    x = Activation('relu')(x)
    x0 = Conv3D(int(nb_filter * compression), (1,1,1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
#    x = AveragePooling3D((2,2,2), strides=(2,2,2))(x)
    return x0


def transition_block(ip, nb_filter,weight_decay=5e-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        compression: calculated as 1 - reduction. Reduces the number of feature maps
                    in the transition block.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis)(ip)
    x = Activation('relu')(x)
    x0 = Conv3D(nb_filter, (1,1,1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
#    x = AveragePooling3D((2,2,2), strides=(2,2,2))(x)
    return x0




def __transition_up_block(ip, nb_filters,type='deconv',output_shape=(None,1,16,64,64),weight_decay=5E-4):
    ''' SubpixelConvolutional Upscaling (factor = 2)
    Args:
        ip: keras tensor
        nb_filters: number of layers
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
        weight_decay: weight decay factor
    Returns: keras tensor, after applying upsampling operation.
    '''

    if type == 'upsampling':
        x = UpSampling3D()(ip)
    else:
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = BatchNormalization(axis=concat_axis)(ip)
        x = Activation('relu')(x)
        x = Deconv3D(nb_filters,(3,3,3),strides=(2,2,2), activation='relu', padding='same',data_format='channels_first',
                            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

    return x
  
def conv_block(ip, nb_filter,dropout_rate=None, weight_decay=5e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis)(ip)
    x = Activation('relu')(x)

    x = Conv3D(nb_filter, (3,3,3), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
#    if dropout_rate:
#        x = Dropout(dropout_rate)(x)

    return x


def multi(x):
    return x[0]*x[1]

def Apply_Attention(input1,input2):
      atten=Activation('sigmoid')(input1)
      out=multiply([atten,input2])
      print('ADD',K.int_shape(out))
      out = add([out,input1])
      return out


         
def down_stage(ip,nb_layers,nb_filter,growth_rate,dropout_rate,weight_decay,compression,pooling=True):
    
    x0, nb_filter = __dense_block(ip,nb_layers , nb_filter, growth_rate,bottleneck=True,dropout_rate=dropout_rate,
                                     weight_decay=weight_decay)
    print('x:',K.int_shape(x0))
    print(nb_filter)      
    x1=transition_block(x0, nb_filter,weight_decay=weight_decay)
    x =transition_block(ip, nb_filter,weight_decay=weight_decay)
#    addx=Apply_Attention(x1,x)
    addx=add([x,x1])
    if pooling:    
        out= AveragePooling3D(strides=(2,2,2))(addx)
        return addx,out
    return addx


def up_stage(ip,nb_layers,nb_filter,growth_rate,dropout_rate,weight_decay,compression,type='up'):
    
    
#    x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay,type=type)
    x0, nb_filter = __dense_block(ip,nb_layers , nb_filter, growth_rate,bottleneck=True,dropout_rate=dropout_rate,
                                     weight_decay=weight_decay)
    x1 =transition_block(x0, nb_filter,weight_decay=weight_decay)
#    x =transition_block(ip, nb_filter,weight_decay=weight_decay)    
#    addx=Apply_Attention(x1,x)
    addx=add([ip,x1])                                 
    return addx
 
def Decon_stage(x,nb_filters,kernel_size,strides,weight_decay):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Deconv3D(nb_filters,kernel_size,strides=strides, activation='relu', padding='same',data_format='channels_first',
                        kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
#    x = UpSampling3D()(x)    
    return x




def output(x,weight_decay):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(1, (1,1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Activation('sigmoid')(x)
    return x

def side_out(x,up_size,weight_decay):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation('relu')(x)
    x = Conv3D(1, (1,1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Activation('sigmoid')(x)
    up1=UpSampling3D((up_size,up_size,up_size))(x)
    return up1
    

img_rows = 96
img_cols = 96
chan =16
same_num=4
def Dense_net(growth_rate=16,reduction=0.5, dropout_rate=0.3, weight_decay=5e-4,upsampling_type='upsampling',init_conv_filters=32):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        nb_layers_per_block: number of layers in each dense block.
            Can be a positive integer or a list.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        nb_upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
        upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
            type of upsampling algorithm used.
        input_shape: Only used for shape inference in fully convolutional networks.
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    
    if  K.image_data_format() == 'channels_last':
      img_input = Input(shape=(img_rows,img_cols,chan,1))
      concat_axis=-1
    else:
      img_input = Input(shape=(1,chan, img_rows, img_cols))
      concat_axis=1


    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # compute compression factor
    compression = 1.0 - reduction
    nb_layers=[4,8,16,8,4,2]
    growth_rate=32
    # Initial convolution

  
    
    x1 = Conv3D(64, (3,3,3),strides=(1,1,1),kernel_initializer='he_normal',padding='same',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=concat_axis)(x1)
    x = Activation('relu')(x)
    x = AveragePooling3D((2,2,2))(x)
    print('x:',K.int_shape(x))

    # Add dense blocks and transition down block
    #stage1
 #  nb_layers[0]=int((nb_filter[0]-32)/growth_rate)
    print('nb_layers:',(nb_layers[0]))
    s1_x0,s1_x = down_stage(x,nb_layers[0],0,growth_rate,dropout_rate,weight_decay,compression)
    print('s10,s11:',K.int_shape(s1_x0),K.int_shape(s1_x))    
    
    #stage2
 #   nb_layers[1]=int((nb_filter[1]-nb_layers[0])/growth_rate)
    print('nb_layers:',(nb_layers[1]))
    s2_x0,s2_x=down_stage(s1_x,nb_layers[1],0,growth_rate,dropout_rate,weight_decay,compression)    
    print('s20,s21:',K.int_shape(s2_x0),K.int_shape(s2_x))  
    
    #stage3 
 #   nb_layers[2]=int((nb_filter[2]-nb_layers[1])/growth_rate)
    print('nb_layers:',(nb_layers[2]))    
    s3_x0     =down_stage(s2_x,nb_layers[2],0,growth_rate,dropout_rate,weight_decay,compression,pooling=False)
    print('s3:',K.int_shape(s3_x0))    
    
    #stage4
    D1=Decon_stage(s3_x0,256,kernel_size=(3,3,3),strides=(2,2,2),weight_decay=weight_decay)
    print('D1:',K.int_shape(D1))
    
    con1 =Apply_Attention(D1,s2_x0) #add([D1,s2_x0])  
 #   nb_layers[3]=int((nb_filter[3]-nb_layers[2])/growth_rate)  
    print('nb_layers:',(nb_layers[3]))    
    s4_x=up_stage(con1,nb_layers[3],0,growth_rate,dropout_rate,weight_decay,compression)
    print('s4:',K.int_shape(s4_x))    
    
    
    #stage5    
    D2=Decon_stage(s4_x,128,kernel_size=(3,3,3),strides=(2,2,2),weight_decay=weight_decay)
    print('D2:',K.int_shape(D2))
    con2 =Apply_Attention(D2,s1_x0) #add([D2,s1_x0])
#    nb_layers[4]=int((nb_filter[4]-nb_layers[3])/growth_rate)   
    print('nb_layers:',(nb_layers[4]))       
    s5_x=up_stage(con2,nb_layers[4],0,growth_rate,dropout_rate,weight_decay,compression)    
    print('s5:',K.int_shape(s5_x))    
   
    #stage6  
    D3=Decon_stage(s5_x,64,kernel_size=(3,3,3),strides=(2,2,2),weight_decay=weight_decay)
    print('D3:',K.int_shape(D3))
    con3 =Apply_Attention(D3,x1) #add([D3,x1])
    print('con3:',K.int_shape(con3))
    s6_x=up_stage(con3,nb_layers[5],0,growth_rate,dropout_rate,weight_decay,compression)       
    print('s6:',K.int_shape(s6_x)) 
#########################################################################  
    
#    output1=Deconv3D(1,(8,8,8),strides=(4,4,4),padding='same')(D1)
#    output1=Activation('relu')(output1)     
    output1 =side_out(D1,4,weight_decay)#,output_shape=(None,1,16,64,64)
    output2 =side_out(D2,2,weight_decay)
    output3 =output(D3,weight_decay)#,output_shape=(None,1,16,64,64)    
    main_out=output(s6_x,weight_decay)  
    print(K.int_shape(output1))
    print(K.int_shape(output2))
    print(K.int_shape(output3))    
    print(K.int_shape(main_out))
    model = Model(img_input, [output1,output2,output3,main_out])
  #  plot_model(model, to_file='Vnet.png',show_shapes=True)
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy',loss_weights=[0.1,0.15,0.25,0.5],metrics=[dice_coef])#'binary_crossentropy'loss_weights=[0.3,0.6,1.0]
    return model  
    

def preprocess(imgs):
    imgs=imgs.reshape(imgs.shape[0],1,chan,imgs.shape[-2],imgs.shape[-1])
    return imgs
    
def preprocess_ge(imgs):
    imgs=imgs.reshape(imgs.shape[0],imgs.shape[-2],imgs.shape[-1])
    return imgs

    
    
def random_crop(image,gtruth, crop_size):
    chan,height, width = image.shape
    dz,dy, dx = crop_size
#    hz,hx,hy=4,16,16
#    num=32
    if width < dx or height < dy:
        return None
    z = np.random.randint(0, chan-dz+1)
    x = np.random.randint(0, width-dx+1)
    y = np.random.randint(0, height-dy+1)
    CropIM=image[z:z+dz, y:y+dy, x:(x+dx)]
    CropGT=gtruth[z:z+dz, y:y+dy, x:(x+dx)]
    
#    for i in range(np.random.randint(0,num)):
#        z = np.random.randint(0,dz-hz+1)
#        x = np.random.randint(0,dx-hx+1)
#        y = np.random.randint(0,dy-hy+1)
#        CropIM[z:z+hz,y:y+hy,x:x+hx]=0
    return [CropIM,CropGT]
    
    
def generate_arrays_from_file(x,y):
    while 1:
#        print x.shape
        for i in range(x.shape[0]):#x.shape[0]

            IM_patch=[]
            GT_patch=[]
            im=preprocess_ge(x[i])#.reshape(x[i].shape[-2],x[i].shape[-1])
            gt=preprocess_ge(y[i])#.reshape(y[i].shape[-2],y[i].shape[-1])            
            for j in range(same_num):
  

    #            print(im.shape)
                ##########get_patch######################
                [impath,gtpath]=random_crop(im,gt,[chan,img_rows,img_cols])
                IM_patch.append(impath)
                GT_patch.append(gtpath)
            
    
            leng=len(IM_patch)            
            imgs_train=np.array(IM_patch).reshape(leng,1,chan,img_rows,img_cols)
            imgs_mask=np.array(GT_patch).reshape(leng,1,chan,img_rows,img_cols)
            imgs_train = (imgs_train.astype('float32'))
            imgs_mask=imgs_mask.astype('float32')
#            print(imgs_train.shape)
            yield (imgs_train,[imgs_mask]*4)
      

      
def generate_arrays_from_train(x,y):
    while 1:
#        print x.shape
        for i in range(x.shape[0]/same_num):#x.shape[0]

            IM_patch=[]
            GT_patch=[]
            for j in range(i*same_num,(i+1)*same_num):
  
                im=preprocess_ge(x[j])#.reshape(x[i].shape[-2],x[i].shape[-1])
                gt=preprocess_ge(y[j])#.reshape(y[i].shape[-2],y[i].shape[-1])
    #            print(im.shape)
                ##########get_patch######################
                [impath,gtpath]=random_crop(im,gt,[chan,img_rows,img_cols])
                IM_patch.append(impath)
                GT_patch.append(gtpath)
            
            leng=len(IM_patch)
            imgs_train=np.array(IM_patch).reshape(leng,1,chan,img_rows,img_cols)
            imgs_mask=np.array(GT_patch).reshape(leng,1,chan,img_rows,img_cols)
            imgs_train = (imgs_train.astype('float32'))
            imgs_mask=imgs_mask.astype('float32')
          #  print(imgs_train.shape)
            yield (imgs_train,[imgs_mask]*4)
            

            
def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    #path='../Data/Inter_Nad_prostate_all_CW/'#'/home/kui/Downloads/Volumetric-ConvNet-for-prostate-segmentation-master/data/TrainAndTest/'#'../DataC/3D_others/'#'/home/kui/prostate_segment/ChallengImage/TrainAndTest/Patch3D/'#'../DataC/'#
    path='../Data/InterNad/Inter_Nad_max_slice_all/'
#    test_x = np.load(path+'test_x_path.npy')
#    test_y = np.load(path+'test_y_path.npy')

    imgs_train = np.load(path+'Augtrain_x_all.npy')
    imgs_mask_train = np.load(path+'Augtrain_y_all.npy')
 
    print('trainsamples',imgs_train.shape)


    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    model = Dense_net()
    model.load_weights('ATMixNet6_all_slice4.hdf5')
    model_checkpoint = ModelCheckpoint('ATMixNet6_all_slice4_1.hdf5', monitor='loss', save_best_only=True)
   # early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    print(imgs_train.shape[0]/same_num)
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit_generator(generate_arrays_from_file(imgs_train,imgs_mask_train),samples_per_epoch=(imgs_train.shape[0]), nb_epoch=20000 ,verbose=2, validation_data=generate_arrays_from_file(imgs_train,imgs_mask_train),validation_steps=imgs_train.shape[0],callbacks=[model_checkpoint])



#    hist= model.fit_generator(imgs_train,imgs_mask_train,samples_per_epoch=len(imgs_train), nb_epoch=10000 ,verbose=2, validation_data=[test_x,test_y],callbacks=[model_checkpoint])
#
#    hist= model.fit(imgs_train,[imgs_mask_train]*3, batch_size=16, nb_epoch=10000, verbose=2, shuffle=True,validation_data=[test_x,[test_y]*3],callbacks=[model_checkpoint])#[imgs_mask_train,imgs_mask_train,imgs_mask_train,imgs_mask_train,imgs_mask_train,imgs_mask_train]
#    print(hist.history)
if __name__ == '__main__':
#     model = Vnet()
#     plot_model(model,'1.png',True,True)
#    model.load_weights('Vnet3_1.hdf5')
#    model.save('vnet3.hdf5')
     train_and_predict()  
 #    model = Dense_net()
