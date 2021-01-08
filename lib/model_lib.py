from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras import models
K.set_image_data_format('channels_first')
kinit = 'he_normal'

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=1)
    #intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    #sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth



def unet(n_ch,patch_height,patch_width):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Dropout(0.2)(conv1)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    #
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Dropout(0.2)(conv2)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    #
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Dropout(0.2)(conv3)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([conv2,up1],axis=1)
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = layers.Dropout(0.2)(conv4)
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = layers.UpSampling2D(size=(2, 2),data_format='channels_first')(conv4)
    up2 = layers.concatenate([conv1,up2], axis=1)
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = layers.Dropout(0.2)(conv5)
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    #
    conv6 = layers.Conv2D(2, (1, 1), activation='relu',padding='same')(conv5)
    conv6 = layers.core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer= Adam(), loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

#for batch normalization
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    x = layers.Conv2D(filters=n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer= 'he_normal',
                     padding = 'same')(input_tensor)
    
    if batchnorm ==True:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters=n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer= 'he_normal',
                     padding = 'same')(x)
    if batchnorm ==True:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    return x





'''
For Attention block

[To Do]
    - Explain 'Function what to do'
    - Explain 'What is the self-attention in cnn?'
    - Explain 'Novelty of this system'

'''
def expend_as(tensor, rep, name):
    # 20.02.11 element axis 3 -> 1
    my_repeat = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis = 1), arguments={'repnum' : rep}, name = 'psi_up'+name)(tensor)
    return my_repeat

def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x 
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
    
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)
    
    print('shape x,g ', shape_x,shape_g)
    
    theta_x = layers.Conv2D(inter_shape, (2,2), strides=(2,2), padding='same', name = 'xl'+name)(x)
    shape_theta_x  = K.int_shape(theta_x)
    print('inter shape :  ', inter_shape)

    
    phi_g = layers.Conv2D(inter_shape, (1,1), padding='same')(g)
    
    # 20.02.11 shape chane x, [1] -> [2] / y, [2] -> [3]
    print('stride x : {} stride y : {}'.format(shape_theta_x[2] // shape_g[2] ,shape_theta_x[3] // shape_g[3] ))
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]),padding='same', name='g_up'+name)(phi_g)  # 16
    #upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),strides=(2, 2),padding='same', name='g_up'+name)(phi_g)
    print('theta_x shape : ', K.int_shape(theta_x))
    print('upsample_g shape : ', K.int_shape(upsample_g))
    
    concat_xg = layers.merge.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    
    psi = layers.Conv2D(1, (1,1), padding= 'same', name = 'psi'+name)(act_xg) #alpha
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    
    # 20.02.11 shape change x -> [2] y -> [3]
    upsample_psi = layers.UpSampling2D(size=(shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(sigmoid_xg)
    
    # 20.02.11 shape[3] -> shape[1]
    upsample_psi = expend_as(upsample_psi, shape_x[1], name)
    
    y = layers.merge.multiply([upsample_psi, x], name = 'q_attn'+name)
    
    # 20.02.11 shape change shape_x[3] -> shape_x[1]
    result = layers.Conv2D(shape_x[1], (1, 1), padding='same',name='q_attn_conv'+name)(y)
    result_bn = layers.BatchNormalization(name='q_attn_bn'+name)(result)
    
    return result_bn

def UnetGatingSignal(input, is_batchnorm, name):
    '''
    this is simply 1x1 convolution, bn, activation
    
    나는 ch-first를 사용한다. 
    shape[3]이 좀 이상하지?
    
    수정 할 필요가 있다.
    20.02.11
    x shape[3] => x shape[1]
    '''
    
    shape = K.int_shape(input)
    # 20.02.11 shape change, shape[3] -> shape[1]
    x = layers.Conv2D(shape[1] * 1, (1, 1), strides=(1, 1), padding="same", name=name + '_conv')(input)
    if is_batchnorm:
        x = layers.BatchNormalization(name=name + '_bn')(x)
    x = layers.Activation('relu', name = name+ '_act')(x)
    return x

def channel_attention(input_feature, ratio=8):
    print('input feature shape', input_feature._keras_shape)
    channel = input_feature._keras_shape[1]

    shared_layer_one = layers.Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    shared_layer_two = layers.Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    print('before reshpae avg pool', avg_pool._keras_shape)
    avg_pool = layers.Reshape((channel,1, 1))(avg_pool)
    print('after reshpae avg pool', avg_pool._keras_shape)

    avg_pool = shared_layer_one(avg_pool)
    print('after shared layer 01 avg pool', avg_pool._keras_shape)

    avg_pool = shared_layer_two(avg_pool)
    print('after shared layer 02 avg pool', avg_pool._keras_shape)


    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    print('before reshpae max pool', max_pool._keras_shape)

    max_pool = layers.Reshape((channel,1, 1))(max_pool)
    print('after reshpae max pool', max_pool._keras_shape)

    max_pool = shared_layer_one(max_pool)
    print('after shared layer 01 max pool', max_pool._keras_shape)

    max_pool = shared_layer_two(max_pool)
    print('before shared layer 02 max pool', max_pool._keras_shape)


    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    
    return layers.multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, kernel_size=7):
        avg_pool = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(input_feature)
        #print('shape of avg pool and input feature : ', np.shape(input_feature), np.shape(avg_pool))
        max_pool = layers.Lambda(lambda x: K.max(x, axis=1, keepdims=True))(input_feature)
        #print('shape of max pool and input feature : ', np.shape(input_feature), np.shape(max_pool))

        concat = layers.Concatenate(axis=1)([avg_pool, max_pool])
        
        cbam_feature = layers.Conv2D(filters=1,
                      kernel_size=kernel_size,
                      strides=1,
                      padding='same',
                      activation='sigmoid',
                      kernel_initializer='he_normal',
                      use_bias=False)(concat)
        return layers.multiply([input_feature, cbam_feature])
    
def cbam_block(cbam_feature, ratio=2):
        # https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
        cbam_feature = channel_attention(cbam_feature, ratio)
        cbam_feature = spatial_attention(cbam_feature)
        return cbam_feature
    
    
def unet_norm(n_ch,patch_height,patch_width):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = conv2d_block(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    #
    conv2 = conv2d_block(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    #
    conv3 = conv2d_block(pool2, n_filters= 128, kernel_size=3, batchnorm=True)
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([conv2,up1],axis=1)
    
    
    conv4 = conv2d_block(up1, n_filters= 64, kernel_size=3, batchnorm=True)
    
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.concatenate([conv1,up2], axis=1)
    conv5 = conv2d_block(up2, n_filters= 32, kernel_size=3, batchnorm=True)
    
    conv6 = layers.Conv2D(2, (1, 1), activation='relu',padding='same')(conv5)
    
    conv6 = layers.core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

def naive_attn_unet(n_ch,patch_height,patch_width):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = conv2d_block(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    #conv1 = layers.SpatialDropout2D(0.1)(conv1)
    conv1 = spatial_attention(conv1)
    
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    #
    conv2 = conv2d_block(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    #conv2 = layers.SpatialDropout2D(0.1)(conv2)
    conv2 = spatial_attention(conv2)
    
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    #center
    conv3 = conv2d_block(pool2, n_filters= 128, kernel_size=3, batchnorm=True)
    #conv3 = layers.SpatialDropout2D(0.4)(conv3)
    conv3 = spatial_attention(conv3)
    
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([conv2,up1],axis=1)
    conv4 = conv2d_block(up1, n_filters= 64, kernel_size=3, batchnorm=True)
    conv4 = spatial_attention(conv4)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.concatenate([conv1,up2], axis=1)
    conv5 = conv2d_block(up2, n_filters= 32, kernel_size=3, batchnorm=True)
    conv5 = spatial_attention(conv5)
    
    conv6 = layers.Conv2D(2, (1, 1), activation='relu',padding='same')(conv5)
    
    conv6 = layers.core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=adam, loss=dice_coef_loss,metrics=[dice_coef])
    model.summary()
    return model




def cbam_attn_unet(n_ch,patch_height,patch_width):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = conv2d_block(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    conv1 = layers.SpatialDropout2D(0.1)(conv1)
    conv1 = cbam_block(conv1)
    
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    #
    conv2 = conv2d_block(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    conv2 = layers.SpatialDropout2D(0.1)(conv2)
    conv2 = cbam_block(conv2)
    
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    #center
    conv3 = conv2d_block(pool2, n_filters= 128, kernel_size=3, batchnorm=True)
    conv3 = layers.SpatialDropout2D(0.4)(conv3)
    conv3 = cbam_block(conv3)
    
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([conv2,up1],axis=1)
    conv4 = conv2d_block(up1, n_filters= 64, kernel_size=3, batchnorm=True)
    conv4 = cbam_block(conv4)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.concatenate([conv1,up2], axis=1)
    conv5 = conv2d_block(up2, n_filters= 32, kernel_size=3, batchnorm=True)
    conv5 = cbam_block(conv5)
    
    conv6 = layers.Conv2D(2, (1, 1), activation='relu',padding='same')(conv5)
    
    conv6 = layers.core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)


'''
from https://github.com/nabsabraham/focal-tversky-unet/blob/master/newmodels.py

multi-input attn model

'''



def conv2d_block2(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    x = layers.Conv2D(filters=n_filters, kernel_size = (kernel_size, kernel_size), strides = (1,1),kernel_initializer= 'he_normal',
                     padding = 'same')(input_tensor)
    
    if batchnorm ==True:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters=n_filters, kernel_size = (kernel_size, kernel_size), strides = (1,1),kernel_initializer= 'he_normal',
                     padding = 'same')(x)
    if batchnorm ==True:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    return x

def attn_unet(n_ch,patch_height,patch_width,num_classes):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    print('input shape : ', K.int_shape(inputs))
    conv1 = conv2d_block2(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = conv2d_block2(pool1, n_filters= 32, kernel_size=3, batchnorm=True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = conv2d_block2(pool2, n_filters= 64, kernel_size=3, batchnorm=True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = conv2d_block2(pool3, n_filters= 64, kernel_size=3, batchnorm=True)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    #center
    
    center = conv2d_block2(pool4, n_filters= 128, kernel_size=3, batchnorm=True)

    gating1 = UnetGatingSignal(center,True, 'gating01')
    print('\ngating shape : {}, conv4 shape : {}'.format(K.int_shape(gating1), K.int_shape(conv4)))
    
    attn1 = AttnGatingBlock(conv4, gating1, 128 ,'attn01')
    print('\nattn1 shape : {} center shape : {} '.format(K.int_shape(attn1), K.int_shape(center)))
    up1 = layers.concatenate([layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(center), attn1], axis = 1)
    print('\nattn1 shape : {} up1 shape : {}'.format(K.int_shape(attn1), K.int_shape(up1)))
    
    gating2 = UnetGatingSignal(up1, True, 'gating02')

    attn2 = AttnGatingBlock(conv3, gating2, 64,'attn02' )
    up2 = layers.concatenate([layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up1), attn2], axis = 1)
    
    gating3 = UnetGatingSignal(up2, True, 'gating03')
    attn3 = AttnGatingBlock(conv2, gating3, 64,'attn03' )
    up3 = layers.concatenate([layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up2), attn3], axis = 1)
    
    up4 = layers.concatenate([layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up3), conv1], axis = 1)

    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(up4)
    
    conv6 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam ,loss=dice_coef_loss,metrics=[dice_coef])
    #model.compile(optimizer=adam ,loss=hybrid_loss(gamma=3., alpha=.25),metrics=['accuracy'])
    model.summary()
    return model

def small_attn_unet(n_ch,patch_height,patch_width,num_classes):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    print('input shape : ', K.int_shape(inputs))
    conv1 = conv2d_block2(inputs, n_filters= 16, kernel_size=3, batchnorm=True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = conv2d_block2(pool1, n_filters= 32, kernel_size=3, batchnorm=True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = conv2d_block2(pool2, n_filters= 64, kernel_size=3, batchnorm=True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    #center
    center = conv2d_block2(pool3, n_filters= 128, kernel_size=3, batchnorm=True)

    gating1 = UnetGatingSignal(center,True, 'gating01')
    attn1 = AttnGatingBlock(conv3, gating1, 64 ,'attn01')
    up1 = layers.concatenate([layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(center), attn1], axis = 1)
    
    gating2 = UnetGatingSignal(up1, True, 'gating02')
    attn2 = AttnGatingBlock(conv2, gating2, 32,'attn02' )
    up2 = layers.concatenate([layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up1), attn2], axis = 1)
    
    up4 = layers.concatenate([layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up2), conv1], axis = 1)

    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(up4)
    
    conv6 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam ,loss=dice_coef_loss,metrics=[dice_coef])
    #model.compile(optimizer=adam ,loss=hybrid_loss(gamma=3., alpha=.25),metrics=['accuracy'])
    model.summary()
    return model

