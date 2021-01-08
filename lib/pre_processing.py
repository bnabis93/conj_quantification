import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.filters.rank import median
from skimage.morphology import disk
import skimage

import random

# augmentation method
def augmentations(patches,ratio = 0.1):
    '''
    patches
        range [0.0, 1.0]
        
    ratio
        determine augmentation ratio (add noise, add blur)
        
    4 combination
        1. gaussian noise
        2. gaussian blur
        3. gaussian noise + blur
        4. gaussian blur + noise
    
    '''
    sampling_num = int(len(patches) * ratio)
    augmentation_patches = patches.transpose(0,2,3,1)
    sampling_list = random.sample(range(len(patches)), sampling_num)
    s_cnt = 0
    print('[Augmentation function] patches shape : ',np.shape(patches))
    print('[Augmentation function] augmentation patches shape : ',np.shape(augmentation_patches))
    
    for i in range(sampling_num):
        choice_num = random.randint(0,3)
        
        if choice_num ==0:
            #print('[Augmentation loop] aug patches shape : ',np.shape(augmentation_patches[sampling_list[s_cnt]]))
            augmentation_patches[sampling_list[s_cnt]] = add_noise(augmentation_patches[sampling_list[s_cnt]])
            
        elif choice_num ==1:
            #print('[Augmentation loop] aug patches shape : ',np.shape(augmentation_patches[sampling_list[s_cnt]]))
            augmentation_patches[sampling_list[s_cnt]] = add_blur(augmentation_patches[sampling_list[s_cnt]])
            
        elif choice_num ==2:
            #print('[Augmentation loop] aug patches shape : ',np.shape(augmentation_patches[sampling_list[s_cnt]]))
            augmentation_patches[sampling_list[s_cnt]] = add_noise(augmentation_patches[sampling_list[s_cnt]])
            augmentation_patches[sampling_list[s_cnt]] = add_blur(augmentation_patches[sampling_list[s_cnt]])
            
        elif choice_num ==3:
            #print('[Augmentation loop] aug patches shape : ',np.shape(augmentation_patches[sampling_list[s_cnt]]))
            augmentation_patches[sampling_list[s_cnt]] = add_blur(augmentation_patches[sampling_list[s_cnt]])
            augmentation_patches[sampling_list[s_cnt]] = add_noise(augmentation_patches[sampling_list[s_cnt]])
            
        s_cnt +=1
        
    augmentation_patches = augmentation_patches.transpose(0,3,1,2)
    return augmentation_patches
    

def add_noise(patches):
    '''
    patches
        range [0.0,1.0]
        
    not change image's order
    '''
    rand_num = random.randint(1,3)
    mean = 0.0   # some constant
    std = 0.00    # some constant (standard deviation)
    std = std + (rand_num * 0.01)

    noisy_img = patches + (np.random.normal(mean, std, patches.shape))
    noisy_img_clipped = np.clip(noisy_img, 0.0, 1.0)
    
    return noisy_img_clipped
    
def add_blur(patches):
    kernel_size_list = [3,5]
    rand_num = random.randint(0,1)
    
    blur_patches = cv2.GaussianBlur(patches,(kernel_size_list[rand_num],kernel_size_list[rand_num]), cv2.BORDER_DEFAULT)
    blur_patches = np.expand_dims(blur_patches, axis=-1)
    return blur_patches

def add_rotate(mat,masks, angle ,h,w):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    cnt =0 
    height, width = (h,w) # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    angle = random.randint(-angle,angle)
    
    mask_ch = masks.shape[-1]
    rotated_masks = np.zeros((h,w,mask_ch))
    
    #print('mask shape : ', np.shape(rotated_masks))

        
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (width, height))
    for i in range(mask_ch):
        rotated_masks[:,:,i] = cv2.warpAffine(masks[:,:,i], rotation_mat, (width, height))
    
    #print('mat shape : ', np.shape(rotated_mat))
    #print('mask shape : ', np.shape(rotated_masks))
    rotated_mat = np.expand_dims(rotated_mat,-1)
    
    return rotated_mat, rotated_masks
#and using help function

def my_preprocessing(data):
    assert(len(data.shape) ==4) #data has (sample num,ch, height, weith) == 4, tensor data
    assert(data.shape[1] == 3 or data.shape[1] ==1) # data.shape[1] stored ch data, shape[1] == 3 => has RGB ch
    
    # RGB => black & white(gray) ch
    train_imgs = rgb2gray(data)
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    #train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    
    return train_imgs

def my_gray_preprocessing(data):
    #assert(len(data.shape) ==4) #data has (sample num,ch, height, weith) == 4, tensor data
    #assert(data.shape[1] == 3 or data.shape[1] ==1) # data.shape[1] stored ch data, shape[1] == 3 => has RGB ch
    
    # RGB => black & white(gray) ch
    # train_imgs = rgb2gray(data)
    train_imgs = data
    #train_imgs = non_uniform_back(train_imgs,30)
    #train_imgs = dataset_normalized(train_imgs)
    
    train_imgs = clahe_equalized(train_imgs)
    #train_imgs = adjust_gamma(train_imgs, 1.2)
    print('[DEBUG preprocessing] : ', np.shape(train_imgs))
    for i in range(train_imgs.shape[0]):
        train_imgs[i,0] =  (train_imgs[i,0] - np.min(train_imgs[i,0]))/(np.max(train_imgs[i,0]) - np.min(train_imgs[i,0]))
    return train_imgs

# preprocessing function

def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs



#==== histogram equalization

def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1 or imgs.shape[1]==3)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
#adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    #assert (len(imgs.shape)==4)  #4D arrays
    #assert (imgs.shape[1]==1 or imgs.shape[1]==3)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    #assert (len(imgs.shape)==4)  #4D arrays
    #assert (imgs.shape[1]==1 or imgs.shape[1]==3)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    print('[DEBUG] normalize shape : ',np.shape(imgs_normalized) )
    print('[DEBUG] i normalize shape : ',np.shape(imgs_normalized[1]) )

    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


#LUT function : https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f

def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1 or imgs.shape[1]==3)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def non_uniform_back(target_img, blur_size,debug_option = 'off'):
    """non uniform background subtraction.

    Parameters
    ----------
    image : (M, N[, C]) 2D numpy array / (pixel / color space)
        2D numpy array image
    
    blur_size : median blur kernel size
        recommended kernel size is 30~40.
        it it up to your image.
      
    debug_option : for debugging.
        debugging option == 'on'
            show histogram & show data tape
            (default = 'off')
            
    Returns
    -------
        uniform background image 't'
    
    Example
    -------

    """
    img = target_img.copy()
    
    for img_num in range(img.shape[0]):
        nUniImg = img[img_num,0]
        backImg = median(nUniImg,disk(51))
        meanVal = np.mean(nUniImg)
        
        img[img_num,0] = np.divide(nUniImg,backImg) * meanVal
        col,row = nUniImg.shape[0],nUniImg.shape[1]
        for i in range(col):
            for j in range(row):
                if img[img_num,0][i,j] <= 0:
                    img[img_num,0][i,j] = 0
                elif img[img_num,0][i,j] >=255:
                    img[img_num,0][i,j] = 255
                    
        if debug_option == 'on':
            print('data type \n')
            print('Img : {}, backImg : {}, resultImg : {}'.format(nUniImg.dtype,backImg.dtype,t.dtype))

            plt.hist(shadeResult.ravel(),256,[shadeResult.min(),shadeResult.max()])
            plt.title('Histogram')
            plt.show()
            
    return img

def correction_non_uniform_back(img, blur_size,debug_option = 'off'):
    """non uniform background subtraction.
    by this paper: 
    Automated segmentation of the optic nerve head for diagnosis of glaucoma
    
    R.Chrastek, M.Wolf, K.Donath, H.Niemann, D.Paulus, T. Hothorn, B.Lausen, R.Lammer, C.Y.Mardin, G.Michelson
    
    MEDICAL IMAGE ANALYSIS
    
    correction of non-uniform illuminaition.

    Parameters
    ----------
    image : (M, N[, C]) 2D numpy array / (pixel / color space)
        2D numpy array image
    
    blur_size: median blur kernel size
        recommended kernel size is 30~40.
        it it up to your image.
    
    debug_option : for debugging.
        debugging option == 'on'
            show histogram & show data tape
            (default = 'off')
            
    Returns
    -------
        uniform background image 't'
    
    Example
    -------

    """
    if img.shape[2] == None:
        corImg = img
        backImg = cv2.medianBlur(corImg,blur_size)
        maxGrayVal = np.max(backImg)
        
        col,row = corImg.shape[0], corImg.shape[1]
        rList = np.zeros((col,row))

        for i in range(col):
            for j in range(row):
                if backImg[i,j] != 0:
                    rList[i,j] = maxGrayVal / backImg[i,j]
                    
        meanVal =np.mean(backImg,dtype = 'int') 
        c = maxGrayVal - meanVal
        p = np.multiply(img,rList)
        t = p - c
        for i in range(col):
            for j in range(row):
                if t[i][j] >= 255:
                    t[i][j] = 255
                elif t[i][j] <=0:
                    t[i][j] = 0
        
        if debug_option == 'on':
            print('data type \n')
            print('corImg : {}, backImg : {}, resultImg : {}'.format(corImg.dtype,backImg.dtype,t.dtype))
            
            plt.hist(t.ravel(),256,[t.min(),t.max()])
            plt.title('Histogram')
            plt.show()
            
        return t
        
    elif img.shape[2] == 3:
        corImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        backImg = cv2.medianBlur(corImg,blur_size)
        maxGrayVal = np.max(backImg)
        
        col,row = corImg.shape[0], corImg.shape[1]
        rList = np.zeros((col,row))

        for i in range(col):
            for j in range(row):
                if backImg[i,j] != 0:
                    rList[i,j] = maxGrayVal / backImg[i,j]
                    
        meanVal =np.mean(backImg,dtype = 'int') 
        c = maxGrayVal - meanVal
        p = np.multiply(img,rList)
        t = p - c
        for i in range(col):
            for j in range(row):
                if t[i][j] >= 255:
                    t[i][j] = 255
                elif t[i][j] <=0:
                    t[i][j] = 0
        if debug_option == 'on':
            print('data type \n')
            print('corImg : {}, backImg : {}, resultImg : {}'.format(corImg.dtype,backImg.dtype,t.dtype))

            plt.hist(t.ravel(),256,[t.min(),t.max()])
            plt.title('Histogram')
            plt.show()            
        return t
    else:
        print('Color Space Error!')