import os, shutil
import sys
import h5py
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, './lib/')
from help_functions import write_hdf5
import configparser


config = configparser.RawConfigParser()
config.read('pre_configuration.txt')
mode = config.get('setting', 'mode')
original_img_train_path = config.get('path','original')
ground_truth_img_train_path = config.get('path','ground')
border_masks_imgs_train_path = config.get('path','mask')
size_mode = config.get('setting', 'size_mode')
resize_constant = config.get('setting','resize_constant')
resize_constant = float(resize_constant)

dataset_dir_path = config.get('path','save_path')
# explicit data path 

if mode == 'DRIVE' :
    #DRIVE training data path
    original_img_train_path = './data/DRIVE/DRIVE/training/images/'
    ground_truth_img_train_path = './data/DRIVE/DRIVE/training/1st_manual/'
    border_masks_imgs_train_path = './data/DRIVE/DRIVE/training/mask/'

    #DRIVE test data path
    original_img_test_path = './data/DRIVE/DRIVE/test/images/'
    ground_truth_img_test_path = './data/DRIVE/DRIVE/test/1st_manual/'
    border_masks_imgs_test_path = './data/DRIVE/DRIVE/test/mask/'

    dataset_dir_path = './hdf5_datasets_training_testing/DRIVE/'
    
    num_imgs = 20
    channels = 3
    img_height = 584
    img_width = 565

elif mode == 'COMB_DRIVE': 
    original_img_train_path = './data/DRIVE/DRIVE/add_test_train/images/'
    ground_truth_img_train_path = './data/DRIVE/DRIVE/add_test_train/1st_manual/'
    border_masks_imgs_train_path = './data/DRIVE/DRIVE/add_test_train/mask/'
    dataset_dir_path = './hdf5_datasets_training_testing/COMB_DRIVE/'
    
    num_imgs = 40
    channels = 3
    img_height = 584
    img_width = 565
    
elif mode =='STARE':
    original_img_train_path = './data/STARE/train/'
    ground_truth_img_train_path = './data/STARE/ground_truth/'
    dataset_dir_path = './hdf5_datasets_training_testing/STARE/'

    
    num_imgs = 20
    channels = 3
    
    img_height = 605
    img_width = 700

elif mode =='CHASE':
    original_img_train_path = './data/CHASE_DB/train/'
    ground_truth_img_train_path = './data/CHASE_DB/1st_ground_truth/'
    dataset_dir_path = './hdf5_datasets_training_testing/CHASE_DB/'

    num_imgs = 28
    channels = 3
    
    img_height = 960
    img_width = 999
    
    
elif mode =='resize_STARE':
    original_img_train_path = './data/STARE/train/'
    ground_truth_img_train_path = './data/STARE/ground_truth/'
    dataset_dir_path = './hdf5_datasets_training_testing/STARE/'

    
    num_imgs = 20
    channels = 3
    
    img_height = 605
    img_width = 700
    resized_img_height = 584
    resized_img_width = 565
    
elif mode == 'Data_pool':
    original_img_train_path = './data/data_pool/train/'
    ground_truth_img_train_path = './data/data_pool/ground_truth/'
    dataset_dir_path = './hdf5_datasets_training_testing/data_pool/'
    
    num_imgs = 60
    channels = 3
    img_height = 584
    img_width = 565
    
elif mode == 'HRF':
    path, dirs, files = next(os.walk(original_img_train_path))
    num_imgs = len(files)
    img = cv2.imread(original_img_train_path + files[0])
    img_shape = np.shape(img)
    
    if len(img_shape) == 3:
        #ch = 3 or 1
        channels = img_shape[2]
        img_height = img_shape[0]
        img_width = img_shape[1]
        
    elif len(img_shape) == 2:
        # ch = none
        img_height = img_shape[0]
        img_width = img_shape[1]

elif mode == 'fixed_conj':
    path, dirs, files = next(os.walk(original_img_train_path))
    num_imgs = len(files)
    img = cv2.imread(original_img_train_path + files[0])
    img_shape = np.shape(img)
    
    if len(img_shape) == 3:
        #ch = 3 or 1
        channels = img_shape[2]
        img_height = img_shape[0]
        img_width = img_shape[1]
        if size_mode == 'resize':
            resize_height = int(img_height * resize_constant) 
            resize_width = int(img_width * resize_constant)
        
    elif len(img_shape) == 2:
        # ch = none
        img_height = img_shape[0]
        img_width = img_shape[1]
        if size_mode == 'resize':
            resize_height = int(img_height * resize_constant) 
            resize_width = int(img_width * resize_constant)
        
    
else:
    raise ValueError('mode value error')


'''
#make dir
if os.path.isdir(train_dir) == False:
    os.mkdir(train_dir)
else:
    print('already exist the folder in this path : {}'.format(train_dir))
    
if os.path.isdir(valid_dir) == False:
    os.mkdir(valid_dir)
else:
    print('already exist the folder in this path : {}'.format(valid_dir))

if os.path.isdir(test_dir) == False:
    os.mkdir(test_dir)
else:
    print('already exist the folder in this path : {}'.format(test_dir))
'''    
#first, you should masking the each folder's image.
'''


DIR TREE

base_path  - train datasets
          |
          |- validation datasets
          |
          |- test datasets
          
          
original_path  - train 
              |
              |- test
              |
              |- masked train
              |
              |- masked test
          
'''
def get_datasets_resize_STARE(imgs_dir,groundTruth_dir):
    
    imgs = np.empty((num_imgs,resized_img_height,resized_img_width,channels))
    groundTruth = np.empty((num_imgs,resized_img_height,resized_img_width))
    
    print('img dir : ',imgs_dir)
    for count, filename in enumerate(sorted(os.listdir(imgs_dir)), start=0):
        
        print ("original image: " +filename)
        img = cv2.imread(imgs_dir+filename)
        #print(np.shape(img))
        img = cv2.resize(img, (resized_img_width, resized_img_height))
        
        #print('resize', np.shape(img))
        imgs[count] = np.asarray(img)
        print('file name : ',imgs_dir+ filename)
    
    print('ground truth dir : ', groundTruth_dir)
    for count, filename in enumerate(sorted(os.listdir(groundTruth_dir)), start=0):
        groundTruth_name = filename
        print ("ground truth name: " + groundTruth_name)
        g_truth = cv2.imread(groundTruth_dir + groundTruth_name)
        g_truth = g_truth[:,:,0]
        g_truth = cv2.resize(g_truth, (resized_img_width, resized_img_height))
        groundTruth[count] = np.asarray(g_truth)

          
    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))

    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (num_imgs,channels,resized_img_height,resized_img_width))
    groundTruth = np.reshape(groundTruth,(num_imgs,1,resized_img_height,resized_img_width))
    
    assert(groundTruth.shape == (num_imgs,1,resized_img_height,resized_img_width))
    
    return imgs, groundTruth

def get_datasets_data_pool(imgs_dir,groundTruth_dir):
    
    imgs = np.empty((num_imgs,img_height,img_width,channels))
    groundTruth = np.empty((num_imgs,img_height,img_width))
    
    print('img dir : ',imgs_dir)
    for count, filename in enumerate(sorted(os.listdir(imgs_dir)), start=0):
        if filename.startswith('.ipynb') ==False:
            print ("original image: " +filename)
            img = cv2.imread(imgs_dir+filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs[count] = np.asarray(img)
            print('file name : ',imgs_dir+ filename)
    
    print('ground truth dir : ', groundTruth_dir)
    for count, filename in enumerate(sorted(os.listdir(groundTruth_dir)), start=0):
        if filename.startswith('.ipynb') ==False:
            groundTruth_name = filename
            print ("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[count] = np.asarray(g_truth)
            print ("imgs max: " +str(np.max(groundTruth[count])))
            print ("imgs min: " +str(np.min(groundTruth[count])))

          
    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    
    print ("imgs max: " +str(np.max(groundTruth)))
    print ("imgs min: " +str(np.min(groundTruth)))
    
    assert(np.max(groundTruth)==255 and np.min(groundTruth) == 0)
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (num_imgs,channels,img_height,img_width))
    groundTruth = np.reshape(groundTruth,(num_imgs,1,img_height,img_width))
    
    assert(groundTruth.shape == (num_imgs,1,img_height,img_width))
    
    return imgs, groundTruth



def get_datasets2(imgs_dir,groundTruth_dir,borderMasks_dir = False ,train_test="null"):
    #imgs = np.empty((num_imgs,img_height,img_width,channels))
    #groundTruth = np.empty((num_imgs,img_height,img_width))
    if size_mode =='original':
        imgs = np.empty((num_imgs,img_height,img_width,channels))
        groundTruth = np.empty((num_imgs,img_height,img_width)) # RGB ch labels (multi segmentation)
    elif size_mode == 'resize':
        resized_imgs = np.empty((num_imgs, resize_height, resize_width, channels))
        resized_groundTruth = np.empty((num_imgs,resize_height,resize_width))
                                       
    print('img dir : ',imgs_dir)
    for count, filename in enumerate(sorted(os.listdir(imgs_dir)), start=0):
        print ("original image: " +filename)
        #img = Image.open(imgs_dir+filename)
        #imgs[count] = np.asarray(img)
        #print('file name : ',imgs_dir+ filename)
                                       
        if size_mode == 'original':
            img = Image.open(imgs_dir+filename)
            print('img shape : ',np.shape(img))
            imgs[count] = np.asarray(img)

        elif size_mode == 'resize':
            img = Image.open(imgs_dir+filename)
            img = np.asarray(img)
            resized_img = cv2.resize(img, (resize_width, resize_height), interpolation = cv2.INTER_LANCZOS4)
            resized_imgs[count] = resized_img
                                       
    
    print('ground truth dir : ', groundTruth_dir)
    for count, filename in enumerate(sorted(os.listdir(groundTruth_dir)), start=0):
        groundTruth_name = filename
        print ("ground truth name: " + groundTruth_name)
        if size_mode =='original':
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            if len(np.shape(g_truth)) !=2:
                g_truth = cv2.imread(groundTruth_dir+groundTruth_name)
                g_truth = g_truth[:,:,0]
            groundTruth[count] = np.asarray(g_truth)

        elif size_mode == 'resize':
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            if len(np.shape(g_truth)) !=2:
                g_truth = cv2.imread(groundTruth_dir+groundTruth_name)
                g_truth = g_truth[:,:,0]
            g_truth = np.asarray(g_truth)
            resized_gTruth = cv2.resize(g_truth, (resize_width, resize_height), interpolation = cv2.INTER_LANCZOS4)
            resized_groundTruth[count] = resized_gTruth
                              
    if size_mode == 'original':
        print ("imgs max: " +str(np.max(imgs)))
        print ("imgs min: " +str(np.min(imgs)))
        assert(np.max(groundTruth)==255)
        assert(np.min(groundTruth)==0)
        print ("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
        #reshaping for my standard tensors
        imgs = np.transpose(imgs,(0,3,1,2))
        assert(imgs.shape == (num_imgs,channels,img_height,img_width))
        groundTruth = np.reshape(groundTruth,(num_imgs,1,img_height,img_width))
        assert(groundTruth.shape == (num_imgs,1,img_height,img_width))

        return imgs, groundTruth
                                           
    elif size_mode == 'resize':
        print('resized \n')
        print ("imgs max: " +str(np.max(resized_imgs)))
        print ("imgs min: " +str(np.min(resized_imgs)))

        resized_imgs = np.transpose(resized_imgs,(0,3,1,2)) #num c h w
        resized_groundTruth= np.reshape(resized_groundTruth,(num_imgs,1,resize_height,resize_width))
        #assert(imgs.shape == (num_imgs,channels,img_height,img_width))
        #groundTruth = np.reshape(groundTruth,(num_imgs,3,img_height,img_width))
        assert(resized_imgs.shape == (num_imgs,channels,resize_height,resize_width))

        assert(resized_groundTruth.shape == (num_imgs,1,resize_height,resize_width))
        print('Done!')

        return resized_imgs, resized_groundTruth



if mode == 'DRIVE' :

    imgs_train,groundTruth_train,border_masks_train =get_datasets2(original_img_train_path,ground_truth_img_train_path,border_masks_imgs_train_path,"train")
    print ("saving train datasets")
    write_hdf5(imgs_train, dataset_dir_path + "DRIVE_dataset_imgs_train.hdf5")
    write_hdf5(groundTruth_train, dataset_dir_path + "DRIVE_dataset_groundTruth_train.hdf5")
    write_hdf5(border_masks_train,dataset_dir_path + "DRIVE_dataset_borderMasks_train.hdf5")

    print('[Training Done]')

    #getting the testing datasets
    imgs_test, groundTruth_test, border_masks_test = get_datasets2(original_img_test_path,ground_truth_img_test_path,border_masks_imgs_test_path,"test")

    write_hdf5(imgs_test,dataset_dir_path + "DRIVE_dataset_imgs_test.hdf5")
    write_hdf5(groundTruth_test, dataset_dir_path + "DRIVE_dataset_groundTruth_test.hdf5")
    write_hdf5(border_masks_test,dataset_dir_path + "DRIVE_dataset_borderMasks_test.hdf5")

elif mode == 'STARE' :

    imgs_train,groundTruth_train =get_datasets2(original_img_train_path,ground_truth_img_train_path)
    write_hdf5(imgs_train, dataset_dir_path + "STARE_dataset_imgs_train.hdf5")
    write_hdf5(groundTruth_train, dataset_dir_path + "STARE_dataset_groundTruth_train.hdf5")

elif mode == 'CHASE' :

    imgs_train,groundTruth_train =get_datasets2(original_img_train_path,ground_truth_img_train_path)
    write_hdf5(imgs_train, dataset_dir_path + "CHASE_dataset_imgs_train.hdf5")
    write_hdf5(groundTruth_train, dataset_dir_path + "CHASE_dataset_groundTruth_train.hdf5")
    
elif mode == 'COMB_DRIVE': 
    imgs_train,groundTruth_train,border_masks_train =get_datasets2(original_img_train_path,ground_truth_img_train_path,border_masks_imgs_train_path,"train")
    write_hdf5(imgs_train, dataset_dir_path + "DRIVE_dataset_imgs_train_test.hdf5")
    write_hdf5(groundTruth_train, dataset_dir_path + "DRIVE_dataset_groundTruth_train_test.hdf5")
    write_hdf5(border_masks_train,dataset_dir_path + "DRIVE_dataset_borderMasks_train_test.hdf5")
    
elif mode =='resize_STARE':
    imgs_train,groundTruth_train = get_datasets_resize_STARE(original_img_train_path,ground_truth_img_train_path)
    write_hdf5(imgs_train, dataset_dir_path + "STARE_dataset_imgs_resize_train.hdf5")
    write_hdf5(groundTruth_train, dataset_dir_path + "STARE_dataset_groundTruth_resize_train.hdf5")
    
elif mode =='Data_pool':
    imgs_train,groundTruth_train = get_datasets_data_pool(original_img_train_path,ground_truth_img_train_path)
    write_hdf5(imgs_train, dataset_dir_path + "Data_pool_train.hdf5")
    write_hdf5(groundTruth_train, dataset_dir_path + "Data_pool_ground_truth.hdf5")

elif mode =='HRF':
    imgs_train,groundTruth_train,border_masks_train =get_datasets2(original_img_train_path,ground_truth_img_train_path,border_masks_imgs_train_path,'train')
    write_hdf5(imgs_train, dataset_dir_path + "HRF_dataset_imgs_train.hdf5")
    write_hdf5(groundTruth_train, dataset_dir_path + "HRF_dataset_groundTruth_train.hdf5")
    write_hdf5(border_masks_train,dataset_dir_path + "HRF_dataset_borderMasks_train.hdf5")

elif mode =='fixed_conj':
    if size_mode == 'original':
        imgs_train,groundTruth_train =get_datasets2(original_img_train_path,ground_truth_img_train_path)
        write_hdf5(imgs_train, dataset_dir_path + "fixed_conj_dataset_imgs_train.hdf5")
        write_hdf5(groundTruth_train, dataset_dir_path + "fixed_conj_dataset_groundTruth_train.hdf5")
        #write_hdf5(border_masks_train,dataset_dir_path+what_data + "_borderMasks_train.hdf5")    
    elif size_mode == 'resize':
        imgs_train,groundTruth_train =get_datasets2(original_img_train_path,ground_truth_img_train_path)
        write_hdf5(imgs_train, dataset_dir_path+"_conj_resized_train.hdf5")
        write_hdf5(groundTruth_train, dataset_dir_path + "_conj_resized_groundTruth_train.hdf5")
                                       
    #imgs_train,groundTruth_train  =get_datasets2(original_img_train_path,ground_truth_img_train_path)
    #write_hdf5(imgs_train, dataset_dir_path + "fixed_conj_dataset_imgs_train.hdf5")
    #write_hdf5(groundTruth_train, dataset_dir_path + "fixed_conj_dataset_groundTruth_train.hdf5")
    

    #write_hdf5(border_masks_train,dataset_dir_path + "HRF_dataset_borderMasks_train.hdf5")
'''

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,train_test="null"):
    
    imgs = np.empty((num_imgs,img_height,img_width,channels))
    groundTruth = np.empty((num_imgs,img_height,img_width))
    border_masks = np.empty((num_imgs,img_height,img_width))
    
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print ("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            print('file name : ',imgs_dir+ files[i])
            
            #corresponding ground truth
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            print ("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:2] + "_training_mask.gif"
            elif train_test=="test":
                border_masks_name = files[i][0:2] + "_test_mask.gif"
            else:
                print ("specify if train or test!!")
                exit()
            print ("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)
            print(b_mask,'\n')
            
    print(border_masks.shape)
          
    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print ("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (num_imgs,channels,img_height,img_width))
    groundTruth = np.reshape(groundTruth,(num_imgs,1,img_height,img_width))
    border_masks = np.reshape(border_masks,(num_imgs,1,img_height,img_width))
    assert(groundTruth.shape == (num_imgs,1,img_height,img_width))
    assert(border_masks.shape == (num_imgs,1,img_height,img_width))
    return imgs, groundTruth, border_masks
    

if not os.path.exists(dataset_dir_path):
    os.makedirs(dataset_dir_path)
    
#getting the training datasets
imgs_train, groundTruth_train, border_masks_train = get_datasets(original_img_train_path,ground_truth_img_train_path,border_masks_imgs_train_path,"train")
print ("saving train datasets")
write_hdf5(imgs_train, dataset_dir_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_dir_path + "DRIVE_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_dir_path + "DRIVE_dataset_borderMasks_train.hdf5")

#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test = get_datasets(original_img_test_path,ground_truth_img_test_path,border_masks_imgs_test_path,"test")
write_hdf5(imgs_test,dataset_dir_path + "DRIVE_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_dir_path + "DRIVE_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_dir_path + "DRIVE_dataset_borderMasks_test.hdf5")
'''