import cv2
import sklearn 
import numpy as np
import os
import matplotlib.pyplot as plt

import scipy.ndimage as ndi

from PIL import Image

from skimage.data import binary_blobs
from skimage.morphology import medial_axis, skeletonize, skeletonize_3d,thin
from skimage import data
from skimage.util import invert
from skimage.morphology import remove_small_objects



def get_skeleton(thr_img):
    '''
    parameters
        thr_img : (boolean image)
            thresholded image
            
    return 
        skeleton image
    
    '''
    if np.max(thr_img) == 255:
        ret,thresh = cv2.threshold(thr_img,127,1,cv2.THRESH_BINARY)

    return skeletonize(thresh)


'''
How to get the branch points from the image
    example
        test_img = skeleton
        branch_img = branches(test_img)
        temp = extract_branch_cordi(branch_img)

    => get skeleton image -> branches function -> extract_branch_cordi function
'''


def _neighbors_conv(image):
    """
    Counts the neighbor pixels for each pixel of an image:
            x = [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]
            ]
            _neighbors(x)
            [
                [0, 3, 0],
                [3, 4, 3],
                [0, 3, 0]
            ]
    :type image: numpy.ndarray
    :param image: A two-or-three dimensional image
    :return: neighbor pixels for each pixel of an image
    """
    image = image.astype(np.int)
    k = np.array([[1,1,1],[1,0,1],[1,1,1]])
    neighborhood_count = ndi.convolve(image,k, mode='constant', cval=1)
    neighborhood_count[~image.astype(np.bool)] = 0
    return neighborhood_count

def branches(image):
    """
    Returns the nodes in between edges
    Parameters
    ----------
    image : binary (M, N) ndarray
    Returns
    -------
    out : ndarray of bools
        image.
    """
    return _neighbors_conv(image) > 2

def extract_branch_cordi(branch_img):
    row, col = branch_img.shape
    #print(row, col)
    branch_list = []
    for i in range(0,row):
        for j in range(0,col):
            if branch_img[i,j] == True:
                branch_list.append([i,j])
                
    return branch_list

def draw_circle_branch_points_on_oriImg(ori_img, branch_points):
    '''
    parameters
        ori_img 
            original img, 
        branch_points
            branch points
        
    return 
        copy_img
            draw the circle on branch points
    '''
    copy_img = ori_img.copy()
    for i in range(len(branch_points)):
        tR,tC = branch_points[i]
        tR,tC = tC,tR
        tempRC = tuple([tR,tC])
        cv2.circle(copy_img,tempRC, 1,(0,255,0))
        
    return copy_img
    
def draw_circle_branch_points_on_skelImg(skel_img, branch_points):
    '''
    parameters
        skel_img 
            skeleton img from the vessel segmentaiton map 
        branch_points
            branch points
        
    return 
        copy_img
            draw the circle on branch points
    '''
    copy_skelImg = skel_img.copy()
    I = np.dstack([copy_skelImg.astype(np.uint8) * 255, copy_skelImg.astype(np.uint8) * 255, copy_skelImg.astype(np.uint8)* 255])
    for i in range(len(branch_points)):
        tR,tC = branch_points[i]
        tR,tC = tC,tR
        tempRC = tuple([tR,tC])
        cv2.circle(I,tempRC, 1,(0,255,0) )
        
    return I

def remove_branch_points(skel_img, branch_imgs):
    '''
    parameters
        skel_img
            skeleton image from segemnted image
            
        branch_img
            branch img (not branch points)
    
    return
        removedBranchImg
            divide from branch points
    '''
    
    skeleton = skel_img.copy()
    noBranch = np.invert(branch_imgs)
    removedBranchImg = np.bitwise_and(skeleton, noBranch)
    
    return removedBranchImg

def post_processing_skelImg(skel_img, connectivity_thr = 20):
    '''
    parameters
        skel_img
        connectivity_thr
        
    return 
        post_processed_skeleton
    '''
    skeleton = skel_img.copy()
    post_processed_skeleton = remove_small_objects(skeleton, connectivity= connectivity_thr)
    return post_processed_skeleton
