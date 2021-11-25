import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt 
from skimage.filters.rank import entropy
from skimage.morphology import disk,closing
from sklearn.ensemble import RandomForestClassifier
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import pdb 
from skimage import exposure
from  skimage.filters import sobel_v
from skimage import feature
from PIL import Image
import argparse

colors = [(255,128,0),(102,255,255),(0,204,204),(255,255,0),(0,102,204),(255,0,0) ]


def show_line(line,img,color = (255,0,0)):
    '''
    mathod for showing a specified line on the image

    Parameters
    ----------
    line : np array
        line to display on the image.
    img : np array
        image where to show the line on.

    Returns
    -------
    None.

    '''

    for i in range(len(line)):
        x,y = line[i]
        cv2.circle(img, (x,y), radius=0, color=color, thickness=5)


def get_lines_intersection(m1,m2,b1,b2,img):
    '''
    method for computing the intersection of two lines

    Parameters
    ----------
    
    m1: angular coefficient for line 1
    m2: angular coefficient for line 2
    b1: intercept for line 1 
    b2:intercept for line 2
    img: image where to display the intersection point on
    Returns
    -------
    
    (xi,yi): point of intersection between the two lines

    '''
        
    #get intersection of above lines for getting thetorso location
    xi = int((b1-b2) / (m2-m1))
    yi = int(m1 * xi + b1)
    cv2.circle(img, (int(xi),int(yi)), radius=0, color=(255, 0,0), thickness=5)
    return (xi,yi)


#this method ahs been left incomplete for lack of time 
#my idea wa the following: 
#By using trivial geometry  over deteted keypoints, I wanted to:
    #
    #-detect the torso by crossing the lines linking (left elbow,right shoulder) and (right elbow, left shoulder)
    #- once got the intersection, I would just move the shoulder line below in order to distinguishing arms and torso
    #- how to split them? maybe by using the same method used for the hip line
def get_upper_body_measures(keypoint_dict,width,img):
    
    
    copy = img.copy()
    
    #get lines to display
    _,left_arm_to_right_shld,m1,b1 = get_line(keypoint_dict['l_e'],keypoint_dict['r_s'],width)
    _,right_arm_to_left_shld,m2,b2 = get_line(keypoint_dict['l_s'],keypoint_dict['r_e'],width)
    _,left_shoulder_to_left_elb,_,_ = get_line(keypoint_dict['l_s'],keypoint_dict['l_e'],width)
    _,right_shoulder_to_right_elb,_,_ = get_line(keypoint_dict['r_s'],keypoint_dict['r_e'],width)
    _,left_hip_to_left_shld,m3,b3 = get_line(keypoint_dict['l_s'],keypoint_dict['l_hip'],width)
    _,rigth_hip_to_ight_shld,m4,b4 = get_line(keypoint_dict['r_hip'],keypoint_dict['r_s'],width)
    full_shoulder_line,shoulder_line,_,_ = get_line(keypoint_dict['l_s'],keypoint_dict['r_s'],width)
    
    #display lines
    show_line(left_arm_to_right_shld,img,colors[0])
    show_line(left_hip_to_left_shld,img,colors[1])
    show_line(left_shoulder_to_left_elb,img,colors[2])
    show_line(right_shoulder_to_right_elb,img,colors[3])
    show_line(rigth_hip_to_ight_shld,img,colors[4])
    show_line(right_arm_to_left_shld,img,colors[5])
    

    #get intersection
    xi,yi = get_lines_intersection(m1,m2,b1,b2,img)

    #adjust shoulder line to get the approximated torso line
    y_half = full_shoulder_line[int(len(full_shoulder_line)/2 -1 )][1]
    torso_line = []
    for i in range(len(full_shoulder_line)):
        x,y = full_shoulder_line[i]
        cv2.circle(img, (x,y+ abs(yi-y_half)), radius=0, color=(255, 0,0), thickness=5)
        torso_line.append([x,y+ abs(yi-y_half)])
        


    get_arm_boundaries(torso_line,cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY))

def get_arm_boundaries(torso_line,img):
    '''
    method for finding the boundaries of each model's arm.
    I equalised the image's histogram for better detecting edges. I then used canny to get edges and used closing (morphological op)
    to fill possible holes.

    Parameters
    ----------

    torso_line : np array
        line linking the arms
    img : np array
        the image to filter with canny.

    Returns
    -------
    None.

    '''
    img = exposure.equalize_adapthist(img, clip_limit=0.03)
    edges1 = feature.canny(img)
    footprint = disk(6) 
    edges1 = closing(edges1, footprint)
    edges1 = np.where(edges1, 255, 0).astype("uint8")

    
    for i in range(0,len(torso_line)):
        
        x,y = torso_line[i]
        if(edges1[y,x] > 0):
            min_p = (x,y)
            break

    
    for i in range(len(torso_line)-1,0,-1):
        x,y = torso_line[i]
        
        if(edges1[y,x] > 0):
           max_p = (x,y)
           break
    edges1 = cv2.cvtColor(np.array(edges1), cv2.COLOR_GRAY2RGB)  
    cv2.circle(edges1, (min_p[0],min_p[1]), radius=0, color=(0, 255,0), thickness=10)
    cv2.circle(edges1, (max_p[0],max_p[1]), radius=0, color=(0, 255, 0), thickness=10)
    im = Image.fromarray(edges1)
    im.show()

def get_measures(keypoint_dict):
    '''
    method for computing the mean arm's length and mean leg's length

    Parameters
    ----------
    keypoint_dict : dictionary
        dictionary containing the keypoints.

    Returns
    -------
    mean_leg_length : float
        the mean leg's length in pixel.
    mean_arm_length : float
        the mean arm's length in pixel.

    '''

    
    #get arms measures
    dist_s_e_l,dist_s_e_r = abs(np.linalg.norm(keypoint_dict['l_s']-keypoint_dict['l_e'])), \
            abs(np.linalg.norm(keypoint_dict['r_s']-keypoint_dict['r_e']))
            
    dist_h_e_l,dist_h_e_r = abs(np.linalg.norm(keypoint_dict['l_h']-keypoint_dict['l_e'])), \
            abs(np.linalg.norm(keypoint_dict['r_h']-keypoint_dict['r_e']))
            
    left_arm_length = dist_s_e_l + dist_h_e_l
    right_arm_length = dist_s_e_r + dist_h_e_r
    mean_arm_length = (left_arm_length + right_arm_length)/2
    

    #get legs measures
    dist_k_l_l,dist_k_l_r = abs(np.linalg.norm(keypoint_dict['l_hip']-keypoint_dict['l_k'])), \
            abs(np.linalg.norm(keypoint_dict['r_hip']-keypoint_dict['r_k']))
            
    dist_k_f_l,dist_k_f_r = abs(np.linalg.norm(keypoint_dict['l_k']-keypoint_dict['l_f'])), \
            abs(np.linalg.norm(keypoint_dict['r_k']-keypoint_dict['r_f']))
            
    left_leg_length = dist_k_l_l + dist_k_f_l
    right_leg_length = dist_k_l_r + dist_k_f_r
    mean_leg_length = (left_leg_length + right_leg_length)/2
    
    return mean_leg_length,mean_arm_length


def create_gabor_kernels():
    '''
    method for creating the gabor filters

    Returns
    -------
    kernels : np array
        a set of gabor filters.

    '''
    # prepare filter bank kernels
    sigma = 1
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
    return kernels


def compute_feats(image, kernels):
    '''
    
    method which compute features by convolving gabor kernels with the specified portion of the image
    Parameters
    ----------
    image : np array
        cropped image we want to analyse with gabor filter
    kernels : np array 
        the gabor filters we use for extracting features

    Returns
    -------
    feats : the extracted features
    

    '''
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def get_line(p1,p2,width):
    '''
    we get the line intersecting two points 

    Parameters
    ----------
    p1 : np.array
         the first keypoint
    p2 :  np.array
        the second keypoint
    width: the image's width

    Returns
    -------
    full_line: the full line covering the two specified points
    linking_line: the line linking the two specified points
    m: the angular coefficient of the line
    b: the intercept

    '''
    
    
    # we look for the left most point of the two
    if(p1[0] <= p2[0]):   
        x1,y1 = p1
        x2,y2 = p2
        
    else:
        x1,y1 = p2
        x2,y2 = p1
    
    #we compute the linking line
    m = (y1-y2)/(x1-x2)
    b =  (x1*y2 - x2*y1)/(x1-x2)
    
    full_line,linking_line = [],[]
    
    for x in range(width):
        y = int(m*x + b)
        if(x >= x1 and x <= x2):
            linking_line.append([x,y])

        
        full_line.append([x,y])
    return full_line,linking_line,m,b


def get_bg_data(left_hand_point,right_hand_point, full_line,img,entropy,kernels):
    '''
    method for extracting background data from the waist line. We cansider as bg what is outside each hand of the model

    Parameters
    ----------
    left_hand_point : np.array
         the keypoint for the left hand
    right_hand_point :  np.array
        the keypoint for the right hand
    full_line : np.array
        the waist line covering the full image width 
    img : np.array
        the numpy image.
    entropy : np.array
        numpy array containing the entropy for the specified image
    kernels : np.array
       array of gabor filters

    Returns
    -------
    training dataset,its labels and the test set

    '''
    
    train_X,train_y,test_X = [],[],[]
    for i in range(3,len(full_line)-4):
        #we get the coordinates of the point along the hip line
        x,y = full_line[i]
        
        #we compute the features for the spicified window (7,7) surrounding the current hip point
        feats = compute_feats( np.mean(img[y-3:y+4,x-3:x+4,:],axis=2) ,kernels)
        
        #we store features and relative labels for the training dataset
        if( (x < left_hand_point[0] or x > right_hand_point[0])):
            train_X.append([img[y,x,c] for c in range(img.shape[2])])

            train_X[-1].append(entropy[y,x])
            train_y.append(0)
            
            for f in feats:
                train_X[-1].append(f[0])
                train_X[-1].append(f[1])
            
        #we store features for the test dataset
        test_X.append([img[y,x,c] for c in range(img.shape[2])])
        test_X[-1].append(entropy[y,x])
        
        for f in feats:
            test_X[-1].append(f[0])
            test_X[-1].append(f[1])

    return train_X,train_y,test_X

def get_fg_data(left_hip,right_hip, hip_line,img,entropy,kernels):
    '''
    method for extracting foreground data from the waist line. We cansider as fg what is between the the left hip and the right hip

    Parameters
    ----------
    left_hand_point : np.array
         the keypoint for the left hand
    right_hand_point :  np.array
        the keypoint for the right hand
    full_line : np.array
        the waist line covering the full image width 
    img : np.array
        the numpy image.
    entropy : np.array
        numpy array containing the entropy for the specified image
    kernels : np.array
       array of gabor filters

    Returns
    -------
    training dataset,its labels and the test set

    '''  
    train_X,train_y = [],[]
    for i in range(len(hip_line)):
        
        #we get the coordinates of the point along the hip line
        x,y = hip_line[i]
        #we compute the features for the spicified window (7,7) surrounding the current hip point
        feats = compute_feats( np.mean(img[y-3:y+4,x-3:x+4,:],axis=2) ,kernels)
        
        #we store features and relative labels for the training dataset
        train_X.append([img[y,x,c] for c in range(img.shape[2])])
        train_X[-1].append(entropy[y,x])
        train_y.append(1)
        
        
        for f in feats:
            train_X[-1].append(f[0])
            train_X[-1].append(f[1])
            
    return train_X,train_y





def get_data(keypoint_dict,full_line,hip_line, img,entropy, kernels):
    '''
    method for creating the dataset which will be used for segmenting the waist line

    Parameters
    ----------
    keypoint_dict : dict
        dictionary containing the retrieved keypoints from the image
    full_line : list
        list of points representing the full line covering the waist
    waist_line : list
        list of points covering only the waist
    img : np.array
        original rgb/yuv image
    entropy : np.array
        numpy array containing the entropy for the specified image
    kernels : np.array
       array of gabor filters

    Returns
    -------
    TYPE
        training dataset,its labels and the test set

    '''
    # we define what we mean as background
    X_bg,y_bg,test_X = get_bg_data(keypoint_dict['l_h'],keypoint_dict['r_h'],full_line,img,entropy,kernels)
    # we define what we mean as foreground
    X_fg,y_fg = get_fg_data(keypoint_dict['l_hip'],keypoint_dict['r_hip'],hip_line,img,entropy,kernels)
    
    #convert to numpy 
    X_bg,y_bg,X_fg,y_fg = np.array(X_bg),np.array(y_bg),np.array(X_fg),np.array(y_fg)

    # concatenate bg and fg into training dataset
    X = np.concatenate((X_bg,X_fg), axis = 0)
    y = np.concatenate((y_bg,y_fg), axis = 0)
    

    return X,y,np.array(test_X)

def predict_pixel_values(keypoint_dict,train_X,train_y,test_X):
    '''
    method for predicting each pixel along the hip line 

    Parameters
    ----------
    keypoint_dict : dict
        dictionar containing keypoints.
    train_X : np array
        training data.
    train_y : np array
        labels for training data.
    test_X : np array
        test data.

    Returns
    -------
    
        list of prediction for each point along the hip line.

    '''
    clf = RandomForestClassifier(max_depth=10,random_state=0)
    clf.fit( np.array(train_X),train_y.reshape(-1,1))
    return clf.predict_proba(test_X)
    

        
def build_dict(keypoint_list):
    '''
    method which build a dictionary containing every keypoint of interest for the task. I redefine name such that they are
    more human readable

    Parameters
    ----------
    keypoint_list : dict
        dict of keypoint of interest.

    Returns
    -------
    keypoint_dict : dict
        dictionary of human readable keypoints of interest.

    '''
    keypoint_dict = {} 
    keypoint_dict['l_h'],keypoint_dict['r_h'] = np.array(keypoint_list['5']), np.array(keypoint_list['8'])
    keypoint_dict['l_hip'],keypoint_dict['r_hip'] =  np.array(keypoint_list['10']), np.array(keypoint_list['13'])
    keypoint_dict['l_s'],keypoint_dict['r_s'] =  np.array(keypoint_list['3']), np.array(keypoint_list['6'])
    keypoint_dict['l_e'],keypoint_dict['r_e'] =  np.array(keypoint_list['4']), np.array(keypoint_list['7'])
    keypoint_dict['l_f'],keypoint_dict['r_f'] =  np.array(keypoint_list['12']), np.array(keypoint_list['15'])
    keypoint_dict['l_k'],keypoint_dict['r_k'] =  np.array(keypoint_list['11']), np.array(keypoint_list['14'])
    
    return keypoint_dict
