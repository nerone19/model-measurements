# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 08:44:26 2021

@author: gabri
"""



from features.utils import *



def main(filename = None):
    print(filename,os.getcwd())
    #running with FASTAPI
    if(filename is not None):
        os.environ['filename'],file_extension = filename.split('.')[:]
        print('../data/processed' + filename)
        rgb = cv2.imread(os.path.join('../data/processed' , filename))
        print(rgb.shape)
        os.environ['label_path'] = '../labels/processed/'
    else: 
        file_extension = os.environ['filename'].split('.')[1]
        #get image in rgb,yuv,grayscale
        rgb = cv2.imread( os.path.join(os.environ['image_path'] ,os.environ['filename']))

    
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    #get entropy image
    entr_img = entropy(gray, disk(5))
    scaled_entropy = entr_img / entr_img.max() 
    entr_img = scaled_entropy

    #for debugging purposes: entropy image
    # plt.imshow(entr_img > 0.4)
    # plt.show()

    #stack rgb and yuv for more information about the colours and their  intensities
    rgb_yuv = np.concatenate((np.array(rgb),np.array(yuv) ), axis = 2 )

    json_file = os.environ['filename'].split('.')[0] + ".json"
    with open( os.path.join(os.environ['label_path'] ,json_file) ) as f:
        data = json.load(f)
        pose_keypoints =  data['people'][0]['pose_keypoints_2d']
        

        
    #drawing keypoints over the image for understanding what is what
    counter = 1
    keypoint_list = {}
    toKeep = [3,4,5,6,7,8,9,10,11,13,12,14,15]
    for i in range(0,len(pose_keypoints),3):
        x,y = pose_keypoints[i],pose_keypoints[i+1]
        x,y = int(x),int(y)
        
        if(counter in toKeep):
            keypoint_list[str(counter)] = (x,y)
            cv2.circle(rgb, (x,y), radius=0, color=(0, 0, 255), thickness=5)
            cv2.putText(rgb, str(counter), (x - 20, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        counter +=1 

    #building human readable dictionary
    keypoint_dict = build_dict(keypoint_list)
    height,width,_ = rgb.shape
    #create gabor kernels
    kernels = create_gabor_kernels()
    #get hip line
    full_line ,waist_line,_,_ =get_line(keypoint_dict['l_hip'],keypoint_dict['r_hip'], width)
    #get training data
    train_x,train_y,test_x = get_data(keypoint_dict,full_line,waist_line, rgb_yuv,entr_img,kernels)
    #predicti pixel over the hip line
    pred = predict_pixel_values(keypoint_dict,train_x,train_y,test_x)


    #for debuggin purposes
    res_line = []
    for i in range(len(pred)):
        x,y = full_line[i]
        res = np.argmax(pred[i])
        res_line.append(res)
        if(res == 0):
            cv2.circle(rgb, (x+3,y), radius=0, color=(255, 0, 0), thickness=5)

    #approximate waist width

    toll = 10
    l_w_point = keypoint_dict['l_h']
    for i in range(0,keypoint_dict['l_hip'][0],1):
        if(res_line[i] == 1):
            toll -= 1
        else:
            toll= 10
        if( toll == 0):
            x,y = full_line[i]
            l_w_point = [x+3-10,y]
            cv2.circle(rgb, (x+3-10,y), radius=0, color=(0, 255, 0), thickness=20)
            break
            
            
    toll = 10
    r_w_point = keypoint_dict['r_h']
    for i in range(rgb.shape[1]-8,keypoint_dict['r_hip'][0],-1):
        if(res_line[i] == 1):
            toll -= 1
        else:
            toll= 10
        if( toll == 0):
            x,y = full_line[i]
            r_w_point = [x-4+10,y]
            cv2.circle(rgb, (x-4+10,y), radius=0, color=(0, 255, 0), thickness=20)

            break


    full_shoulder_line,shoulder_line,_,_ =get_line(keypoint_dict['l_s'],keypoint_dict['r_s'],width)
    get_arm_boundaries(full_shoulder_line,gray)
    mean_leg_length,mean_arm_length = get_measures(keypoint_dict)
    hip_length = abs(np.linalg.norm(np.array(l_w_point)-np.array(r_w_point)))
    get_upper_body_measures(keypoint_dict,width,rgb)
    print('leg length: {},  arm length: {}, hip_length: {}'.format(mean_leg_length,mean_arm_length,hip_length))

    rgb = cv2.resize(rgb, ( int(width/2),int(height/2) ))  
    cv2.imshow('img',rgb)
    cv2.waitKey(0)

    res = {}
    res['leg-length'] = mean_leg_length
    res['arm-length'] = mean_arm_length
    res['hip-length'] = hip_length
    return json.dumps(res)

def dir_path(string):
    if os.path.isdir(string):
        return string.replace('\\','/')
    else:
        raise NotADirectoryError(string)

if __name__ == "__main__":
    
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()
    
    # where the original images (used for creating the synthetic images) are  
    parser.add_argument(
    "--image_path",
    type=str
    )

    parser.add_argument(
    "--label_path",
    type=str
    )

    parser.add_argument(
    "--filename",
    type=str
    )

    args = parser.parse_args()
    assert(args.image_path is not None)
    os.environ['image_path'] = args.image_path.replace('\\\\', ' ')
    assert(args.label_path is not None)
    os.environ['label_path'] = args.label_path.replace('\\\\', ' ')
    assert(args.filename is not None)
    os.environ['filename'] = args.filename
    main()
