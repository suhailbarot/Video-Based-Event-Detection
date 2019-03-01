#!/home/ubuntu/anaconda3/envs/py37/bin/python

import sys
import cv2
import os
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image




def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval,video_name):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    # 
    # f = open(surf_feat_video_filename, "w+")
    images = get_keyframes(downsampled_video_filename,keyframe_interval)
    data = []
    counter = 0
    for image in images:
        image_g =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imwrite("frame.jpg", image) 
        data.append(get_vector("frame.jpg").numpy())
    data = np.array(data)
    print(data.shape)
    np.savez('cnn/'+video_name+'.npz',data)

        
    # print(downsampled_video_filename)
    # try:
    #     data = np.array(data)
    #     print(data.shape)
    # except:
    #     not_done.write(downsampled_video_filename+"\n")
    # if data is not []:
    #     np.savez("cnn/"+video_name+'.npz',data)




def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)
    not_done = open('notdone.txt','w+')
    model = models.resnet18(pretrained=True)
    layer = model._modules.get('avgpool')
    model.eval()
    scaler = transforms.Scale((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                 std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create SURF object

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    i=0
    for line in fread.readlines():
        print(i)
        i+=1
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')

        if not os.path.isfile(downsampled_video_filename):
            print("s")

        # Get SURF features for one video
        get_surf_features_from_video(downsampled_video_filename,
                                     surf_feat_video_filename, keyframe_interval,video_name)
