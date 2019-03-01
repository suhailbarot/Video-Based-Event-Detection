#!/home/ubuntu/anaconda3/envs/py37/bin/python

import os
import sys
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb


def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval,video_name):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    # 
    # f = open(surf_feat_video_filename, "w+")
    images = get_keyframes(downsampled_video_filename,keyframe_interval)
    data = []
    counter = 0
    for image in images:
        image_g =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        surf = cv2.SURF(400,2,3,1)
        keypoints, descriptors = surf.detectAndCompute(image_g,None, useProvidedKeypoints = False)
        if descriptors is not None:
            for row in descriptors: 
                data.append(row)
    print(downsampled_video_filename)
    try:
        data = np.array(data)
        print(data.shape)
    except:
        not_done.write(downsampled_video_filename+"\n")
    if data is not []:
        np.savez("surf/"+video_name+'.npz',data)




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
    print(sys.argv)
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)
    not_done = open('notdone.txt','w+')

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))
    print(sys.path)

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
