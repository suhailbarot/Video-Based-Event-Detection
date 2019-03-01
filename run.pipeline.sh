#!/bin/bash



# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores, d) computation of class labels for kaggle submission.

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -k true -y filepath

# Reading of all arguments:


while getopts p:f:m:k:y: option		# p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
	k) KAGGLE=$OPTARG;;                # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	esac
	done

export PATH=~/anaconda3/bin:$PATH

if [ "$PREPROCESSING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

    # steps only needed once
    video_path=/home/ubuntu/11775_videos/video  # path to the directory containing all the videos.
    mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
    awk '{print $1}' ../hw1_code/list/train > list/train.video  # save only video names in one file (keeping first column)
    awk '{print $1}' ../hw1_code/list/val > list/val.video
    cat list/train.video list/val.video list/all_test.video > list/all.video    #save all video names in one file
    downsampling_frame_len=60
    downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    start=`date +%s`
    for line in $(cat "list/remaining.video"); do

         ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
     done
     end=`date +%s`
     runtime=$((end-start))
     echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization

    # 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
     ./surf_feat_extraction.py list/all.video config.yaml

    # 3. TODO: Extract CNN features from keyframes of downsampled videos
	

fi

if [ "$FEATURE_REPRESENTATION" = true ] ; then

    echo "#####################################"
    echo "#  SURF FEATURE REPRESENTATION      #"
    echo "#####################################"

    python2 train_kmeans.py 500


    python2 create_kmeans.py kmeans_model_500 500 list/all.video

	echo "#####################################"
    echo "#   CNN FEATURE REPRESENTATION      #"
    echo "#####################################"

	./cnn_feature_extraction.py  list/all.video config.yaml
    python2 train_kmeans_cnn.py 500
    python2 create_kmeans_cnn.py kmeans_cnn_model_500 500 list/all.video


    # 2. TODO: Create kmeans representation for CNN features

fi

if [ "$MAP" = true ] ; then

    echo "#######################################"
    echo "# MED with SURF Features: MAP results #"
    echo "#######################################"

    # Paths to different tools;
    map_path=/home/ubuntu/tools/mAP
    export PATH=$map_path:$PATH
    mkdir -p cnn_pred
    # iterate over the events
    feat_dim_surf=500
    feat_dim_cnn=500
    mode="surf"
    cat "list/train" "list/val" >"./merged_train"
    for event in P001 P002 P003; do
        if [ "$mode" = "cnn" ] ; then

          echo "=========  Event $event  ========="
          # now train a svm model

          python2 train_svm.py $event "kmeans_cnn/" $feat_dim_cnn cnn_pred/svm.$event.$feat_dim_cnn.cnn.model || exit 1;
          # apply the svm model to *ALL* the testing videos;
          # output the score of each testing video to a file ${event}_pred 
          python2 test_svm.py cnn_pred/svm.$event.$feat_dim_cnn.cnn.model "kmeans_cnn/" $feat_dim_cnn cnn_pred/${event}_cnn.lst $event || exit 1;
        #   # compute the average precision by calling the mAP package
        python2 evaluator.py list/${event}_val_label.txt cnn_pred/${event}_cnn.lst
          # ap list/${event}_val_label.txt cnn_pred/${event}_cnn.lst
        fi
        if [ "$mode" = "surf" ] ; then

          echo "=========  Event $event  ========="
          # now train a svm model

          python2 train_svm.py $event "kmeans/" $feat_dim_surf surf_pred/svm.$event.$feat_dim_surf.model || exit 1;
          # apply the svm model to *ALL* the testing videos;
          # output the score of each testing video to a file ${event}_pred 
          python2 test_svm.py surf_pred/svm.$event.$feat_dim_surf.model "kmeans/" $feat_dim_surf surf_pred/${event}_surf.lst $event || exit 1;
        #   # compute the average precision by calling the mAP package
                python2 evaluator.py list/${event}_val_label.txt surf_pred/${event}_surf.lst

          # ap list/${event}_val_label.txt surf_pred/${event}_surf.lst
        fi
    done

    echo "#######################################"
    echo "# MED with CNN Features: MAP results  #"
    echo "#######################################"


    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission

fi


if [ "$KAGGLE" = true ] ; then

    echo "##########################################"
    echo "# MED with SURF Features: KAGGLE results #"
    echo "##########################################"

    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

    # 4. TODO: Test SVM with test set saving scores for submission


    echo "##########################################"
    echo "# MED with CNN Features: KAGGLE results  #"
    echo "##########################################"

    # 1. TODO: Train SVM with OVR using only videos in training set.

    # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.

	# 3. TODO: Train SVM with OVR using videos in training and validation set.

	# 4. TODO: Test SVM with test set saving scores for submission

fi
