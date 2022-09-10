#!/usr/bin/python3

import numpy as np
import argparse
import os
import time
import pickle
import datetime
from typing import Tuple, Union

import tensorflow as tf
import torch
import cv2
from torchvision.transforms import ToTensor
from cv_bridge import CvBridge

import rospy
from yolov7_ros.msg import ObjectsStamped
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from prediction_ros.msg import PredictionUncertainty


from set_args import args
from model import Model
from dataset import DataLoader
from visual import draw_predictions
from core_funcs import *

class MotionPred:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = build_model(args)
    
    def inference(self, test_data):
        """
        :param test_data: list[array(obs_length,2)], len(list)=num_test_traj
        :returns complete_traj : array(num_test_traj, seq_length, 2),
                gauss_param : array(num_test_traj, pred_length, 5)      
        """
        print("start inference!!!")
        # load test model weights
        checkpoint_dir = './training_checkpoints'
        tf.train.latest_checkpoint(checkpoint_dir)

        args.batch_size = len(test_data)

        test_model = build_model(args) # Model(args)
        test_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        test_model.build(tf.TensorShape([args.batch_size, None,  2])) # original: tf.TensorShape([1, None,  2])
        
        pred_trajs, truth_trajs = [], []

        # start prediction
        
        # Get the source, target data for the next batch
        batch = test_data # list[array(obs_length,2)] len(list)=batch_size

        base_pos = np.array([[ele_batch[0] for _ in range(len(ele_batch))] for ele_batch in batch]) # array(batch_size, obs_length, 2)
        batch = batch - base_pos # array(batch_size, obs_length, 2)

        test_model.reset_states()
        # test_model.initial_state = None
        
        # The observed part of the trajectory
        obs_observed_traj = tf.convert_to_tensor(test_data, dtype=tf.float32) # Tensor(batch_size, obs_length, 2)

        complete_traj = np.empty([args.batch_size, args.seq_length, 2])     
        complete_traj[:,:args.obs_length,:] = batch[:,:args.obs_length,:] # array(batch_size, obs_length, 2)

        gauss_param = np.empty([args.batch_size, args.pred_length, 5])      
    
        for pred_id in range(args.pred_length):
            tensor_batch = obs_observed_traj # Tensor(batch_size, obs_length or 1, 2)

            logits = test_model(tensor_batch) # Tensor(batch_size, obs_length or 1, output_size)

            [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits) # Tensor(batch_size, obs_length or 1, 1)
            
            next_pos = []
            for traj_id in range(args.batch_size): # here, arg.batch_size is also num_traj = len(test_data)
                next_x, next_y = sample_gaussian_2d(o_mux[traj_id][-1][0], o_muy[traj_id][-1][0], o_sx[traj_id][-1][0], o_sy[traj_id][-1][0], o_corr[traj_id][-1][0]) # float

                gauss_param[traj_id, pred_id, :] = np.array([o_mux[traj_id][-1][0], o_muy[traj_id][-1][0], o_sx[traj_id][-1][0], o_sy[traj_id][-1][0], o_corr[traj_id][-1][0]]) # (1, 5, 1)

                complete_traj[traj_id, args.obs_length+pred_id, :] = np.array([next_x, next_y])
                
            obs_observed_traj = tf.convert_to_tensor(complete_traj[:, args.obs_length:args.obs_length+pred_id+1, :]) # Tensor(batch_size, 1, 2)
        
        return complete_traj, gauss_param

class MotionPredPublisher:
    def __init__(self, img_topic: str, det_topic: str, pub_topic: str = "pred_result",
                 img_size: Union[Tuple[int, int], None] = (640, 640),
                 device: str = "cuda", queue_size: int = 1, visualize: bool = False):
        """
        :param img_topic: name of the image topic to listen to (for visualization)
        :param det_topic: name of the detection topic to listen to (receive and process)
        :param pub_topic: name of the output topic (will be published under the
            namespace '/prediction')
        :param img_size: (height, width) to which the img is resized before being
            fed into the yolo network. Final output coordinates will be rescaled to
            the original img size.
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the prediction results visualized in the image
        """
        
        self.device = device

        vis_topic = pub_topic + "visualization" if pub_topic.endswith("/") else \
            pub_topic + "/visualization"
        self.visualization_publisher = rospy.Publisher(
            vis_topic, Image, queue_size=queue_size
        ) if visualize else None

        self.bridge = CvBridge()

        self.tensorize = ToTensor()
        self.model = MotionPred(device=device)
        
        self.img_subscriber = rospy.Subscriber(
            img_topic, Image, self.process_img_msg
        )
        
        self.det_subscriber = rospy.Subscriber(det_topic, ObjectsStamped, self.process_det_msg)
        
        self.prediction_publisher = rospy.Publisher(
            pub_topic, Float64MultiArray, queue_size=queue_size
        )
        
        rospy.loginfo("Prediction Module initialization complete. Ready to start inference")
    
    def process_det_msg(self, det_msg: ObjectsStamped):
        """callback function for detection"""
        det_msg_objects = det_msg.objects()
        test_data = []
        for det_msg_obj in det_msg_objects:
            label_id = det_msg_obj.label_ID
            pos_x = det_msg_obj.center.x
            pos_y = det_msg_obj.center.y
        
            test_data.append(np.array([pos_x, pos_y]).reshape(-1,1))
    
    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough'
        )

        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        h_orig, w_orig, c = np_img_orig.shape
        if c == 1:
            np_img_orig = np.concatenate([np_img_orig] * 3, axis=2)
            c = 3

        # Automatically resize the image to the next smaller possible size
        w_scaled, h_scaled = self.img_size

        # w_scaled = w_orig - (w_orig % 8)
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

        #Conversion to torch tensor (copied from original yolov7 repo)
        if np_img_resized.shape[2] == 4: #Removing extra channel if RGBA
            np_img_resized = np_img_resized[:,:,:3]
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)
        
        test_data = img # TODO list[array(obs_length,2)], len(list)=num_test_traj
        #Inference & rescaling the output to original img size
        predictions, pred_uncertainty = self.model.inference(test_data) 
        # predictions : array(args.batch_size, args.seq_length, 2)
        # pred_uncertainty: array(args.batch_size, args.pred_length, 5)

        #Publishing predictions
        prediction_msg = Float64MultiArray()  # the data to be sent, initialise the array
        prediction_msg.data = predictions
        self.prediction_publisher.publish(prediction_msg)

        #Publishing Visualization if Required
        if self.visualization_publisher:
            vis_img = draw_predictions(np_img_orig, predictions)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img)
            self.visualization_publisher.publish(vis_msg)


if __name__ == "__main__":
    rospy.init_node("motion_pred")

    ns = rospy.get_name() + "/"

    img_topic = rospy.get_param(ns + "img_topic")
    det_topic = rospy.get_param(ns + "det_topic")
    out_topic = rospy.get_param(ns + "out_topic")
    queue_size = rospy.get_param(ns + "queue_size")
    img_size = rospy.get_param(ns + "img_size")
    visualize = rospy.get_param(ns + "visualize")
    device = rospy.get_param(ns + "device")

    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.")

    publisher = MotionPredPublisher(
        img_topic=img_topic,
        det_topic=det_topic,
        pub_topic=out_topic,
        img_size=(img_size, img_size),
        device=device,
        queue_size=queue_size,
        visualize=visualize
    )

    rospy.spin()
