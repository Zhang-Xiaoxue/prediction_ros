# %%
import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import datetime
import cv2
from cv_bridge import CvBridge

from model import Model
from dataset import DataLoader
from visual import visual, visual_for_real
from set_args import args
from core_funcs import *

current_path = os.path.abspath(__file__)
pkg_src_dir = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
# %tensorboard --logdir logs/

# %%
class Test:
    def __init__(self):
        # load test model weights
        checkpoint_dir = pkg_src_dir + '/training_checkpoints'
        tf.train.latest_checkpoint(checkpoint_dir)
        
        args.batch_size = 1
        
        self.test_model = build_model(args) # Model(args)
        self.test_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        
        self.imgFileDir = pkg_src_dir + '/data/ros_simu_images/'
        
    def predict(self, test_data):
        """do prediction based on the observed data: obs_data

        Args:
            test_data (list[array]): list[array(obs_length,2)], len(list)=num_test_traj=num_ped in this sequence
        """
        num_ped = len(test_data) 
        self.test_model.build(tf.TensorShape([num_ped, None,  2])) # original: tf.TensorShape([1, None,  2])
        
        base_pos = np.array([test_data_ped[0,:].reshape(1,-1) for test_data_ped in test_data]) # (batch_size, 1, 2)
        test_data_diff = test_data - np.tile(base_pos,(1,args.obs_length,1)) # array(batch_size, obs_length, 2)
        
        # The observed part of the trajectory
        observed_traj_diff = tf.convert_to_tensor(test_data_diff, dtype=tf.float32) # Tensor(batch_size, obs_length, 2)

        complete_traj_diff = np.empty([num_ped, args.seq_length, 2])     
        complete_traj_diff[:,:args.obs_length,:] = test_data_diff[:,:args.obs_length,:] # array(batch_size, obs_length, 2)

        gauss_param_diff = np.empty([num_ped, args.pred_length, 5]) # TODO : generalize scalar 5 setting   

        for pred_id in range(args.pred_length):
            
            logits = self.test_model(observed_traj_diff) # Tensor(batch_size, obs_length or 1, output_size)

            [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits) # Tensor(batch_size, obs_length or 1, 1)
            
            next_pos_diff = []
            for ped in range(num_ped):
                next_x_diff, next_y_diff = sample_gaussian_2d(o_mux [ped,-1,0], o_muy[ped,-1,0], 
                                                              o_sx  [ped,-1,0],  o_sy[ped,-1,0], 
                                                              o_corr[ped,-1,0]) # float

                gauss_param_diff[ped, pred_id, :] = np.array([o_mux [ped,-1,0], o_muy[ped,-1,0], 
                                                              o_sx  [ped,-1,0], o_sy [ped,-1,0], 
                                                              o_corr[ped,-1,0]]) # (1, 5, 1)

                complete_traj_diff[ped, args.obs_length+pred_id, :] = np.array([next_x_diff, next_y_diff])
            
            obs_observed_traj_diff = tf.convert_to_tensor(
                complete_traj_diff[:, args.obs_length+pred_id:args.obs_length+pred_id+1, :]
                ) # Tensor(batch_size, 1, 2)
            
        complete_traj = complete_traj_diff + np.tile(base_pos,(1,args.seq_length,1))
        gauss_param = gauss_param_diff
        gauss_param[:,:,0:2] = gauss_param_diff[:,:,0:2] + np.tile(base_pos,(1,args.pred_length,1))
        
        return complete_traj, gauss_param

    def run(self, isSaveVideo=True, isShowImage=True):    
        for idx, test_dataset in enumerate(args.test_dataset):    
            print(f"processing the {idx}th dataset : {test_dataset}")   
            
            img_prefix = os.path.splitext(os.path.split(test_dataset)[1])[0]
            
            self.data = np.genfromtxt(test_dataset, delimiter=',') # 2D-array (4, numTraj) -- seq_id, ped_id, pos_x, pos_y
            
            # Get the number of pedestrians in the current dataset
            numPeds = np.size(np.unique(self.data[1, :]))
            numTraj = self.data.shape[1]
            maxSeqiId = int(self.data[0,-1]) # maximum sequence id in the data
            
            pointer = 0 # pointer in numTraj or idx in selff.data                    
            # TODO: this will be used when there are multiple moving persons... Need to modify
            
            imgs_dataset = []
            
            for seq_id in range(args.obs_length-1, maxSeqiId):
                
                img_suffix = "-visualization_{:05}".format(seq_id) + '.png'
                img = cv2.imread(pkg_src_dir+'/data/ros_simu_images/'+img_prefix+img_suffix)
                
                obs_data = self.stack_obs(seq_id)
                
                if obs_data is None:
                    # plot current pos point
                    # idx = np.argwhere(self.data[0,:]==seq_id).item()
                    # img = cv2.circle(img, (int(self.data[2,idx]), int(self.data[3,idx])), 
                    #                  radius=5, color=(0, 124, 255), thickness=5)
                    if isShowImage:
                        cv2.imshow('Prediction', img) 
                        cv2.waitKey (100) # 显示 1000 ms 即 1s 后消失
                        
                else:
                    obs_data = [obs_data[:,2:4]] # TODO need to chage self.stack_obs if multiple detected persons in data
                    complete_traj, gauss_param = self.predict(obs_data) # array(batch_size, obs_length, 2)
                    for ped_id in range(complete_traj.shape[0]):
                        for _ in range(complete_traj.shape[1]):
                            # 2:4 means pos_x, pos_y
                            if _ < args.obs_length: # observed traj
                                img = cv2.circle(img, 
                                                (int(complete_traj[ped_id,_,0]), int(complete_traj[ped_id,_,1])), 
                                                radius=5, color=(255, 0, 0), thickness=5) # blue
                            else: # predicted traj
                                img = cv2.circle(img, 
                                                (int(complete_traj[ped_id,_,0]), int(complete_traj[ped_id,_,1])), 
                                                radius=5, color=(0, 0, 255), thickness=5) # red
                    
                    if isShowImage:
                        cv2.imshow('Prediction', img)         
                        cv2.waitKey (100) # 显/示 10000 ms 即 10s 后消失
                
                imgs_dataset.append(img)
            
            if isShowImage:
                cv2.destroyAllWindows()  
            
            if isSaveVideo:
                height, width, layers = img.shape
                video_name  = img_prefix+'-visualization.avi'
                video_dir = pkg_src_dir+'/results/'
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
                out = cv2.VideoWriter(video_dir+video_name, 
                                    cv2.VideoWriter_fourcc(*'DIVX'), 
                                    15, (width, height))
                for _ in range(len(imgs_dataset)):
                    out.write(imgs_dataset[_])
                out.release()
                
    
    def stack_obs(self, seq_id):
        # # One method
        # start_id = seq_id-args.obs_length+1
        # if (start_id in self.data[0,:]) and (seq_id in self.data[0,:]):
        #     obs_count = 0
        #     obs_data = np.empty((args.obs_length, self.data.shape[0])) # seq_id, ped_id, pos_x, pos_y
        #     for idx in range(start_id, seq_id+1):
        #         if idx in self.data[0,:]:
        #             obs_data[obs_count, : ] = self.data[:, np.argwhere(self.data[0,:]==idx).item()]
        #             obs_count += 1
        #             if obs_count == args.obs_length :
        #                 return obs_data
        #             else: return None
        #         else: return None
        # else: return None
        
        # Another method
        if seq_id in self.data[0,:]:
            obs_count = 0
            obs_data = np.empty((args.obs_length, self.data.shape[0])) # seq_id, ped_id, pos_x, pos_y
            for hist_id in range(seq_id-args.obs_length+1, seq_id+1):
                if hist_id in self.data[0,:]:
                    obs_data[obs_count, : ] = self.data[:, np.argwhere(self.data[0,:]==hist_id).item()]
                    obs_count += 1
                    if obs_count == args.obs_length :
                        return obs_data
                else: return None
            else: return None

# %%   
if __name__=='__main__':         
    test_res = Test()
    test_res.run()
    
# %%
