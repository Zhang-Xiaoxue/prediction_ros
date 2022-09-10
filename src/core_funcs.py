# %%
import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import datetime

from model import Model
from dataset import DataLoader
from visual import visual, visual_for_real
from set_args import args
    
# %%
def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
    # See Bivariate case in multivariate normal distribution:
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution.
    normx = tf.math.subtract(x, mux)
    normy = tf.math.subtract(y, muy)
    # Calculate sx*sy
    sxsy = tf.math.multiply(sx, sy)
    # Calculate the exponential factor
    z = tf.math.square(tf.math.divide(normx, sx)) + tf.math.square(tf.math.divide(normy, sy)) - \
        2*tf.math.divide(tf.math.multiply(rho, tf.math.multiply(normx, normy)), sxsy)
    
    negRho = 1 - tf.math.square(rho)
    # Numerator
    result = tf.math.exp(tf.math.divide(-z, 2*negRho))
    # Normalization constant
    denom = 2 * np.pi * tf.math.multiply(sxsy, tf.math.sqrt(negRho))
    # Final PDF calculation
    result = tf.math.divide(result, denom) # Tensor(batch_size, seq_length, 1)

    return result

def get_coef(output):
    """generate mu, sigma, rho from the output of RNN model

    Args:
        output (Tensor): output of RNN model : Tensor(batch_size, seq_length, 1)

    Returns:
        tuple (mu_x, mu_y, sigma_x, sigma_y, rho) : Tensor(batch_size, seq_length, 1) 
    """
    z = output

    z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, -1) # 5 is output_size
    # z_mux, z_muy, z_sx, z_sy, z_corr : (batch_size, seq_length, 1)

    z_sx = tf.exp(z_sx)
    z_sy = tf.exp(z_sy)
    z_corr = tf.tanh(z_corr)

    return [z_mux, z_muy, z_sx, z_sy, z_corr]

def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
    """loss function of the RNN model, 
       Sum of negative log-likelihood estimates of all predicted trajectory points

    Args:
        z_mux (Tensor): RNN unit output mu_x, Tensor(batch_size, seq_length, 1)
        z_muy (Tensor): RNN unit output mu_y, Tensor(batch_size, seq_length, 1)
        z_sx (Tensor):  RNN unit output sigma_x, Tensor(batch_size, seq_length, 1)
        z_sy (Tensor):  RNN unit output sigma_y, Tensor(batch_size, seq_length, 1)
        z_corr (Tensor): RNN unit output rho, Tensor(batch_size, seq_length, 1)
        x_data (Tensor): groundtruth x, Tensor(batch_size, seq_length, 1)
        y_data (Tensor): groundtruth y, Tensor(batch_size, seq_length, 1)

    Returns:
        float: loss
    """
    # z_mux, z_muy, z_sx, z_sy, z_corr : output results
    result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

    epsilon = 1e-20

    result1 = -tf.math.log(tf.math.maximum(result0, epsilon))  # Numerical stability

    return tf.reduce_sum(result1)

def get_mean_error(pred_traj, true_traj, observed_length):
    """Compute ADE:
    The sum of distances between all predicted points and the 
    GroundTruth points / number of predicted trajectory points

    Args:
        pred_traj (_type_): _description_
        true_traj (_type_): _description_
        observed_length (_type_): _description_

    Returns:
        _type_: _description_
    """
    error = np.zeros(len(true_traj) - observed_length)
    for i in range(observed_length, len(true_traj)):
        # The predicted position
        pred_pos = pred_traj[i, :]
        # The true position
        true_pos = true_traj[i, :]

        # The euclidean distance is the error
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)

def get_final_error(pred_traj, true_traj):
    """Compute FDE
    The distance between the last predicted point and the corresponding GroundTruth point

    Args:
        pred_traj (_type_): _description_
        true_traj (_type_): _description_

    Returns:
        _type_: _description_
    """

    error = np.linalg.norm(pred_traj[-1, :] - true_traj[-1, :])

    # Return the mean error
    return error

def sample_gaussian_2d(mux, muy, sx, sy, rho):
    """sample pred pos (x,y) based on mu, sigma, rho

    Returns:
        pred_x, pred_y: float, float
    """
    # Extract mean
    mean = [mux, muy]

    # Extract covariance matrix
    cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

# def test_real(args, test_data, IS_VISUALIZE):
#     """_summary_

#     Args:
#         args (dict): _description_
#         test_data (list[array]): list[array(obs_length,2)], len(list)=num_test_traj
#         IS_VISUALIZE (bool): whether to visulize the prediction results
#     """
    
#     # load test model weights
#     checkpoint_dir = './training_checkpoints'
#     tf.train.latest_checkpoint(checkpoint_dir)

#     args.batch_size = len(test_data)

#     test_model = build_model(args) # Model(args)
#     test_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
#     test_model.build(tf.TensorShape([args.batch_size, None,  2])) # original: tf.TensorShape([1, None,  2])
    
#     pred_trajs, truth_trajs = [], []

#     # start prediction
    
#     # Get the source, target data for the next batch
#     batch = test_data # list[array(obs_length,2)] len(list)=batch_size

#     base_pos = np.array([[ele_batch[0] for _ in range(len(ele_batch))] for ele_batch in batch]) # array(batch_size, obs_length, 2)
#     batch = batch - base_pos # array(batch_size, obs_length, 2)

#     test_model.reset_states()
#     # test_model.initial_state = None
    
#     # The observed part of the trajectory
#     obs_observed_traj = tf.convert_to_tensor(test_data, dtype=tf.float32) # Tensor(batch_size, obs_length, 2)

#     complete_traj = np.empty([args.batch_size, args.seq_length, 2])     
#     complete_traj[:,:args.obs_length,:] = batch[:,:args.obs_length,:] # array(batch_size, obs_length, 2)

#     gauss_param = np.empty([args.batch_size, args.pred_length, 5])      
 
#     for pred_id in range(args.pred_length):
#         tensor_batch = obs_observed_traj # Tensor(batch_size, obs_length or 1, 2)

#         logits = test_model(tensor_batch) # Tensor(batch_size, obs_length or 1, output_size)

#         [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits) # Tensor(batch_size, obs_length or 1, 1)
        
#         next_pos = []
#         for traj_id in range(args.batch_size): # here, arg.batch_size is also num_traj = len(test_data)
#             next_x, next_y = sample_gaussian_2d(o_mux[traj_id][-1][0], o_muy[traj_id][-1][0], o_sx[traj_id][-1][0], o_sy[traj_id][-1][0], o_corr[traj_id][-1][0]) # float

#             gauss_param[traj_id, pred_id, :] = np.array([o_mux[traj_id][-1][0], o_muy[traj_id][-1][0], o_sx[traj_id][-1][0], o_sy[traj_id][-1][0], o_corr[traj_id][-1][0]]) # (1, 5, 1)

#             complete_traj[traj_id, args.obs_length+pred_id, :] = np.array([next_x, next_y])
            
#         obs_observed_traj = tf.convert_to_tensor(complete_traj[:, args.obs_length:args.obs_length+pred_id+1, :]) # Tensor(batch_size, 1, 2)
    
#     if IS_VISUALIZE:
#         visual_for_real(complete_traj, gauss_param)

def build_model(args):
    """Embedding layer, LSTM/GRU layer, Output layer
    Embedding层将坐标(x,y)嵌入到64维的向量空间
    输出层输出每个预测点的二维高斯分布参数(包含5个参数:mux, muy, sx, sy, corr), 
    """
    output_size = 5 # mux, muy, sx, sy, corr
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(args.embedding_size, 
                              activation = tf.keras.activations.relu,
                              batch_input_shape = [args.batch_size, None, 2]),
        tf.keras.layers.GRU(args.rnn_size, 
                            return_sequences=True, # return the last output, or the full sequence
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(output_size)
    ])

    return model

def calc_prediction_error(mux, muy, sx, sy, corr, offset_positions, args):

    traj_nums = mux.shape[0] # batch_size

    pred_nums = mux.shape[1] # seq_length

    mean_error = 0.0
    final_error = 0.0
    for index in range(traj_nums):
        pred_traj = np.zeros((pred_nums, 2))
        for pt_index in range(pred_nums):
            next_x, next_y = sample_gaussian_2d(mux[index][pt_index][0],
                            muy[index][pt_index][0], sx[index][pt_index][0],
                            sy[index][pt_index][0], corr[index][pt_index][0])

            pred_traj[pt_index][0] = next_x
            pred_traj[pt_index][1] = next_y

        mean_error += get_mean_error(pred_traj, offset_positions[index], args.obs_length)
        final_error += get_final_error(pred_traj, offset_positions[index])

    mean_error = mean_error / traj_nums
    final_error = final_error / traj_nums

    return mean_error, final_error

def create_stamped_prediction_msg(predictions: np.array, class_names):
    """
    :param prediction: array(num_test_traj, seq_length, 2)
    :param detections: torch tensor of shape [num_boxes, 6] where each element is
        [x1, y1, x2, y2, confidence, class_id]
    :returns: detections as a ros message of type ObjectsStamped
    """
    detection_array_msg = ObjectsStamped()
    i = 0
    # header
    header = create_header()
    detection_array_msg.header = header
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        
        single_detection_msg = Object()
        # bbox
        w = int(round(x2 - x1))
        h = int(round(y2 - y1))
        cx = int(round(x1 + w / 2))
        cy = int(round(y1 + h / 2))

        bounding_box_2d = BoundingBox2Di()
        
        bounding_box_2d.corners[0] = [x1, y1]
        bounding_box_2d.corners[1] = [x1, y2]
        bounding_box_2d.corners[2] = [x2, y1]
        bounding_box_2d.corners[3] = [x2, y2]

        single_detection_msg.bounding_box_2d = bounding_box_2d

        single_detection_msg.center = Pose2D()
        single_detection_msg.center.x = cx
        single_detection_msg.center.y = cy

        single_detection_msg.label = class_names[int(cls)]
        single_detection_msg.label_id = i
        single_detection_msg.confidence = conf

        detection_array_msg.objects.append(single_detection_msg)
        i = i + 1
    return detection_array_msg 


# %%
# train and set
# train(args)
# test(args, IS_VISUALIZE=True)

# test_data = [np.random.random_sample((args.seq_length,2)), 
#              np.random.random_sample((args.seq_length,2)), 
#              np.random.random_sample((args.seq_length,2))]
# test_real(args, test_data, IS_VISUALIZE=True)