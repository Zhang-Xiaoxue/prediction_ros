# %%
import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle
import datetime
import json

from model import Model
from dataset import DataLoader
from visual import visual, visual_for_real
from set_args import args
from core_funcs import *

current_path = os.path.abspath(__file__)
pkg_src_dir = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
# %tensorboard --logdir logs/

# %%
def test(args, IS_VISUALIZE):

    checkpoint_dir = pkg_src_dir+'/training_checkpoints'

    # Initialize the dataloader object to
    # Get sequences of length obs_length+pred_length
    data_loader = DataLoader(1, args.pred_length + args.obs_length, 
                             args.alpha, args.test_dataset, True)

    # Reset the data pointers of the data loader object
    data_loader.reset_batch_pointer()

    tf.train.latest_checkpoint(checkpoint_dir)

    args.batch_size = 1

    test_model = build_model(args) # Model(args)

    test_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    test_model.build(tf.TensorShape([1, None,  2]))

    # Maintain the total_error until now
    total_error = 0
    counter = 0
    final_error = 0.0

    truth_trajs = []
    pred_trajs = []
    gauss_params = []

    for batch_id in range(data_loader.num_batches):
        # Get the source, target data for the next batch
        batch, batch_next = data_loader.next_batch() # list[array(seq_length,2)] len(list)=batch_size

        base_pos = np.array([[ele_batch[0] for _ in range(len(ele_batch))] for ele_batch in batch]) # array(batch_size, seq_length, 2)
        batch = batch - base_pos # array(batch_size, seq_length, 2)

        # The observed part of the trajectory
        obs_observed_traj = batch[0][:args.obs_length] # array(obs_length, 2)
        obs_observed_traj = tf.expand_dims(obs_observed_traj, 0) # Tensor(1, obs_length, 2)

        complete_traj = batch[0][:args.obs_length] # array(obs_length, 2)

        test_model.reset_states()

        # test_model.initial_state = None
        gauss_param = np.array([])

        for idx in range(args.pred_length):
            tensor_batch = tf.convert_to_tensor(obs_observed_traj) # Tensor(1, obs_length or 1, 2)

            logits = test_model(tensor_batch) # Tensor(1, obs_length or 1, output_size)

            [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits) # Tensor(1, obs_length or 1, 1)

            next_x, next_y = sample_gaussian_2d(o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]) # float

            obs_observed_traj = tf.expand_dims([[next_x, next_y]], 0) # Tensor(1, obs_length or 1, 2)

            if len(gauss_param) <=0:
                gauss_param = np.array([o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]])
            else:
                gauss_param = np.vstack((gauss_param, [o_mux[0][-1][0], o_muy[0][-1][0], o_sx[0][-1][0], o_sy[0][-1][0], o_corr[0][-1][0]])) # finally (pred_length, output_size)


            complete_traj = np.vstack((complete_traj, [next_x, next_y])) # from (obs_length, 2) to (seq_length, 2)

        total_error += get_mean_error (complete_traj + base_pos[0], batch[0] + base_pos[0], args.obs_length)
        final_error += get_final_error(complete_traj + base_pos[0], batch[0] + base_pos[0])

        pred_trajs.append(complete_traj)
        truth_trajs.append(batch[0])
        gauss_params.append(gauss_param)

        print("Processed trajectory number: {} out of {} trajectories".format(batch_id, data_loader.num_batches))

    # Print the mean error across all the batches
    print("Total mean error of the model is {}".format(total_error/data_loader.num_batches))
    print("Total final error of the model is {}".format(final_error/data_loader.num_batches))

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    output_dir = pkg_src_dir+"/results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_file = output_dir + "/pred_results_"+current_time+".pkl"
    with open(data_file, "wb") as f:
        pickle.dump([pred_trajs, truth_trajs, gauss_params], f)
    
    if IS_VISUALIZE:
        visual(data_file)

def train(args):
    data_loader = DataLoader(args.batch_size, args.seq_length, args.alpha,
                             args.train_dataset, forcePreProcess=True)

    # Create a Vanilla LSTM model with the arguments
    model = build_model(args)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    optimizer = tf.keras.optimizers.RMSprop(args.learning_rate)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # save argments parameters
    argsfile_dir = pkg_src_dir+'/args_logs'
    if not os.path.exists(argsfile_dir):
        os.makedirs(argsfile_dir)
    with open(argsfile_dir+'/args_'+current_time+'.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    train_log_dir = pkg_src_dir+'/logs/gradient_tape/' + current_time + '/train'
    # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # 检查点保存至的目录
    checkpoint_dir = pkg_src_dir+'/training_checkpoints'
    # 检查点的文件名
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    for e in range(args.num_epochs):

        data_loader.reset_batch_pointer()
        model.reset_states() 
        # reset_states clears only the hidden states of your network. 
        # if set stateful=True:
        #   - should call reset_states every time for model calls independent
        # If not set:
        #   - all states are automatically reset after every batch computations 
        #     (so e.g. after calling fit, predict and evaluate also). 

        for batch_id in range(data_loader.num_batches):
            start = time.time()

            batch, next_batch = data_loader.next_batch()
            # len(x)=batch_size; len(y)=batch_size. x, y : list[ array(seq_length,2) ]

            base_pos = np.array([[ele_batch[0] for _ in range(len(ele_batch))] for ele_batch in batch])
            # ele_batch: 2Darray(seq_length,2) ; ele_batch[0] 重复 seq_length, for all ele_batch.
            # base_pos = array(batch_size, seq_length, 2)

            batch_offset = batch - base_pos # array(batch_size, seq_length, 2)
            next_batch_offset = next_batch - base_pos # array(batch_size, seq_length, 2)

            with tf.GradientTape() as tape:
                tensor_batch = tf.convert_to_tensor(batch_offset, dtype=tf.float32)

                logits = model(tensor_batch) # logits: array(batch_size, seq_length, output_size)

                # output -- mux, muy, sx, sy,corr : Tensor(batch_size, seq_length, 1)
                [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(logits)

                tensor_next_batch = tf.convert_to_tensor(next_batch_offset, dtype=tf.float32)
                
                # x_data, y_data in next_batch : Tensor(batch_size, seq_length, 1)
                [x_data, y_data] = tf.split(tensor_next_batch, 2, -1)

                # Compute the loss function
                loss = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data) # loss:float

                mean_error, final_error = calc_prediction_error(o_mux, o_muy, o_sx, o_sy, o_corr, tensor_next_batch, args)

                loss = tf.math.divide(loss, (args.batch_size * args.seq_length))

                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.lr.assign(args.learning_rate * (args.decay_rate ** e))
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss(loss)

            end = time.time()
            # Print epoch, batch, loss and time taken
            print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, mean error = {}, final_error = {}"
                    .format(e * data_loader.num_batches + batch_id,
                            args.num_epochs * data_loader.num_batches,
                            e, loss, end - start, mean_error, final_error))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=e)
            # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        model.save_weights(checkpoint_prefix.format(epoch=e))    

# %%
if __name__ == '__main__':
    # train and set
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=8,
                        help='RNN sequence length')
    parser.add_argument('--alpha', type=int, default=50,
                        help='The superparameter to increase the number of trajectory snippets')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--leaveDataset', type=int, default=1,
                        help='The dataset index to be left out in training')
    # train_datasets_dirs = ['./data/eth/univ', './data/eth/hotel', 
    #                        './data/ucy/zara/zara01', './data/ucy/zara/zara02',
    #                        './data/ucy/univ']
    current_path = os.path.abspath(__file__)
    pkg_src_dir = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    parser.add_argument('--train_dataset', type=list, 
                        default=[pkg_src_dir+"/data/ros_simu/rosworld1-yolov7-yolov7_detection.csv", ],
                        help='Dataset to be trained on')
    parser.add_argument('--test_dataset', type=list, 
                        default=[pkg_src_dir+"/data/ros_simu/rosworld1-yolov7-yolov7_detection.csv", ],
                        help='Dataset to be tested on')
    parser.add_argument('--obs_length', type=int, default=5,
                        help='Observed length of the trajectory')
    parser.add_argument('--pred_length', type=int, default=3,
                        help='Predicted length of the trajectory')

    # args = parser.parse_args(args=[])
    args = parser.parse_args(args=['--batch_size', '50', '--num_epochs', '200', '--alpha', '50',
                                    '--seq_length', '10', '--obs_length', '6', '--pred_length', '4'])
    train(args)
    
    args.alpha = 2
    test(args, IS_VISUALIZE=True)

    # test_data = [np.random.random_sample((args.seq_length,2)), 
    #              np.random.random_sample((args.seq_length,2)), 
    #              np.random.random_sample((args.seq_length,2))]
    # test_real(args, test_data, IS_VISUALIZE=True)
    
# %%

