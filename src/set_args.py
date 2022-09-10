import numpy as np
import argparse
import os

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
# parser.add_argument('--test_dataset', type=int, default=4,
#                     help='Dataset to be tested on')
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
args = parser.parse_args(args=['--batch_size', '50', '--num_epochs', '400', '--alpha', '50',
                               '--seq_length', '10', '--obs_length', '6', '--pred_length', '4'])