# %%
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import pickle
from scipy.stats import multivariate_normal
import cv2
current_path = os.path.abspath(__file__)
pkg_src_dir = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

# %%
def get_random_color(seed):
    gen = np.random.default_rng(seed)
    color = tuple(gen.choice(range(256), size=3))
    color = tuple([int(c) for c in color])
    return color

def draw_predictions(image, args, predictions: np.array):
    color = get_random_color(redictions.shape[0])
    for traj_id in range(predictions.shape[0]):
        image = cv2.drawMarker(image, 
                               (predictions[traj_id][0:args.obs_length, 0], 
                                predictions[traj_id][0:args.obs_length, 1]),
                               color=color[traj_id],
                               markerType=cv2.MARKER_CROSSS,
                               thickness=2
                               )
        image = cv2.drawMarker(image, 
                               (predictions[traj_id][args.obs_length:args.obs_length+args.pred_length, 0], 
                                predictions[traj_id][args.obs_length:args.obs_length+args.pred_length, 1]),
                               color=color[traj_id],
                               markerType=cv2.MARKER_CSTAR,
                               thickness=2
                               )
    return image
# %%
def draw_heatmap(mux, muy, sx, sy, rho, ax, bound=1):
    x, y = np.mgrid[slice(mux - bound, mux + bound, 0.1),
                    slice(muy - bound, muy + bound, 0.1)]
    
    mean = [mux, muy]

    # Extract covariance matrix
    cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    
    gaussian = multivariate_normal(mean = mean, cov = cov)
    d = np.dstack([x, y])
    z = gaussian.pdf(d)

    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    
    ax.pcolormesh(x, y, z, cmap='PiYG', vmin=z_min, vmax=z_max, alpha=0.6) # cmap='bwr'

def visual(data_file=None):

    # if no specific data_file, choose the latest one
    if data_file is None:        
        result_dir = pkg_src_dir+"/results/"
        lists = os.listdir(result_dir)
        lists.sort(key=lambda fn: os.path.getmtime(result_dir+"/"+fn))
        data_file = os.path.join(result_dir, lists[-1])

    with open(data_file, "rb") as f:
        visual_data = pickle.load(f)

    pred_trajs   = visual_data[0] # list(array(seq_length, 2))
    truth_trajs  = visual_data[1] # list(array(seq_length, 2))
    gauss_params = visual_data[2] # list(array(pred_length, 5))
    obs_length = int(pred_trajs[0].shape[0] - gauss_params[0].shape[0])

    traj_num = len(pred_trajs)
    
    fig_width, fig_height = 5, 5

    fig, ax = plt.subplots(1,1,figsize=(fig_width, fig_width))
    
    x_min = min(np.array(truth_trajs)[:, :, 0].min(), np.array(pred_trajs)[:, :, 0].min())-0.1
    x_max = max(np.array(truth_trajs)[:, :, 0].max(), np.array(pred_trajs)[:, :, 0].max())+0.1
    bound = x_max-x_max
    y_min = min(np.array(truth_trajs)[:, :, 1].min(), np.array(pred_trajs)[:, :, 0].min())-0.1
    y_max = max(np.array(truth_trajs)[:, :, 1].max(), np.array(pred_trajs)[:, :, 0].max())+0.1
    
    for index in range(traj_num):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title("Predict %3s out of %s trajectories" % (index+1,traj_num ))
        visual_trajectories(pred_trajs[index], truth_trajs[index], gauss_params[index], obs_length, ax)     
        plt.cla()
        # plt.pause(5)

def visual_trajectories(pred_trajs, true_traj, gauss_param, obs_length, ax):
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    if true_traj is None:
        for pred in range(gauss_param.shape[1]):
            for traj_id in range(gauss_param.shape[0]):
                # plot predicted traj
                ax.plot(pred_trajs[traj_id, 0:obs_length+pred+1, 0], pred_trajs[traj_id, 0:obs_length+pred+1, 1],
                        color=colors[traj_id], linestyle='-', linewidth=1,marker='o', markersize=2, 
                        markeredgecolor=colors[traj_id], markerfacecolor=colors[traj_id])
        
                draw_heatmap(*gauss_param[traj_id,pred,:], ax=ax, bound=10)
                plt.pause(1)
    else:
        for pred in range(gauss_param.shape[0]):
            # plot predicted traj
            ax.plot(pred_trajs[0:obs_length+pred+1, 0], pred_trajs[0:obs_length+pred+1, 1],
                    color = 'g', linestyle = '-', linewidth = 1,
                    marker = 'o', markersize = 7, markeredgecolor = 'g', markerfacecolor = 'g')
            # plot ground truth traj
            ax.plot(true_traj[0:obs_length+pred+1, 0], true_traj[0:obs_length+pred+1, 1], 
                    color = 'r', linestyle = '-', linewidth = 1,
                    marker = 'o', markersize = 5, markeredgecolor = 'r', markerfacecolor = 'r')
    
            draw_heatmap(*gauss_param[pred], ax=ax, bound=10)
            plt.pause(1)

def visual_for_real(pred_trajs, gauss_params):
    """visualization for real test, instead of testing in dataset

    Args:
        pred_trajs  (3D-array): array(batch_size, seq_length, 2))
        gauss_param (3D-array): array(batch_size, pred_length, 5)
    """
    
    obs_length = int(pred_trajs.shape[1] - gauss_params.shape[1])

    traj_num = pred_trajs.shape[0]
    
    fig_width, fig_height = 5, 5

    fig, ax = plt.subplots(1,1,figsize=(fig_width, fig_width))
    
    x_min = np.array(pred_trajs)[:, :, 0].min()-0.1
    x_max = np.array(pred_trajs)[:, :, 0].max()+0.1
    y_min = np.array(pred_trajs)[:, :, 1].min()-0.1
    y_max = np.array(pred_trajs)[:, :, 1].max()+0.1
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    visual_trajectories(pred_trajs, None, gauss_params, obs_length, ax)     
    plt.cla()
    # plt.pause(5)

if __name__=="__main__":
    visual()
    
# %%
