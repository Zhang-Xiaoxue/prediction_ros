#!/usr/bin/python

# Copyright 2010 Ankur Sinha
# Author: Ankur Sinha
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# File : extractRawInfo.py
#
# %%
import rosbag
import sys
import os
import pickle
import argparse
import csv
# from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
from tqdm import tqdm

# Global variable for input file name
current_path = os.path.abspath(__file__)
data_dir = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),'data')
bagfile_dir = data_dir+"/bagfiles/"

# %%
def run(bagname:str ="rosworld1.bag"):
    """
    Main run method. Calls other helper methods to get work done
    # """
    # if len(sys.argv) != 2:
    #     sys.stderr.write('[ERROR] This script only takes input bag file as argument.n')
    # else:
    #     inputFileName = sys.argv[1]
    inputFileName = bagfile_dir + bagname
    print ("[OK] Found bag: %s" % inputFileName)

    bag = rosbag.Bag(inputFileName)
            
    topicList = readBagTopicList(bag)

    while True:
        if len(topicList) == 0:
            print ("No topics in list. Exiting")
            break

        selection  = menu(topicList)

        if selection == -92:
            print ("[OK] Printing them all")
            for topic in topicList:
                extract_data(bag, topic, inputFileName)
            break
        elif selection == -45:
            break
        else:
            topic = topicList[selection]
            extract_data(bag, topic, inputFileName)
            topicList.remove(topicList[selection])

    bag.close()

def extract_data (bag, topic, inputFileName):
    """
    Spew messages to a file

    args:
        topic -> topic to extract and print to csv file
    """    
    if topic.endswith('yolov7_detection'):
        rect_topic = '/yolov7/yolov7_detection'
        outputFileName = data_dir + '/ros_simu/' + os.path.splitext(os.path.split(inputFileName)[1])[0] + rect_topic.replace("/","-") + ".csv"
        print ("[OK] Printing %s" % topic)
        print ("[OK] Output file will be called %s." % outputFileName)
        
        seq_id_list, label_list, label_id_list, pos_x_list, pos_y_list = [], [], [], [], []
        
        count = 0
        
        # save the initial pos_x and pos_y, for the next step to check the person whether he is static or moving
        static_pos_list = []
        for topic, msg, t in bag.read_messages(topics=topic):
            num_person_detected = len([det_obj.label_id for det_obj in msg.objects if det_obj.label=='person'])
            if num_person_detected > 1:
                print("####### More than 1 person are detected")
                print("number of detected persions: ", num_person_detected)
            for det_obj in msg.objects:
                if det_obj.label=='person':
                    if (os.path.splitext(os.path.split(inputFileName)[1])[0] == 'rosworld2') and abs(det_obj.confidence-0.86415)<=0.001:
                        pass
                    else:
                        static_pos_list.append([det_obj.center.x, det_obj.center.y])
            break
        
        # Start to check and save data
        for topic, msg, t in bag.read_messages(topics=topic):    
            num_person_detected = len([det_obj.label_id for det_obj in msg.objects if det_obj.label=='person'])
            
            if num_person_detected > 1:   
                count = 0      
                pos_list = []
                for det_obj in msg.objects:
                    if det_obj.label=='person':
                        # check person is static or moving
                        check_list = []   
                        for static_pos in static_pos_list:
                            check_list.append(np.linalg.norm([det_obj.center.x-static_pos[0], det_obj.center.y-static_pos[1]]) < 21)
                        if True not in check_list:
                            count += 1              
                            # save to result lists
                            seq_id_list.append(msg.header.seq)
                            label_list.append(det_obj.label)
                            label_id_list.append(det_obj.label_id)
                            pos_x_list.append(det_obj.center.x)
                            pos_y_list.append(det_obj.center.y)
                            pos_list.append([det_obj.center.x, det_obj.center.y])
                # print("moving person: ", count)
                if count > 1:
                    raise UserWarning("More than 1 moving person. Data processing may be wrong!")
                
            elif num_person_detected == 1:
                print("####### Only 1 moving person is detected")
                for det_obj in msg.objects:
                    # TODO : only extract label='person' detection data
                    if det_obj.label == 'person': 
                        # print(det_obj.label_id)
                        seq_id_list.append(msg.header.seq)
                        label_list.append(det_obj.label)
                        label_id_list.append(det_obj.label_id)
                        pos_x_list.append(det_obj.center.x)
                        pos_y_list.append(det_obj.center.y)
        
        new_label_id_list = process_label_id_lists(seq_id_list, label_list, label_id_list)
        
        with open(outputFileName, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([seq_id_list, new_label_id_list, pos_x_list, pos_y_list])
    
        print ("[OK] DONE -- convert to csv")
        
    
    elif topic.endswith('yolov7_detection/visualization'):
        outputFileDir = data_dir + '/ros_simu_images/'
        # outputFileDir = outputFileDir + inputFileName
        print ("[OK] Printing %s" % topic)
        print ("[OK] Output images will be stored in %s." % outputFileDir)
        
        bridge = CvBridge()
        total_frames = bag.get_message_count(topic) #Total frames in topic
        print("\nTotal frames in bag file are:{}".format(total_frames))
        for topic, msg, t in bag.read_messages(topics=topic):
            print("Size of the image: W {} x H {}".format(msg.width, msg.height))
            print("Encoding of the frames: {}".format(msg.encoding))
            break
        
        # find the images related to the csv file
        count = 0
        with tqdm(total=total_frames, desc='writing frames:', leave=True) as pbar:
            for topic, msg, t in bag.read_messages(topics=topic):
                cv_img = np.asarray(bridge.imgmsg_to_cv2(msg, "8UC3")) # RGB output "mono8"

                p = outputFileDir + os.path.splitext(os.path.split(inputFileName)[1])[0] + topic.replace("/","-") + "_{:05}".format(count)+".png"
                cv2.imwrite(p, cv_img)
                
                count += 1
                pbar.update(1)
        
        print ("[OK] DONE -- convert to images")

def process_label_id_lists(seq_id_list, label_list, label_id_list):
    # check_results([seq_id_list, label_list, label_id_list, pos_x_list, pos_y_list])
    seq_id_list_diff = diff_list(seq_id_list)
    # print(get_number_of_elements(seq_id_list_diff))
    
    # TODO Now, I just delete all labels except "person". Late, process data for more labels
    
    # slice the list and change the label_id for each segments    
    seg_idx_list = [0,]
    # find the seq_id which is > 1 for segments
    for seq_id in [ele for ele in set(seq_id_list_diff) if ele > 1]: 
        # find the index of the seq_id, indicating where to segment
        for seg_id_idx in [idx for idx,ele in enumerate(seq_id_list_diff) if ele==seq_id]: 
            seg_idx_list.append(seg_id_idx+1) # from seg_id_idx+1, start to change label_id
    seg_idx_list.append(len(seq_id_list))
    
    seg_idx_list.sort()
    
    new_label_id_list = []
    label_id = 1
    for i in range(len(seg_idx_list)-1):
        new_label_id_list.extend([label_id for _ in range(seg_idx_list[i], seg_idx_list[i+1]) ])
        label_id += 1
    
    return new_label_id_list

            
def diff_list(list_):
    list_diff = []
    for n in range(1, len(list_)):
        list_diff.append(list_[n] - list_[n-1])
    return list_diff

def list_duplicates(seq):
    from collections import defaultdict
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)
    
def check_results(all_list=[]):
    for list_ in all_list:
        print('----------------------------------')
        print(get_number_of_elements(list_))
        
        for dup in sorted(list_duplicates(list_)):
            print(dup)


def get_number_of_elements(list_):
    res_dict = {}
    for key in set(list_):
        count = 0
        for item in list_:
            if item == key:
                count += 1
        res_dict[key] = count    
    return res_dict

def menu (topicList):
    """
    Print the user menu and take input

    args:
        topicList: tuple containing list of topics

    returns:
        selection: user selection as integer
    """

    i = 0
    for topic in topicList:
        print ('[{0}] {1}'.format(i, topic))
        i = i+1
    if len(topicList) > 1:
        print ('[{0}] Extract all'.format(len(topicList)))
        print ('[{0}] Exit'.format(len(topicList) + 1))
    else:
        print ('[{0}] Exit'.format(len(topicList)))

    if len(topicList) == 1:
        return 0
    else:
        print ('Enter a topic number to extract raw data from:')
        selection = input('>>>')
        if int(selection) == len(topicList):
            return -92 # print all
        elif int(selection) == (len(topicList) +1):
            return -45 # exit
        elif (int(selection) < len(topicList)) and (int(selection) >= 0):
            return int(selection)
        else:
            print ("[ERROR] Invalid input")

def readBagTopicList(bag):
    """
    Read and save the initial topic list from bag
    """
    print ("[OK] Reading topics in this bag. Can take a while..")
    topicList = []
    for topic, msg, t in bag.read_messages():
        if topicList.count(topic) == 0:
            topicList.append (topic)

    print ('{0} topics found:'.format(len(topicList)))
    return topicList

# %%
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # current_path = os.path.abspath(__file__)
    # pkg_src_dir = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    # pkg_dir = os.path.abspath(os.path.dirname(current_path) + os.path.sep + "..")
    # bags_dir = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".."),'bagfiles')
    # parser.add_argument('--bag_file', type=str, 
    #                     default=bags_dir+"/ros_simu_data_1.bag",
    #                     help='bag files for convert')
    # args = parser.parse_args(args=[])
    run(bagname='rosworld1.bag')
# %%
