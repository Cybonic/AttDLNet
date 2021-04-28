#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


# Getting latend space using Hooks :
#  https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

# Binary Classification
# https://jbencook.com/cross-entropy-loss-in-pytorch/

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import signal, sys
from sklearn.neighbors import NearestNeighbors
from torch import optim
import torch 

from networks.attdlnet import *

from dataset_utils.kitti import parser_test as pars
from utils.session_plot_utils import pose_plots

import time

import random
from utils.dynamic_plot_lib_v3 import dynamic_plot
from torch.utils.data import DataLoader, random_split
from utils.utils import dump_info


def evaluation(local_pred,gt_idx,gt_loop):
  
  map_size = len(gt_loop)

  plot_eval = np.zeros((map_size,4))
  global_eval = np.array([0,0,0,0])

  retrieved_idx = np.array([],dtype=int)
  for lp,gt in zip(local_pred,gt_idx):
    local_eval = np.array([0,0,0,0])
    for p in lp:
      if p in gt: # True Positive
        tp,fp,tn,fn = 1,0,0,0
        retrieved_idx = np.append(retrieved_idx,p)
      else: # False positive
        tp,fp,tn,fn = 0,1,0,0
      # Save 
      plot_eval[p] = tp,fp,tn,fn
      local_eval +=[tp,fp,tn,fn]

    if local_eval[0] == 0: # False Negative
      local_eval += [0,0,0,1]
      plot_eval[gt[0]] = [0,0,0,1]
    
    global_eval += local_eval

  for label,idx in zip(gt_loop,range(map_size)):
    if idx in retrieved_idx:
      continue 
    if label == 0: 
      tp,fp,tn,fn = 0,0,1,0 # True Negative
      global_eval += [tp,fp,tn,fn]
      plot_eval[idx] = tp,fp,tn,fn

    metric = eval_metrics(global_eval)

  print(global_eval)

  return(metric,plot_eval)

def dump_info_to_file(**arg):
  root = arg['root']
  file_name = arg['name']
  data = arg['DATA']
  arch = arg['ARCH']
  session = arg['Session']

  if not os.path.isdir(root):
      os.makedirs(root)
  file = os.path.join(root,file_name + '.txt')
  print("[INF] Save Log at File: " + file)
  f = open(file,'w')

  txt = "{}:{}\n"

  for key, value in arch.items():
    print(txt.format(key,value))
    f.write(txt)
  
  for key, value in session.items():
    print(txt.format(key,value))
    f.write(txt)


def conv_label_back(labels,interv):
  low_out_label = min(interv)
  high_out_label = max(interv)

  low_in_label = min(labels)
  high_in_label = max(labels)

  if low_in_label != 1:
    labels[labels == low_in_label] = low_out_label
 
  return(labels)

def knn_place_recognition(place_models,place_map,top_candid = 1):

    scores,winners = [],[]
    sigma = 0.69
    neigh = NearestNeighbors(metric = 'cosine',p = 8, n_neighbors = top_candid )
    # only 1st visited places are used
    neigh.fit(place_map)

    for model in tqdm.tqdm(place_models,"Place recognition"):
    
        # Belief Generation
        ##########################################################
        score,nn = neigh.kneighbors(model.reshape(1,-1))
        winner,score = nn[0],score[0]
        
        score = 1-score
        
        winners.append(winner)
        scores.append(score)

    return(winners,scores)


def knn_metric(models,pose_map,top_candid = 10,range_value = 3):

    scores,winners = [],[]
    neigh = NearestNeighbors(metric = 'euclidean', radius = range_value , p = 8 )
    # only 1st visited places are used
    neigh.fit(pose_map)

    loop_labels = np.zeros(len(pose_map))

    for model in tqdm.tqdm(models,"Ground truth"):

        score,nn = neigh.radius_neighbors(model.reshape(1,-1))
        winner,score = nn[0],score[0]

        winners.append(winner)
        scores.append(score)
        loop_labels[winner] = 1

    return(winners,scores,loop_labels)


def eval_metrics(eval_scores):
  # tp,fp,tn,fn = eval_scores.sum(axis=0)
  tp,fp,tn,fn = eval_scores

  if tp + fp == 0:
        precision = 0
  else: 
      precision = tp/(tp + fp)
  
  if tp+fn == 0:
      recall =  0
  else:
      recall = tp/(tp+fn)


  if (precision + recall) == 0:
      f1 = 0
  else:
      f1 = 2 * (precision * recall) / (precision + recall)

  acc = (tp + tn)/(tp + tn + fp + fn)

  return({'f1':f1,'r':recall,'p':precision,'a':acc})


def eval_net(net, dataset, device,**arg):
  """Evaluation without the densecrf with the dice coefficient"""
  
  loader   = dataset.get_set()
  
  net.eval()
  n_val = len(loader)  # the number of batch
  range_thres = 6
  top_candid  = 1
  if 'range_thres' in arg:
    range_thres = arg['range_thres']
  if 'top_candid' in arg:
    top_candid = arg['top_candid']
  
  print("[INFO] top_candid: " + str(top_candid))

  criterion = torch.nn.CosineSimilarity(dim=1)
 
  FPS = []
  data = {'labels':[],'descriptors':[],'index':[]}
  
  with torch.no_grad():
    with tqdm.tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for anchor,label,indice in loader:
            
            start_time = time.time() # start time of the loop
            anchor = anchor.float().to(device)
            fa = net(anchor).detach().cpu().tolist()[0]
            # Compute processing time
            fps =  1.0 / (time.time() - start_time)
            FPS = np.append(FPS,fps)
            # Store data
            data['index'].append(indice.item())
            data['descriptors'].append(fa)
            data['labels'].append(label)
            pbar.update()
  
  poses  = dataset.get_triplets()['poses'] 

  labels = np.array(data['labels'],dtype = int)
  global_indx = np.array(data['index'],dtype= int)
  # Split elements belonging to map and do queries
  query_idx = np.where( labels == 2)[0]
  map_idx   = np.where( labels < 2)[0]
  # Compute Ground Truth  
  map_pose   = poses[global_indx[map_idx]]
  model_pose = poses[global_indx[query_idx]]
  gt_loop, dist,gt_loop_labels = knn_metric(model_pose,map_pose,range_value=range_thres)
  # Place recognition
  models = np.array(data['descriptors'])[query_idx]
  p_map  = np.array(data['descriptors'])[map_idx]
  loop_pred, scores = knn_place_recognition(models,p_map,top_candid=top_candid )
  # Evaluation
  metric,plot_eval_scores = evaluation(loop_pred,gt_loop,gt_loop_labels)
  # compute and store mean frame rate
  metric['fps'] = np.mean(FPS)
  # Process data for plotting 
  tp_idx = np.where(plot_eval_scores[:,0]==1)[0]
  fp_idx = np.where(plot_eval_scores[:,1]==1)[0]
  tn_idx = np.where(plot_eval_scores[:,2]==1)[0]
  fn_idx = np.where(plot_eval_scores[:,3]==1)[0]

  fp_pose = poses[global_indx[map_idx[fp_idx]]]
  tn_pose = poses[global_indx[map_idx[tn_idx]]]
  fn_pose = poses[global_indx[map_idx[fn_idx]]]
  tp_pose = poses[global_indx[map_idx[tp_idx]]]

  pred_map = {'query': poses[global_indx[query_idx]],
            'tp':tp_pose,
            'fp':fp_pose,
            'tn':tn_pose,
            'fn':fn_pose}

  text = "F1:%0.3f R:%0.3f P:%0.3f A:%0.3f FPS:%2.0f"%(  metric['f1'],
                                                            metric['r'],
                                                            metric['p'],
                                                            metric['a'],
                                                            metric['fps'])

  print("[INF] " + text)                                                      

  net.train()

  return(metric,pred_map,text)



if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      default = "kitti",
      required=False,
      help='Dataset to train with. No Default',
  )
  parser.add_argument(
      '--corr', '-c',
      type=str,
      default= ''
  )

  parser.add_argument(
      '--model', '-m',
      type=str,
      required=False,
      default='3bb_1a_norm',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--sess_cfg', '-f',
      type=str,
      required=False,
      default='cross_val_08',
      #default='cosine_small_session',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--pretrained', '-p',
      type=str,
      required=False,
      #default="checkpoints/darknet53-512",
      default="checkpoints/3BB0AN_cross_val_08_best",
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--debug', '-b',
      type=int,
      required=False,
      default=True,
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--plot',
      type=int,
      required=False,
      default=1,
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--results',
      type=str,
      required=False,
      default='ex8_3E_01_recall_results.txt',
      help='Directory to get the trained model.'
  )

  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("Dataset:", FLAGS.dataset)
  print("Sequence: ", FLAGS.corr)
  print("Model: ", FLAGS.model)
  print("Debug flag: ", FLAGS.debug)
  print("Pretrained flag: ", FLAGS.pretrained)
  print("----------\n")

  # open arch config file
  cfg_file = os.path.join('dataset_utils',FLAGS.dataset,'data_cfg_hd.yaml')
  try:
    print("Opening data config file: %s" % cfg_file)
    DATA = yaml.safe_load(open(cfg_file , 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  model_cfg_file = os.path.join('model_cfg', FLAGS.model + '.yaml')
  try:
    print("Opening model config file: %s" % model_cfg_file)
    ARCH = yaml.safe_load(open(model_cfg_file, 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  session_cfg_file = os.path.join('sessions', FLAGS.sess_cfg + '.yaml')
  try:
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  
  ###################################################################### 
  
  debug_flag   = FLAGS.debug
  dataset_name = FLAGS.dataset
 
  print("[WARN] Loading training parm from yam file")
 
  dataset = pars.Parser(
                        dataset = DATA["dataset"],
                        session = SESSION['test']
                        )

  # Loading pretrained weights
  model = attdlnet(ARCH)
  model_name = model.get_model_name()

  if os.path.isfile(FLAGS.pretrained + '.pth'):
    model.load_state_dict(torch.load(FLAGS.pretrained + '.pth'))
    print("[INF] Pretrained: " + FLAGS.pretrained)
  else: 
    print("[WRN] Pretrained Failed") 

  # Device configuration
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()

  model.to(device)
  model.train()

  print("[INF] Loaded Model: " + model_name)
  print("[INF] Device: " + device)
  print("[INF] Result File: " + FLAGS.results)

  metric,val_poses,text = eval_net( model,
                                    dataset, 
                                    device,
                                    range_thres = SESSION['test']['range_thres'],
                                    top_candid = SESSION['test']['top_cand']
                                  ) 
  text_to_store = {}
  text_to_store['model'] = model_name
  text_to_store['session'] =   SESSION['name']
  text_to_store['F1'] =  round(metric['f1'],3)
  text_to_store['R'] =   round(metric['r'],3)
  text_to_store['P'] =   round(metric['p'],3)
  text_to_store['A'] =   round(metric['a'],3)
  # text_to_store['epoch'] =  "%d/%d"%(metric['epoch'],epochs)
  text_to_store['param'] = model.get_parm_size()
  text_to_store['FPS'] =   metric['fps']

  output_txt = dump_info( FLAGS.results, text_to_store,'a')
  print("[INF] " + output_txt)

  if FLAGS.plot == True:
    plot_pose = pose_plots('Map')
    
    poses = dataset.get_triplets()['poses']

    plot_pose.update(ref = poses,
                    query = val_poses['query'],
                    tn =  val_poses['tn'],
                    tp = val_poses['tp'],
                    #fp  = val_poses['fp'],
                    fn =  val_poses['fn'])
 
    plot_pose.hold()


