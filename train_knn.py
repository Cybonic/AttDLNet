#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


# Getting latend space using Hooks :
#  https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

# Binary Classification
# https://jbencook.com/cross-entropy-loss-in-pytorch/


'''

Version: 3.1 
 - pretrained model is automatically loaded based on the model and session names 
 
'''
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

from inference import eval_net
from dataset_utils.kitti import parser_test as test_pars
from dataset_utils.kitti import parser_train as train_pars

from utils.session_plot_utils import pose_plots,metric_plots,loss_plots,distribution_plots

#from modules.rattnet import *
from datetime import datetime

import random
from torch.utils.data import DataLoader, random_split
from utils.utils import dump_info


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
      #default='cross_val_00',
      default='cosine_small_session',
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--pretrained', '-p',
      type=str,
      required=False,
      default="darknet53-512",
      #default="checkpoints/isr/sim_isr_1_attention_cross_val_00_f1_78.pth",
      help='Directory to get the trained model.'
  )

  parser.add_argument(
      '--debug', '-b',
      type=int,
      required=False,
      default=False,
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
      default='session_results.txt',
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
  
  
  data_root_path = DATA['dataset']['path']['root']
  
  training_setup = SESSION['train']
  
  print("[WARN] Loading training parm from yam file")

  # Loading dataset 
  train_dataset = train_pars.Parser(
                                  dataset = DATA["dataset"],
                                  session = SESSION['train'])
  test_dataset = test_pars.Parser(
                                  dataset = DATA["dataset"],
                                  session = SESSION['test'])

  train_loader = train_dataset.get_set()

  val_loader   = test_dataset.get_set()
  triplet_idx  = test_dataset.get_triplets() 
  
  # Loading the network and pretrained weights
  model = attdlnet(ARCH)
  model_name = model.get_model_name()
  
  # Load Pretrained weights  
  pretrained_root = SESSION['pretrained_root']
  pretrained = os.path.join(pretrained_root,FLAGS.pretrained + '.pth')
  if os.path.isfile(pretrained):
    pretrained_to_load = pretrained

  else: 
    # If no pretrained model is given, try to load a pretrained model with the session name
    session_name  = FLAGS.sess_cfg
    pretrained_to_load = os.path.join(pretrained_root,model_name + '_' + session_name + '.pth')
  try:
    # Verify if the selected pretrained exists 
    if os.path.isfile(pretrained_to_load):
      model.load_state_dict(torch.load(pretrained_to_load))
      print("[INF] Pretrained model was loaded: " + pretrained_to_load)
    else: 
      print("[INF] No pretrained model loaded!")
  except:
      print("[WRN] Something went wrong while loading the pretrained model!")


  # Device configuration
  device = 'cpu'
  if torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()

  model.to(device)
  model.train()
  
  # Loss function
  loss_cf = SESSION['loss_function']
  print("\n----------")
  print("Loss Function Param:")
  print("Margin: ", loss_cf['margin'])
  print("Reduction: ", loss_cf['reduction'])

  criterion = torch.nn.CosineEmbeddingLoss(margin  = loss_cf['margin'],reduction= loss_cf['reduction'])

  print("\n----------")
  print("Data Param:")
  print("Root:              ", data_root_path)
  print("Batch Size:        ", training_setup['batch_size'])
  #print("Dataset Fraction:  ", fraction)
  
  # Optimizer parameterization
  lr            = ARCH['optimizer']['lr']
  w_decay       = ARCH['optimizer']['w_decay']
  amsgrad       = ARCH['optimizer']['amsgrad']
  epsilon       = ARCH['optimizer']['epsilon_w']
  betas         = tuple(ARCH['optimizer']['betas'])

  print("\n----------")
  print("Optimization Param:")
  print("Learning rate: ", lr)
  print("Weight decay:  ",w_decay)
  print("amsgrad:       ",amsgrad)
  print("epsilon:       ", epsilon)
  print("Betas:         ", betas)
  print("----------\n")

  optimizer = optim.Adam(
              model.parameters(), 
              lr = lr,
              weight_decay=w_decay,
              eps = epsilon,
              amsgrad = amsgrad,
              betas = betas
              )


  if FLAGS.plot == 1:
    # Create plot instances 
    plot_loss = loss_plots('Loss')

    plot_train_distro = distribution_plots([0,1],'train')
    plot_val_distro   = distribution_plots([0,1],'val')
    plot_val_metrics  = metric_plots('val metrics')

    plot_pose = pose_plots('map')
    plot_pose.update(ref = triplet_idx['poses'])

  # Create a global dictionary with all important variables 
  global_val_score = {'f1':-1,
                      'r':-1,
                      'p':-1,
                      'a':-1,
                      'loss':2,
                      'fps':-1,
                      'epoch':-1 }
  mean_fps_array= []
  
  # Root path to store models weights
 
  if not os.path.isdir(pretrained_root):
    # If it does not exist then create the folder 
    os.makedirs(pretrained_root)

  # Training parameters
  epochs          = training_setup['max_epochs']
  VAL_EPOCH       = training_setup['report_val']
  eval_criterion  = torch.nn.CosineSimilarity(dim=1)

  print("[INF] Loaded Model: " + model_name)
  print("[INF] Device: " + device)
  print("[INF] Result File: " + FLAGS.results)
  print("[INF] Session: " + SESSION['name'])

  try:
    for epoch in range(epochs):
          
      running_loss  = 0
      gt_true = np.array([])
      sim_bag = np.array([])
      loss_bag = np.array([])
      
      sub_epoch = epoch
      xx = np.array([])
      itr = 1/len(train_loader)
      
      for projA,projB, gt_label in tqdm.tqdm(train_loader):
        
        projA = projA.to(device)
        projB = projB.to(device)

        gt_label = gt_label.to(device).view(-1,1)
        
        optimizer.zero_grad()

        # compute output
        fa = model(projA)
        fb = model(projB)

        loss = criterion(fa,fb,gt_label)
          
        loss.backward()
        optimizer.step()
        
        scores = eval_criterion(fa,fb).detach().cpu().numpy()
        
        labels = gt_label.detach().cpu()
        loss = loss.detach().item()
        running_loss += loss
        
        xx = np.append(xx,round(sub_epoch,3))
        sub_epoch += itr
        gt_true = np.append(gt_true,labels)
        sim_bag = np.append(sim_bag,scores)
        #loss_bag = np.append(loss_bag,loss)

      train_loss = running_loss/len(train_loader)
      gt_true[gt_true==-1] = 0
      
      if FLAGS.plot == 1:
        plot_loss.update(data='mean',scores=train_loss,x=epoch+1)
      
      print("train epoch : {}/{}, loss: {:.6f}".format(epoch, epochs, train_loss))
      
      # Plotting stuff

      weigths_file = '%s_%s.pth' % (model_name,FLAGS.sess_cfg)
      trained_weights = os.path.join(pretrained_root,weigths_file)
      torch.save(model.state_dict(), trained_weights)

      if epoch % VAL_EPOCH == 0:
        
        metric,val_poses,text = eval_net( model,
                                          test_dataset, 
                                          device,
                                          top_candid = 10,
                                          range_thres = 6
                                          ) 
        
        mean_fps_array.append(metric['fps'])
        
        if FLAGS.plot == 1:
          plot_val_metrics.update(epoch = epoch,f1 = metric['f1'],acc = metric['a'])
          #plot_val_distro.update( loops['labels'],loops['scores'])
          plot_pose.update(query = val_poses['query'],
                    tn =  val_poses['tn'],
                    tp = val_poses['tp'],
                    #fp  = val_poses['fp'],
                    fn =  val_poses['fn'])
      
        if metric['f1'] > global_val_score['f1']: 
          
          weigths_file = '%s_%s_best.pth' % (model_name,SESSION['name'])
          trained_weights = os.path.join(pretrained_root,weigths_file)
          torch.save(model.state_dict(), trained_weights)
          # Overwite
          global_val_score  = metric
          global_val_score['loss']  = train_loss
          global_val_score['epoch'] = epoch

          print("[INF] weights stored at: " + weigths_file)
  
  except KeyboardInterrupt:
    print("[INF] CTR + C")
  
  except:
    print("[INF] Error")

  root = 'results/' + model_name + '_' + SESSION['name']
  
  if FLAGS.plot == 1:
    plot_loss.save_data_file(root) 
    plot_val_metrics.save_data_file(root)
  
  text_to_store = {}
  text_to_store['model'] = model_name
  text_to_store['session'] =   SESSION['name']
  text_to_store['F1'] =  round(global_val_score['f1'],3)
  text_to_store['R'] =   round(global_val_score['r'],3)
  text_to_store['P'] =   round(global_val_score['p'],3)
  text_to_store['A'] =   round(global_val_score['a'],3)
  text_to_store['epoch'] =  "%d/%d"%(global_val_score['epoch'],epochs)
  text_to_store['param'] = model.get_parm_size()
  text_to_store['FPS'] =   round(np.mean(mean_fps_array),1)

  output_txt = dump_info( FLAGS.results, text_to_store,'a')
  print("[INF] " + output_txt)


