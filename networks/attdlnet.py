#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from collections import OrderedDict

from networks.attention_net import attention_net as attnet 

class full_connected(nn.Module):
    def __init__(self,arch):
            
      super(full_connected,self).__init__()      
      input_dim = arch['input_dim']
      hidden_dim = arch['hidden_dim']
      out_dim = arch['output_dim']
      dropout = arch['dropout']

      self.fc = nn.Sequential(
              nn.Linear(input_dim,hidden_dim),
              nn.ReLU(),
              nn.LayerNorm(1024),
              nn.Dropout(dropout),
              nn.Linear(hidden_dim,out_dim)
          )
    
    def forward(self,x):
      x = self.fc(x)
      return(x)



class outlayer(nn.Module):
  def __init__(self,param):
    super(outlayer,self).__init__()
    self.dim = param['norm']['dim']
    self.norm_layer = nn.LayerNorm(self.dim)

  def forward(self,x):
    
    self.norm_layer(x)
    x,i = torch.max(x,1)
    x = torch.flatten(x,start_dim=1)
    return(x)
  

class attdlnet(nn.Module):
  def __init__(self, ARCH, architecture= ['backbone'], path=None, path_append="", strict=False):
    super().__init__()
    self.ARCH = ARCH
    self.path     = path
    self.path_append = path_append
    self.strict      = False
    self.architecture = ARCH['architecture']

    self.model_blocks = []
    self.mode_name = []
    # get the model
    device = ARCH['device']

    #self.dev = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    self.total_weights = 0

    if 'backbone' in self.architecture: 
      #for param in ARCH['backbone'].items():
      #    print(param)

      backbone_model_path = os.path.join('networks',self.ARCH["backbone"]["name"] + '.py')
      bboneModule = imp.load_source("bboneModule",backbone_model_path)
      self.backbone = bboneModule.Backbone(params=self.ARCH["backbone"])

      name = str(len(ARCH['backbone']['encoders']))
      # train backbone?
      if not self.ARCH["backbone"]["train"]:
        for w in self.backbone.parameters():
          w.requires_grad = False
        name += 'bb'
        ## self.mode_name.append('bb')
      else:
        name += 'BB'
      
      self.mode_name.append(name)

      weights = sum(p.numel() for p in self.backbone.parameters())
      print("[INF][%s] Encoder Param: %d"%(__name__,weights))
      #self.model_blocks['backbone'] = self.backbone
      self.total_weights += weights
      self.model_blocks.append(("back",self.backbone))

    # ================================================================================
    # Attention 

    if 'attention' in self.architecture: 
      
      #for param in ARCH['attention'].items():
      #  print(param)
      
      self.attention = attnet(ARCH['attention'])

      name = str(ARCH['attention']['n_layers'])
      # Freeze parameters
      if not self.ARCH["attention"]["train"]:
        for w in self.attention.parameters():
          w.requires_grad = False
          name +='a'
      else:
          name +='A'
      
      self.mode_name.append(name)

      weights = sum(p.numel() for p in self.attention.parameters())
      print("[INF][%s] Attention Param Param: %d"%(__name__,weights))
      self.total_weights += weights
      #self.model_blocks['attention'] = self.attention
      self.model_blocks.append(("att",self.attention))
    # ================================================================================
    # Full connected 

    if 'fc' in self.architecture:
      
      for param in ARCH['fc'].items():
        print(param)

      self.fc = full_connected(ARCH['fc'])
      # Freeze parameters
      if not self.ARCH["fc"]["train"]:
        for w in self.classifier.parameters():
          w.requires_grad = False
      # Print model size
      weights = sum(p.numel() for p in self.fc.parameters())
      print("[INF][%s] Full connected Param: %d"%(__name__,weights))
      self.total_weights += weights
      self.model_blocks['fc'] = self.fc

    # ================================================================================

    self.olayer = outlayer(self.ARCH["outlayer"])

    #self.model_blocks['outlayer'] = self.olayer
    
    
    if not self.ARCH["outlayer"]["train"]:
      for w in self.olayer.parameters():
          w.requires_grad = True
      self.mode_name.append('n')  
    else:
      self.mode_name.append('N')  
    
    weights = sum(p.numel() for p in self.olayer.parameters())
    print("[INF][%s] Norm Param: %d"%(__name__,weights))
    self.total_weights += weights
    self.model_blocks.append(("out",self.olayer))
   # ------------------------------------------------------------------------------------------
    self.net = nn.Sequential(OrderedDict(self.model_blocks))
  
    print("[INF][%s] Total parameters: %d "%(__name__,self.total_weights))


  def forward(self, x):
    
    x = self.net(x)
    
    return x

  def get_parm_size(self):
    return(self.total_weights)
  # ------------------------------------------------------------------------------------------

  def load_checkpoint(self,file):
    # Load pretrained weights  

    # get weights
    if path is not None:
      
      # Load backbone model weights from rangenet ++
      if ARCH["backbone"]['name']  in path: # Only backbone weights are available
        
        try:
          w_dict = torch.load(path + "/backbone" + path_append,
                              map_location=lambda storage, loc: storage)
          self.backbone.load_state_dict(w_dict, strict=True)

          print("Successfully loaded model backbone weights")

        except Exception as e:
          print()
          print("Couldn't load backbone, using random weights. Error: ", e)
          if strict:
            print("I'm in strict mode and failure to load weights blows me up :)")
            raise e
   
    else:
      print("No path to pretrained, using random init.")


   # ------------------------------------------------------------------------------------------

  def save_checkpoint(self, logdir, suffix=""):
    # Save the weights
    torch.save(self.backbone.state_dict(), logdir +
               "/backbone" + suffix)

    torch.save(self.attention.state_dict(), logdir +
               "/attention" + suffix)

    torch.save(self.otlayer.state_dict(), logdir +
               "/outputlayer" + suffix)

  def get_model_name(self):
    return(''.join(self.mode_name)) 

def load_model_weights(name,model,weigths):

  pretrained_dict = weights_parser(name,weigths)
  model_dict = model.state_dict()
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)
  return(model)

def weights_parser(name,w_dict):
  pretrained_dict = {}
  query = name + '.'
  for k, v in w_dict.items():
    if query in k :
      sk = k.split(query)[1] 
      pretrained_dict[sk] =  v 
  return(pretrained_dict)





