from results_perser_utils import file_parser
from dynamic_plot_lib_v3 import dynamic_plot 
import argparse
import os
import numpy as np
import math


class plots():
  def __init__(self,title,grid_on = True):
    SIZE = 12
    fontsize = {'text':15,
                'xtick':SIZE,
                'xtick':15,
                'title':15,
                'axis':SIZE,
                'legend':15,
                'labels':SIZE,
                'figure':SIZE}

    self.plot = dynamic_plot(title,'#Attention Layers','F1',fontsize = fontsize)
    self.grid_on = grid_on
    
  def update(self,**arg):
    
    key  = arg['label']
    f1     = np.array(arg['f1'])
    layers = np.array(arg['layers'],dtype=int)
    color  = 'g'
    karg   = {'linestyle':'-'}

    if 'color' in arg:
      color = arg['color']
    
    if 'linestyle' in arg:
      karg['linestyle'] = arg['linestyle']

    idx = np.argmax(f1)
    m_max,f1_max = layers[idx],f1[idx]
    self.axis_limit = {'xmin':min(layers),'xmax':max(layers),'ymin':0,'ymax':1}

    self.plot.add_plot( key,
                        color=color,
                        save=False,
                        scale = 5,
                        window = 0,
                        label = key + " (best layer = %s)"%(m_max),
                        **karg)
    
    self.plot.add_plot('scatter',
                    color=color,
                    save=False,
                    window = 0,
                    scale= 70,
                    framework='scatter'
                    )
    # Add line data
    self.plot.update_plot(key,layers,f1)
    # Add best point 
    self.plot.update_plot('scatter',m_max,f1_max,color = color)
    # Add fill area on the graph
    if 'fill' in arg:
      self.plot.addon(key, fill = arg['fill'])

    self.plot.show(grid_on =self.grid_on,axis= self.axis_limit)
  
  def save_data_file(self,root): 
    self.plot.save_data_file(root)
  
  def hold(self):
    self.plot.hold_fig()

def compt_sequence_stats(results,seq,field):
  attention = range(5)
  net = 5
  sessions = 'cross_val_' + '%02d'%(seq)
  net_values = {}
  values = {'mean':[],'std':[],'layers':[]}
  for att in attention:
    
    #for layer in layers:
    encoder = results[(results.session == sessions) & (results.modelA == att)& (results.modelB == net)][field]
    mean_value = round(np.mean(encoder[encoder!=-1]),3)
    std_value = round(np.std(encoder[encoder!=-1]),3)
    values['mean'].append(mean_value)
    values['std'].append(std_value)
    values['layers'].append(att)
  

  return(values)

def compt_attention_stats(results,filed):

  layers = range(5)
  encoder = 5
  values = {'mean':[],'std':[],'layers':[]}
  for layer in layers:
    attention = results[(results.modelA == layer) & (results.modelB == encoder)][filed]
    mean_value = round(np.mean(attention[attention!=-1]),3)
    if math.isnan(mean_value) :
      continue
    std_value = round(np.std(attention[attention!=-1]),3)
    if math.isnan(std_value):
      std_value= 0
    values['mean'].append(mean_value)
    values['std'].append(std_value)
    values['layers'].append(layer)
  return(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
      '--file', '-f',
      type=str,
      #default = "results/attention_select_study.txt",
      default = "results_paper/attention_study.txt",
      required=False,
      help='Dataset to train with. No Default',
    )

    FLAGS, unparsed = parser.parse_known_args()
    # Get File
    file_to_parse = FLAGS.file
    # Parse file
    results = file_parser(file_to_parse)
    # Demo: get all data belonging to cross_val_00
    
    f1_scores = compt_attention_stats(results,'F1')
    
    print(f1_scores['mean'])
    fig = plots('',grid_on=False)
    sequences = [0,2,5,6,8]

    for seq in  sequences:
      f1_scores_08 = compt_sequence_stats(results,seq,'F1')
      print("F1:%d "%(seq))
      print(f1_scores_08['mean'])

    fig.update(color = 'k',
              label='mean',
              f1 = f1_scores['mean'],
              layers=f1_scores['layers'], 
              fill = f1_scores['std'],
              linestyle='--'
              )
    
    fig.update(color = 'k',
              label='08',
              f1 = f1_scores_08['mean'],
              layers=f1_scores_08['layers'],
              linestyle='-'
              )
    
    fp_scores = compt_attention_stats(results,'FPS')
    print(fp_scores['mean'])
    sequences = [0,2,5,6,8]
    for seq in  sequences:
      fps_scores = compt_sequence_stats(results,seq,'FPS')
      print("FPS:%d "%(seq))
      print(fps_scores['mean'])
    
    fig.hold()
   

