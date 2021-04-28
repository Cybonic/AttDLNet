from utils.results_perser_utils import conv2dic
from utils.dynamic_plot_lib_v3 import dynamic_plot 
import argparse
import os
import numpy as np
import math
import pandas as pd


class plots():
  def __init__(self,title,grid_on = True):
    SIZE = 25
    fontsize = {'text':15,
                'xtick':SIZE,
                'ytick':SIZE,
                'title':15,
                'axis':SIZE,
                'legend':SIZE,
                'labels':SIZE,
                'figure':SIZE}

    self.plot = dynamic_plot(title,'N-Number of top candidates ','Recall@N',fontsize = fontsize)
    self.grid_on = grid_on
    
  def update(self,label,x,y,**arg):
    
    key  = label
    y = np.array(y)
    x = np.array(x)

    # default parameters
    color  = 'g'
    karg   = {'linestyle':'-'}

    if 'color' in arg:
      color = arg['color']
    
    if 'linestyle' in arg:
      karg['linestyle'] = arg['linestyle']

    self.axis_limit = {'xmin':min(x),'xmax':max(x),'ymin':0,'ymax':1}

    self.plot.add_plot( key,
                        color=color,
                        save=False,
                        scale = 10,
                        window = 0,
                        label = key,
                       **karg)
    
    self.plot.update_plot(key,x,y)


    self.plot.show(grid_on =self.grid_on,axis= self.axis_limit)
  
  def save_data_file(self,root): 
    self.plot.save_data_file(root)
  
  def hold(self):
    self.plot.hold_fig()

def file_parser(file):

    if not os.path.isfile(file):
        print("[INF] Result file does not exist!")
        raise Exception

    df = pd.DataFrame() 
    for line in open(file):
        line = line.strip()
        if line =='': # Empty line
            continue
        line = line.split("||")
        if line[1] == '': # Line transition
            continue
        header = line[0]
        data = line[1].split(' ')
        # Convert from str to dictionairy
        persed_data = conv2dic(data)
        # convert to pandas
        persed_data_DF = pd.DataFrame(persed_data,index = persed_data.keys())
        # append to global data frame
        df = df.append(persed_data_DF,ignore_index=True ) 
    return(df)

def compt_sequence_stats(results,seq,field):
  attention = range(5)
  net = 3
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

def compt_recall_stats(results, seq,models):

  top_cand_range = np.unique([int(v) for v in results.C])

  values = {'recall':[],'model':[],'can':[]}
  for enc,At in models.items():
    for at in At:
      model =  "E" + str(enc) + 'A'+str(at) 
      values['model'].append(model)
      recall_array = [] 
      top_cand = []
      for can in top_cand_range:
        recall = results[(results.modelA == at) & (results.modelB == enc) & (results.session == seq) & (results.C == can)]['R']
        
        mean_value = round(np.mean(recall[recall!=-1]),3)
        if math.isnan(mean_value) :
          continue
        
        recall_array.append(mean_value)
        top_cand.append(can)

      values['recall'].append(recall_array)
      values['can'].append(top_cand)
  return(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
      '--file', '-f',
      type=str,
      default = "results_paper/recall_results.txt",
      #default = "results_paper/attention_study.txt",
      required=False,
      help='Dataset to train with. No Default',
    )

    sequences = ['cross_val_00']
    
    fig = plots('',grid_on=True)

    FLAGS, unparsed = parser.parse_known_args()
    # Get File
    file_to_parse = FLAGS.file
    # Parse file
    results = file_parser(file_to_parse)
    # Demo: get all data belonging to cross_val_00
    
    # 
    colors = np.array(['g','b','y','k','r'])
    line = np.array(['-','--',':','-','--'])
    models = {3:[0,1,4],5:[0,3]}
    for seq in sequences:
      recall_curves = compt_recall_stats(results,seq,models)
      recall_a,model_a,top_a = list(recall_curves.values())
      for i,(recall,model,top) in enumerate(zip(recall_a,model_a,top_a)):
        fig.update(label = model,
                  y = recall[0:11],
                  x =top[0:11],
                  color = colors[i],
                  linestyle = line[i]
                  )
    
    
    fig.hold()
   

