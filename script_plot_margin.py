from results_perser_utils import file_parser
from dynamic_plot_lib_v3 import dynamic_plot 
import argparse
import os
import numpy as np


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

    self.plot = dynamic_plot(title,'Margin','F1',fontsize = fontsize)
    self.grid_on = grid_on
    
  def update(self,**arg):
    
    key  = arg['label']
    f1     = np.array(arg['f1'])
    margin = np.array(arg['margin'])
    color  = 'g'
    karg   = {'linestyle':'-'}

    if 'color' in arg:
      color = arg['color']
    
    if 'linestyle' in arg:
      karg['linestyle'] = arg['linestyle']

    idx = np.argmax(f1)
    m_max,f1_max = margin[idx],f1[idx]

    self.plot.add_plot( key,
                        color=color,
                        save=True,
                        scale = 5,
                        window = 1,
                        label = key + " (best margin = %s)"%(m_max),
                        **karg)
    
    self.plot.add_plot('scatter',
                    color=color,
                    save=True,
                    window = 1,
                    scale= 70,
                    framework='scatter'
                    )
    # Add line data
    self.plot.update_plot(key,margin,f1)
    # Add best point 
    self.plot.update_plot('scatter',m_max,f1_max,color = color)
    # Add fill area on the graph
    if 'fill' in arg:
      self.plot.addon(key, fill = arg['fill'])

    axis_limit = {'xmin':0,'xmax':0.95,'ymin':0,'ymax':1}
    self.plot.show(grid_on =self.grid_on,axis= axis_limit)
  
  def save_data_file(self,root): 
    self.plot.save_data_file(root)
  
  def hold(self):
    self.plot.hold_fig()


def compt_statics(results):
  margin_values = results['margin']
  unique_margin_values = np.unique(margin_values)
  f1_array = {'mean':[],'std':[],'x':unique_margin_values}
  for value in  unique_margin_values:
    F1values = results[results.margin == value]['F1']
    mean_f1 = round(np.mean(F1values),3)
    std_f1  = round(np.std(F1values),3)
    f1_array['mean'].append(mean_f1)
    f1_array['std'].append(std_f1)

  return(f1_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
      '--file', '-f',
      type=str,
      default = "results/margin_study.txt",
      required=False,
      help='Dataset to train with. No Default',
    )

    FLAGS, unparsed = parser.parse_known_args()
    # Get File
    file_to_parse = FLAGS.file
    # Parse file
    results = file_parser(file_to_parse)
    # Demo: get all data belonging to cross_val_00
    

    
    #margin_fig = plots('Margin Tuning',grid_on=False)
    #cross_val_00 = results[results.session == 'cross_val_00']
    #margin_fig.update(color = 'g',label = '00',f1 = cross_val_00['F1'],margin=cross_val_00['margin'])
    #cross_val_02 = results[results.session == 'cross_val_02']
    #margin_fig.update(color = 'r',label = '02',f1 = cross_val_02['F1'],margin=cross_val_02['margin'])
    #cross_val_05 = results[results.session == 'cross_val_05']
    #margin_fig.update(color= 'b',label = '05',f1 = cross_val_05['F1'],margin=cross_val_05['margin'])
    #cross_val_06 = results[results.session == 'cross_val_06']
    #margin_fig.update(color ='y',label = '06',f1 = cross_val_06['F1'],margin=cross_val_06['margin'])
    #cross_val_08 = results[results.session == 'cross_val_08']
    #margin_fig.update(color = 'k',label = '08',f1 = cross_val_08['F1'],margin=cross_val_08['margin'])

    statistic = compt_statics(results)
    
    print(statistic['mean'])
    print(statistic['std'])

    margin_mean_fig = plots('',grid_on=False)
    margin_mean_fig.update(color = 'k',
                      label = 'mean',
                      f1 = statistic['mean'],
                      margin= statistic['x'],
                      linestyle='--',
                      fill = statistic['std']
                      )

    #margin_fig.hold()
    margin_mean_fig.hold()