from utils.dynamic_plot_lib_v3 import dynamic_plot
import numpy as np


class pose_plots():
  def __init__(self,title):

    self.pos_label = 1
    self.neg_label = 0

    self.plot = dynamic_plot(title,'Epoch','scores')
    #self.plot.add_plot('ref',color='k',save=True,window = 1,framework='scatter')
    self.plot.add_plot('tn',color='b',save=False,window = 1,framework='scatter')
    self.plot.add_plot('query',color='m',save=False,window = 1,framework='scatter',scale = 80)
    self.plot.add_plot('tp',color='g',save=False,window = 1,framework='scatter')
    #self.plot.add_plot('fp',color='r',save=False,window = 1,framework='scatter',marker = 'x',scale=80)
    
    self.plot.add_plot('fn',color='y',save = False, window = 1, framework='scatter', marker = 'x', scale=80)
  
  def update(self,**arg):
    
    for k in self.plot.plot_v.keys():
      if k in arg.keys():
        pose = arg[k]
        self.plot.update_plot(k,pose[:,0],pose[:,1])
    
    self.plot.show()
  
  def save_data_file(self,root): 
    self.plot.save_data_file(root)
  
  def hold(self):
    self.plot.hold_fig()


# ========================================================================


class metric_plots():
  def __init__(self,title):

    self.pos_label = 1
    self.neg_label = 0

    self.plot = dynamic_plot(title,'Epoch','scores')
    self.plot.add_plot('f1',color='g',save=True,window = 1)
    self.plot.add_plot('acc',color='r',save=True,window = 1)
  
  def update(self,**arg):
    
    f1 = np.array(arg['f1'])
    acc = np.array(arg['acc'])
    x = np.array(arg['epoch'])
    self.plot.update_plot('acc',x,acc)
    self.plot.update_plot('f1',x,f1)  
    
    self.plot.show()
  
  def save_data_file(self,root): 
    self.plot.save_data_file(root)
  
  def hold(self):
    self.plot.hold_fig()


# ========================================================================


class loss_plots():
  def __init__(self,title):

    self.pos_label = 1.0
    self.neg_label = 0

    self.loss = dynamic_plot(title,'Epoch','scores')
    self.loss.add_plot('pos',color='g',save=True,window = 20)
    self.loss.add_plot('neg',color='r',save=True,window = 20)
    self.loss.add_plot('mean',color='b',save=True,window = 3)
  
  def update(self,data,**arg):
    
    if data != 'mean':
      scores = np.array(arg['scores'])
      labels = np.array(arg['labels']) 
      sub_epochs = np.array(arg['x'])

      positives = scores[labels == self.pos_label]
      pos_xx    = sub_epochs[labels ==self.pos_label]
      negative  = scores[labels == self.neg_label]
      neg_xx    = sub_epochs[labels ==self.neg_label]
      
      self.loss.update_plot('pos',pos_xx,positives)
      self.loss.update_plot('neg',neg_xx,negative)
    
    else: 
      scores = arg['scores']
      x = arg['x']
      self.loss.update_plot('mean',x,scores) 

    self.loss.show()

  def save_data_file(self,root): 
    self.loss.save_data_file(root)
  
  def hold(self):
    self.loss.hold_fig()


# ========================================================================

class distribution_plots():
  def __init__(self,labels,title):
    
    self.pos_label = np.max(labels)
    self.neg_label = np.min(labels)
    
    self.histo = dynamic_plot(title,'scores','%')
    self.histo.add_plot('positive',color='g',save=False,framework = 'hist')
    self.histo.add_plot('negative',color='b',save=False,framework = 'hist')
    
  def update(self,labels,scores):
    scores = np.array(scores)
    labels = np.array(labels) 

    positives = scores[labels ==self.pos_label]
    negative = scores[labels == self.neg_label]
    
    self.histo.update_plot('positive',positives,[])
    self.histo.update_plot('negative',negative,[])
    self.histo.show()
  
  def hold(self):
    self.histo.hold_fig()
  
  def save_data_file(self,root): 
    self.histo.save_data_file(root)