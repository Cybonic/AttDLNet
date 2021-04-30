import os
import numpy as np
import torch
import dataset_utils.kitti.utils as utils
from torch.utils.data import Dataset
from common.laserscan import LaserScan
from torch.utils.data import DataLoader, random_split
from random import seed
import math

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def velo_read(scan_path):
  scan = np.fromfile(scan_path, dtype=np.float32)
  scan = scan.reshape((-1, 4))

  #scan = scan.reshape((-1,4))
  return(np.array(scan))

def load_tripletset(triplet_file,root,dataset_name,corr):
    # path to triplet file
    triplet_set_file = os.path.join(root,dataset_name,'tripletset',corr, triplet_file + '.txt')
    #print("[INF] Loading triplet file: " + triplet_set_file)
    assert os.path.isfile(triplet_set_file)
    # load content of the file (i.e. sequence and file names)
    triplet_set = load_triplet_file(triplet_set_file)
    # build root path to the descriptor folder
    idx = [int(i) for i in  triplet_set['files']]

    return(np.array(idx))

def load_triplet_file(file):
    output = {}
    for line in open(file):
        inf_type,info = line.strip().split(':')
        if inf_type == 'seq':
            output[inf_type] = info
        else: 
            output[inf_type] = info.split(' ')
    return(output)


def data_augment(rotations,anchor_unit,template_unit,label_unit,aug_percentage):
    
    aug_rot      = np.array([],dtype=int)

    n_samples = int(len(anchor_unit))
    aug_n_samples = math.ceil(len(anchor_unit)*aug_percentage) # Total samples to be added
    n_samples_per_rot = math.ceil(aug_n_samples/len(rotations)) # Samples per rotations
    
    # original dataset
    aug_rot      = 0*np.ones(n_samples) 
    aug_anchors  = anchor_unit
    aug_template = template_unit
    aug_labels   = label_unit

    # np.
    for rot in rotations:
      print("[INF] Augmenting data:%f"%(rot))
      idx = np.random.randint(n_samples, size=n_samples_per_rot)

      aug_anchors = np.append(aug_anchors,np.array(anchor_unit)[idx])
      aug_template = np.append(aug_template,np.array(template_unit)[idx])
      aug_labels = np.append(aug_labels,np.array(label_unit)[idx])
      aug_rot = np.append(aug_rot,rot*np.ones(n_samples_per_rot))
    
    return(aug_anchors,aug_template,aug_labels,aug_rot)

class SemanticKitti(Dataset):

  def __init__(self, 
               root,    # directory where data is
               sensor,              # sensor to parse scans from
               sequences = [],     # sequences for this data (e.g. [1,3,4,6])
               max_points=150000,   # max number of points present in dataset
               gt=True,        # send ground truth?
               range_thres = None,
               fraction = -1,
               debug = False):   # Limit the range of the scan        
    
    self.range_thres = range_thres
    
    #self.aug_percentage = aug_percentage
    # save deats
    self.root = root
    self.sequences = sequences
   
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt
    np.random.seed(0)
    debug_flag = debug
  

    print("[INF] Debug Flag: %d"%(debug_flag)) 
   
    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("[INF] Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.anchor_files   = []
    self.template_files = []
    self.labels         = []
    self.rot_angles     = []
    self.rot_fraction   = 0
    self.poses          = {}
    # fill in with names, checking that all sequences are complete
    n_positives = 0
    n_negatives = 0
    n_anchors = 0

    print("================================================")

    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))
      corr = 'ex%d'%(int(seq))

      pose_path = os.path.join(self.root, "sequences", seq,'poses.txt')
      self.poses = utils.poses(pose_path).get_2D_poses()
      
      print("[INF] Parsing seq {}".format(seq))


      # get paths for each
      scan_path = os.path.join(self.root, "sequences", seq, "velodyne")
       # get files
      self.scan_files = np.array([os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)])

      self.total_idx = np.arange(len(self.scan_files))

      print("[INF] Total Seq Files: {}".format(len(self.scan_files)))
      self.triplet_flag = True

      try:
        self.positive = load_tripletset('positive','data','kitti',corr)
        self.negative = load_tripletset('negative','data','kitti',corr)
        self.anchors  = load_tripletset('anchor'  ,'data','kitti',corr)
        self.map_ref = []
        
        # self.no_loop_pool = np.in1d(all_idx,self.loop_pool)
      except:
        print("[ERR] No Triplet data available!")
        self.triplet_flag = False

      if self.gt == True and self.triplet_flag == True: # Training

        neg_len = len(self.negative)
        pos_len = len(self.positive)
        anch_len = len(self.positive)
        
        self.negative_labels = -1 * np.ones(neg_len)
        self.positive_labels = np.ones(pos_len)

        plabel_len = len(self.positive_labels)
        nlabel_len = len(self.negative_labels)

        if debug_flag == True:
          print("Seq Anchors:   {}".format(anch_len))
          print("Seq Negatives: {}".format(neg_len))
          print("Seq Positives: {}".format(pos_len))
          print("Seq P. Labels: {}".format(plabel_len))
          print("Seq N. Labels: {}".format(nlabel_len))

        if fraction > 0:
           
          frac_samples = int(len(self.anchors)*fraction)
          
          step = int(len(self.anchors)/frac_samples)

          sample_idx = np.arange(0,len(self.anchors),step,dtype=int)

          #sample_idx = np.random.randint(0, len(self.anchors), frac_samples, dtype=int)
          
          self.anchors  = self.anchors[sample_idx]
          self.negative = self.negative[sample_idx]
          self.positive = self.positive[sample_idx]

          self.negative_labels = self.negative_labels[sample_idx]
          self.positive_labels = self.positive_labels[sample_idx]

          anch_len   = len(self.anchors)
          neg_len    = len(self.negative)
          pos_len    = len(self.positive)
          nlabel_len = len(self.negative_labels)
          plabel_len = len(self.positive_labels)

          if debug_flag == True:

            print("Fact Samples:   {}".format(frac_samples))
            print("Fact Anchors:   {}".format(anch_len))
            print("Fact Negatives: {}".format(neg_len))
            print("Fact Positives: {}".format(pos_len))
            print("Fact P. Labels: {}".format(plabel_len))
            print("Fact N. Labels: {}".format(nlabel_len))


        n_positives += len(self.positive)
        n_negatives += len(self.negative)
        n_anchors += len(self.anchors)

        self.anchor_unit   = np.append(self.anchors,self.anchors)
        self.template_unit = np.append(self.positive,self.negative)
        self.label_unit    = np.append(self.positive_labels,self.negative_labels)

        label     = self.label_unit
        template  = self.scan_files[self.template_unit]
        anchor    = self.scan_files[self.anchor_unit]
  
        # extend list
        self.anchor_files.extend(anchor)
        self.template_files.extend(template)
        self.labels.extend(label)

    print("[INF] Total Anchors:   {}".format(n_anchors))
    print("[INF] Total Negatives: {}".format(n_negatives))
    print("[INF] Total Positives: {}".format(n_positives))
    print("[INF] Total Labels:    {}".format(len(self.labels)))

    print("================================================")
      #self.scan_files.extend(scan_files)

  def set_rotation_set(self,angles,fraction):
      print("[INF] point clouds will be rotated" )
      aug = data_augment(angles,
                        self.anchor_files,
                        self.template_files,
                        self.labels,
                        fraction)

      aug_anchors, aug_template, aug_labels, rot = aug
      print("[INF] Aug Anchors Total: " + str(len(aug_anchors)))
      print("[INF] Aug template Total: " + str(len(aug_template)))
      print("[INF] Aug labels Total: " + str(len(aug_labels)))
      
      self.anchor_files    = aug_anchors
      self.template_files  = aug_template
      self.labels           = aug_labels
      self.rot_fraction    = fraction
      self.rot_angles       = rot
      # self.aug_rot.extend(aug_rot)

  def get_param(self):
    param = {}
    param['anch']  = len(self.anchor_files)
    param['temp'] = len(self.template_files)
    labels = np.array(self.labels)
    param['pos'] = len(labels[labels == 1])
    param['neg'] = len(labels[labels == 0])
    return(param)
  
  def get_idx(self):
    indices = {}
    indices['positives'] = self.positive
    indices['negatives'] = self.negative
    indices['anchors']   = self.anchors
    indices['all']       = self.sorted_index
    indices['map']        = self.map_ref
    indices['noloops'] = self.no_loop_pool
    
    #indices['no_loop_pool'] = self.no_loop_pool
    #indices['loop_pool'] = self.loop_pool
    indices['poses'] = self.poses
    return(indices)

  def get_files(self):
    files = {}
    
    files['scans'] = self.scan_files
    files['anch']  = self.anchors
    files['neg']   = self.negative
    files['pos']   = self.positive
    files['poses'] = self.poses
    
    return(files)


  def load_files(self,anchor,template,labels,rotation = None):
    self.anchor_files   = anchor
    self.template_files = template
    self.label = labels 
    self.rotate = rotation

  def __getitem__(self, index):
    # get item in tensor shape
    #anchor_file = self.scan_files[index]
    
    anchor_file   =  self.anchor_files[index]

    if self.gt == True:
      template_file =  self.template_files[index]

      label = self.labels[index]
      # open a semantic laserscan
      template_scan = LaserScan(project=True,
                        H=self.sensor_img_H,
                        W=self.sensor_img_W,
                        fov_up=self.sensor_fov_up,
                        fov_down=self.sensor_fov_down)

      # open and obtain scan
      template_scan.open_scan(template_file)
      projB = torch.from_numpy(template_scan.points)
      
    else: 
      projB = []
      label = self.labels[index]
     # Prediction

    anchor_scan = LaserScan(project=True,
                        H=self.sensor_img_H,
                        W=self.sensor_img_W,
                        fov_up=self.sensor_fov_up,
                        fov_down=self.sensor_fov_down)
    
    # Rotate point cloud if a roation is available
    if self.rot_fraction > 0 :
        #print("[Parser] rotation")
        rotation = self.rot_angles[index]
        #print("[INFO] Rot:%f"%(rotation))
        if rotation != 0:
          anchor_scan.set_rotation(True,rotation)
        #anchor_scan.open_scan(anchor_file)
   
    anchor_scan.open_scan(anchor_file)
    # Project point cloud to CNN input format
    projA =  torch.from_numpy(anchor_scan.points)

    # return
    return projA, projB, label

  def __len__(self):
    return len(self.anchor_files)


  def gen_projection(self,scan):
    unproj_n_points = scan.points.shape[0]
    #unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    unproj_xyz = torch.from_numpy(scan.points[::self.step])
    
    return(unproj_xyz)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               dataset,              # directory for data
               session ,        # max points in each scan in entire datase  # sequences to train
               debug = False
               ):  # shuffle training set?
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root      = dataset['path']['root']
    self.sensor    = dataset['sensor']
    self.max_points = dataset['max_points']

    self.sequences  = session['split']
    self.batch_size = session['batch_size']
    self.workers    = session['workers']
    self.shuffle    = session['shuffle']
    self.fraction   = session['fraction']
    self.rotation  = session['rotation']
    # Data loading code
    self.dataset = SemanticKitti(root = self.root,
                                  sensor  = self.sensor,
                                  sequences  = self.sequences,
                                  max_points = self.max_points,
                                      #gt = gt,
                                  fraction = self.fraction,
                                  debug = debug
                                )
    
    if self.rotation['fraction']  > 0 and self.rotation['angles']  is not None:
      self.dataset.set_rotation_set(self.rotation['angles'],self.rotation['fraction'])

    self.loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle= self.shuffle,
                                                  num_workers=self.workers,
                                                  pin_memory=True,
                                                  drop_last=False)


  def get_triplets(self):
    return(self.dataset.get_idx())

  def get_set(self):
    return self.loader

  def get_size(self):
    return len(self.loader)

  def get_param(self):
    return(self.dataset.get_param())
  
  def get_files(self):
    return(self.dataset.get_files())
