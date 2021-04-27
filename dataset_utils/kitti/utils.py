
import os
#import kitti_utils
import numpy as np
import scipy.spatial as spatial
import yaml

ROI = range(1370,1400)
ROI = np.append(ROI,range(1500,1650))
ROI = np.append(ROI,range(3250,3900))
ROI = np.append(ROI,range(4400,4451))

class dataset():
    def __init__(self,root,seq = ''):
        
        if seq == '':
            self._dataset_root = root 
        else: 
            self._root = root 
            self._seq = seq
            self._dataset_root = os.path.join(root,seq)

        self._poses_path = os.path.join(self._dataset_root,'poses.txt')
        self._img_path = os.path.join(self._dataset_root,'image_2')
        self._velo_path = os.path.join(self._dataset_root,'velodyne')
        self._poses = poses(self._poses_path)
        # self.sync_file = 'gps_velo_img_sync.txt'
        #poses_path = super().get_poses_path()
        #super().__init__(poses_path)

    def get_sync_file(self):
        return(self.sync_file)

    def get_img_files(self):

        path = self._img_path
        if not os.path.isdir(path):
            print("[ERR] Path does not exist!")
        files = [f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        return(np.array(files))

    def get_velo_files(self):

        path = self._velo_path
        if not os.path.isdir(path):
            print("[ERR] Path does not exist!")
        files = [f.split('.')[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        return(files)

    def get_2D_poses(self):
        return(self._poses.get_2D_poses())

    def get_3D_poses(self):
        return(self._poses.get_3D_poses())

    def get_image_path(self):
        return(self._img_path)
    
    def get_velo_timestamps(self):

        timestamp_file = [int(v) for v in self.get_velo_files()]
        return(np.array(timestamp_file))

    def get_img_timestamps(self):
        
        timestamp_file = [int(v) for v in self.get_img_files()]
        return(np.array(timestamp_file))

    def get_timestamp(self):

        
        img_timestamps = self.get_img_timestamps()

        velo_timestamps = self.get_velo_timestamps()

        poses_timestamps = np.array(self._poses.get_timestamps())

        return({'poses':poses_timestamps,'img':img_timestamps,'velo':velo_timestamps})

    def get_root_path(self):
        return(self._dataset_root)

    def build_img_file_path(self,name):
        return(os.path.join(self.get_image_path(), name+'.tiff'))

    def build_velo_file_path(self,name):
        return(os.path.join(self._velo_path, name + '.bin'))

    def get_sync_data(self):

        syn_data_idx= dict()
        syn_file = os.path.join(self._dataset_root, 'gps_img_sync.timestamps')
        for line in open(syn_file,'r'):
            linestruct = line.strip().split(':')
            modality = linestruct[0]
            idx = [int(v) for v in linestruct[1].split(' ')]
            syn_data_idx[modality] = idx

        poses = self.get_2D_poses()
        syn_poses = poses[syn_data_idx['poses'],:] 
        
        img_file = self.get_img_files()
        syn_img =  img_file[syn_data_idx['img']]

        velo_file = self.get_velo_files()
        syn_velo =  velo_file[syn_data_idx['velo']]

        return(syn_poses,syn_velo,syn_img)

    def imgread(self,image_file):
        if not os.path.isfile(image_file):
            print("[ERR] Image does not exist!")
        return(cv2.imread(image_file))

    def velo_read(self,scan_path):
        scan = []
        scan_file = open(scan_path,'rb')
        while True: 
            x_str = scan_file.read(2)
            if not x_str:
                break

            
            x = struct.unpack('<H', x_str)[0]
            y = struct.unpack('<H', scan_file.read(2))[0]
            z = struct.unpack('<H', scan_file.read(2))[0]
            i = struct.unpack('B', scan_file.read(1))[0]
            l = struct.unpack('B', scan_file.read(1))[0]
            
            x, y, z = convert(x, y, z)
            scan += [[x, -y, -z, i, l]]

        scan_file.close()

        #scan = scan.reshape((-1,4))
        return(np.array(scan))

def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

class poses():
    def __init__(self,pose_files):
        self.pose_files = pose_files
        self.p = self.load_poses_from_files()
        self.t = self.conv_to_pose(self.p)
        

    def load_poses_from_files(self):
        vector = []
        for line in open(self.pose_files):
            rawline = line.rstrip()
            data = rawline.split(' ')
            vector.append([float(value) for value in data])
        return(np.array(vector))

    def conv_to_pose(self,poses):

        idx = np.array([3,7,11])
        pose_vector = []
        for line in poses:
            pose = np.array(line)[idx]
            pose_vector.append(pose)
        return(np.array(pose_vector))
    
    def get_2D_poses(self):
        x = self.t.T[0,:]
        y = self.t.T[2,:]

        return(np.c_[x,y])
    
    def get_3D_poses(self):
        x = self.t.T[0,:]
        y = self.t.T[2,:]
        z = self.t.T[1,:]

        return(np.c_[x,y,z])


class retrieval():
    def __init__(self,root,session):
        # Path 
        self.root_dataset_path = root

        self.interm_root_path = os.path.join('data' ,'kitti')
        self.seq = session["REF"]
        ref_query = session["QER"]
        #self.model_path = os.path.join(self.dataset_path,model)
        self.loop_file = self.get_loop_file()
        loop_labels = read_loop_labels(self.loop_file)

        pose_file = os.path.join(root,self.seq, 'poses.txt')
        self.poses= poses(pose_file)
        self.query_idx,self.ref_idx = self.conv_to_retrieval(loop_labels)

    def conv_to_retrieval(self,loop_labels):
        ref = loop_labels['0']
        try:
            query = loop_labels['1']
        except:
            query= []
        return(query,ref)

    def get_loop_file(self):
        file_to_load = os.path.join(self.root_dataset_path,self.seq,'loop_labels.txt')
        if not os.path.isfile(file_to_load):
            print("[ERR] File does not exist!: " + file_to_load)
            return(-1)
        return(file_to_load)

    def get_sync_query_poses(self):
        return(self.poses.get_2D_poses()[self.query_idx])
    
    def get_sync_ref_poses(self):
        return(self.poses.get_2D_poses()[self.ref_idx])


def read_loop_labels(filename):
    print(os.getcwd())
    sequence_loop_labels = np.array([int(v) for line in open(filename) for v in line.rstrip().split(' ')])
    unq_labels = np.unique(sequence_loop_labels)
    indices = np.array(range(sequence_loop_labels.size))
    label_indices = {}
    for label in unq_labels:
        label_indices[str(label)] = indices[sequence_loop_labels == label]
    return(label_indices)