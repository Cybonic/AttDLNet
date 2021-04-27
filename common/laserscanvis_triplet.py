#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy import gloo
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
from common.laserscan import LaserScan, SemLaserScan
import vispy.plot as vp


class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, scan, scan_names, poses, offset=0,
               semantics=True, instances=False,W=1024,H = 64,bgcolor='black',cmap= 'viridis'):
    self.scan = scan
    self.scan_names = scan_names
    
    # self.label_names = label_names
    self.offset = offset
    self.semantics = semantics
    self.instances = instances
    self.bgcolor = bgcolor
    self.cmap = cmap
    # sanity check
    if not self.semantics and self.instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.height = H
    self.width = W

    self.rotation = 1
    self.reset()
    

    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True,bgcolor=self.bgcolor)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()


    #self.Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)
    # new canvas prepared for visualizing data
    self.canvas_pose = SceneCanvas(keys='interactive',show=True,bgcolor=self.bgcolor)
    # interface (n next, b back, q quit, very simple)
    self.canvas_pose.events.key_press.connect(self.key_press)
    self.canvas_pose.events.draw.connect(self.draw)
    # grid
    #self.view_pose = self.canvas_pose.central_widget.add_view()
    # self.view_pose.camera = "turntable"
    # self.view_pose.camera.fov = 45
    # self.view_pose.camera.distance = 500

    #self.vis_pose = self.Scatter3D(parent=self.view_pose.scene)
    #self.view_pose.camera.aspect = 1

    # self.fig = vp.Fig(show=True,bgcolor=self.bgcolor)
    # self.fig.events.key_press.connect(self.key_press)
    # self.fig.events.draw.connect(self.draw)

    # grid = vp.visuals.GridLines(color=(0, 0, 0, 0.5))
    # grid.set_gl_state('translucent')
    #self.fig[0, 0].view.add(grid)

    # new canvas for img
    # img canvas size
    self.multiplier = 3
    self.canvas_W = self.width
    self.canvas_H = self.height

    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  size=(self.canvas_W, self.canvas_H * self.multiplier),bgcolor=self.bgcolor)
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()

    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # laserscan part
    self.canvas_vis   = {}
    self.canvas_view  = {}
    self.img_view     = {}
    self.img_vis      = {}

    position = [[0,0],[0,1],[0,2]]
    instances = ['anch','pos','neg']
   
    for l,inst in zip(position,instances):

      # point cloud
      view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(view, l[0], l[1])
      vis = visuals.Markers()
      view.camera = 'turntable'

      view.add(vis)
      visuals.XYZAxis(parent=view.scene)

      self.canvas_vis[inst]  = vis
      self.canvas_view[inst] = view

      # Depth (img)
      img_view = vispy.scene.widgets.ViewBox(
                border_color='white', parent=self.img_canvas.scene)

      self.img_grid.add_widget(
                        img_view, 
                        row = l[0], 
                        col = l[1])

      img_vis = visuals.Image(cmap=self.cmap)
      img_view.add(img_vis)

      self.img_view[inst] = img_view
      self.img_vis[inst]  = img_vis
    
    Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)
    
    
    
    self.pose_view = self.canvas_pose.central_widget.add_view()
    self.pose_view.camera = 'turntable'
    self.pose_vis = Scatter3D(parent=self.pose_view.scene)
    #self.pose_view = vispy.scene.widgets.ViewBox(
    #            border_color='white', parent=self.canvas_pose.scene)
    #self.pose_view = vispy.scene.widgets.ViewBox(
    #            border_color='white', parent=self.canvas_pose.scene)

    #self.pose_vis = vis = visuals.Markers()
    #self.pose_view.add(self.pose_vis)
    #vispy.visuals.MarkersVisual(parent=self.pose_view.scene)

    # Pose
    #Scatter3D = vispy.scene.visuals.create_visual_node(vispy.visuals.MarkersVisual)
    #self.vis_pose = Scatter3D(parent=self.view_pose.scene)
    #self.vis_pose.set_gl_state('translucent', blend=True, depth_test=True)

    

    #self.view_pose.add(self.vis_pose)

    #visuals.XYZAxis(parent=self.view_pose.scene)
    #self.view_pose.camera = 'turntable'



  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_scan(self):
    idx = self.offset
    self.update('anch',idx)
    self.update('pos',idx)
    self.update('neg',idx)
    self.update('poses',idx)

  def update(self,instance,idx):
    # first open data

    if instance != 'poses':
      file_idx = self.scan_names[instance][idx]
      file = self.scan_names['scans'][file_idx]
      
      if self.rotation and instance == 'anch':
        self.scan.set_rotation(1,180)
        self.scan.open_scan(file)
        self.scan.set_rotation(0,150)
      else: 
        self.scan.open_scan(file)
      # then change names
      title = "scan " + str(self.offset) + " of " + str(len(self.scan_names))
      self.canvas.title = title
      self.img_canvas.title = title

      # then do all the point cloud stuff

      # plot scan
      power = 16
      # print()
      range_data = np.copy(self.scan.unproj_range)
      # print(range_data.max(), range_data.min())
      range_data = range_data**(1 / power)
      # print(range_data.max(), range_data.min())
      viridis_range = ((range_data - range_data.min()) /
                      (range_data.max() - range_data.min()) *
                      255).astype(np.uint8)
                      
      viridis_map = self.get_mpl_colormap(self.cmap)
      viridis_colors = viridis_map[viridis_range]

      self.canvas_vis[instance].set_data(self.scan.points,
                            face_color=viridis_colors[..., ::-1],
                            edge_color=viridis_colors[..., ::-1],
                            size=1)


      # now do all the range image stuff
      # plot range image
      data = np.copy(self.scan.proj_range)
      # print(data[data > 0].max(), data[data > 0].min())
      data[data > 0] = data[data > 0]**(1 / power)
      data[data < 0] = data[data > 0].min()
      # print(data.max(), data.min())
      data = (data - data[data > 0].min()) / \
          (data.max() - data[data > 0].min())
      # print(data.max(), data.min())
      self.img_vis[instance].set_data(data)
      self.img_vis[instance].update()

    else: 

      
      anch_idx = self.scan_names['anch'][idx]
      pos_idx = self.scan_names['pos']
      pos = pos_idx[idx]
      neg_idx = self.scan_names['neg'][idx]
      points = self.scan_names['poses']

      print("[Neg] %i"%(neg_idx))
      print("[Pos] %i"%(pos))
      print("[anc] %i"%(anch_idx))

      viridis_map = self.get_mpl_colormap(self.cmap)
      color_frame = 0.5*np.ones((len(points),3))
      color_frame[anch_idx] = [0,0,1]
      color_frame[pos_idx] = [0,1,0]
      color_frame[neg_idx] = [1,0,0]

      mean_p = np.mean(points,axis=0)
      points = points - points[anch_idx]
      
      #viridis_colors = viridis_map[viridis_range]

      size_  = 5*np.ones(len(points))
      size_[anch_idx]= 20
      size_[pos]= 15
      size_[neg_idx] = 20
      
      # line = self.fig[0,0].plot(points,
      #          face_color = color_frame,
      #          edge_color = color_frame,
      #          width=1, symbol='o')
      
      # line.set_gl_state(depth_test=False)
      
      

      self.pose_vis.set_data(points,
                            face_color = color_frame,
                            edge_color = color_frame,
                            size=size_,
                            symbol='o')
      #self.fig.show(run=False)


  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      self.update_scan()
    elif event.key == 'B':
      self.offset -= 1
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()

  def run(self):
    vispy.app.run()
