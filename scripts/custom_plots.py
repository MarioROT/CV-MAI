from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple
import random
from utils import get_ORB
from skimage import feature, segmentation
from skimage.color import rgb2gray, rgba2rgb
import numpy as np
import warnings

class custom_grids():
  """"
  Function to print images in a personalized grid of images according to a layout provided
  or a simple arrangment auto calculated according to the number of columns and rows provided
  it is also possible to add layers of effects to some images as squares, lines, etc.
  """
  def __init__(self,
             imgs: List,
             rows: int = 1,
             cols: int = 1,
             titles: List = None,
             order: List = None,
             figsize: Tuple = (10,10),
             axis: str = None,
             cmap: str = None,
             title_size: int = 12,
             use_grid_spec: bool = True
             ):
      self.imgs = imgs
      self.rows = rows
      self.cols = cols
      self.titles = titles
      self.order = order
      self.figsize = figsize
      self.axis = axis
      self.cmap = cmap
      self.title_size = title_size
      self.use_grid_spec = use_grid_spec
      self.len_imgs = 0
      self.fig = None
      self.axs = None

      if not self.order:
        self.order = [[i, [j, j + 1]] for i in range(self.rows) for j in range(self.cols)]

  def __len__(self):
    return len(self.imgs)

  def show(self): 
    if not self.use_grid_spec:
      self.fig, self.axs = plt.subplots(self.rows, self.cols, figsize=self.figsize)
      if self.rows <= 1 and self.cols <= 1:
        for idx, img in enumerate(self.imgs):
          self.axs.imshow(img, cmap=self.cmap)
          if self.axis:
            self.axs.axis(self.axis)
          if self.titles:
            self.axs.set_title(self.titles[idx], fontsize=self.title_size)
          self.len_imgs += 1
      elif self.rows <= 1 or self.cols <= 1:
        for idx, img in enumerate(self.imgs):
          self.axs[idx].imshow(img, cmap=self.cmap)
          if self.axis:
            self.axs[idx].axis(self.axis)
          if self.titles:
            self.axs[idx].set_title(self.titles[idx], fontsize= self.title_size)
          self.len_imgs += 1
      else:
        for idx, img in enumerate(self.imgs):
          row = round(np.floor(self.len_imgs/self.cols))
          column = self.len_imgs%self.cols
          self.axs[row][column].imshow(img, cmap=self.cmap)
          if self.axis:
            self.axs[row][column].axis(self.axis)
          if self.titles:
            self.axs[row][column].set_title(self.titles[idx], fontsize= self.title_size)
          self.len_imgs += 1
    else:
      self.fig = plt.figure(constrained_layout=True, figsize=self.figsize)
      gs = GridSpec(self.rows, self.cols, figure=self.fig)
      for n, (i, j) in enumerate(zip(self.imgs, self.order)):
        im = self.fig.add_subplot(gs[j[0], j[1][0]:j[1][1]])
        if self.cmap:
          im.imshow(i, cmap=self.cmap)
        else:
          im.imshow(i)
        if self.axis:
          im.axis('off')
        if self.titles:
          im.set_title(self.titles[n], fontsize= self.title_size)

  def add_plot(self, title=None, axis=None, position=None, last=False):
    if self.use_grid_spec:
      warnings.warn("To add graphics you need to set 'use_grid_spec' to false when instantiating the class.")
      return 0
    if self.len_imgs >= (self.rows*self.cols):
      warnings.warn("There is no space available to add a plot. Adjust the number of rows and columns when instantiating the class.")
      return 0
    if not position:
      nextr = round(np.floor(self.len_imgs/self.cols))
      nextc = self.len_imgs%self.cols
      position = [nextr, nextc]

    self.len_imgs += 1

    if last and (self.len_imgs < (self.rows*self.cols)):
      for e in range((self.rows*self.cols)-self.len_imgs):
        nextr = round(np.floor(self.len_imgs/self.cols))
        nextc = self.len_imgs%self.cols
        self.axs[nextr][nextc].axis("off")
        self.len_imgs += 1

    if self.rows <= 1:
      if self.cols <= 1:
        if axis:
          self.axs.axis(axis)
        if title:
          self.axs.set_title(title, fontsize= self.title_size)
        return self.axs
      else:
        if axis:
          self.axs[position[1]].axis(axis)
        if title:
          self.axs[position[1]].set_title(title, fontsize= self.title_size)
        return self.axs[position[1]]
    else:
      if axis:
        self.axs[position[0]][position[1]].axis(axis)
      if title:
        self.axs[position[0]][position[1]].set_title(title, fontsize= self.title_size)
      return self.axs[position[0]][position[1]]
  
  def overlay_image(self, img_idx, overlays, cmp_colors = None, alphas = None):
    if not cmp_colors:
      cmp_colors = [random.choice(plt.colormaps()) for i in range(len(img_idx))]
    elif len(alphas) < len(img_idx):
      cmp_colors = [cmp_colors[i%len(cmp_colors)] for i in range(len(img_idx))]
    if not alphas:
      alphas = [0.5 for i in range(len(img_idx))]
    elif len(alphas) < len(img_idx):
      alphas = [alphas[i%len(alphas)] for i in range(len(img_idx))]
    if len(overlays) < len(img_idx):
      overlays = [overlays[i%len(overlays)] for i in range(len(img_idx))]
    for o_idx, i_idx in enumerate(img_idx):
      if self.use_grid_spec:
        self.fig.axes[i_idx].imshow(overlays[o_idx], cmap=cmp_colors[o_idx], alpha=alphas[o_idx])
      elif self.cols == 1 and self.rows == 1:
        self.axs.imshow(overlays[o_idx], cmap=cmp_colors[o_idx], alpha=alphas[o_idx])
      elif self.cols == 1 or self.rows == 1:
        self.axs[i_idx].imshow(overlays[o_idx], cmap=cmp_colors[o_idx], alpha=alphas[o_idx])
      else:
        nextr = round(np.floor(i_idx/self.cols))
        nextc = i_idx%self.cols
        self.axs[nextr][nextc].imshow(overlays[o_idx], cmap=cmp_colors[o_idx], alpha=alphas[o_idx])

  def add_rects(self, img_idx, rects, rect_clrs = None, linewidth=1, facecolor=False):
    if not rect_clrs:
      rect_clrs = [random.choice(list(mcolors.CSS4_COLORS.keys())) for i in range(len(rects[0]))]
    if facecolor:
      face_clrs = rect_clrs.copy()
    else:
      face_clrs  = ['none' for i in range(len(rects[0]))]
    for i_idx, img in enumerate(rects):
      for r_idx,rect in enumerate(img):
        rect = patches.Rectangle(rect[0], rect[1], rect[2], linewidth=linewidth, edgecolor=rect_clrs[r_idx],
                               facecolor=face_clrs[r_idx])
        self.fig.axes[img_idx[i_idx]].add_patch(rect)

  def match_points(self, img_idx, matches_idxs, autoTitles = None):
    kps1_l, kps2_l, mts_l = [], [], []
    img = self.grayChecker(self.imgs[img_idx])
    match_imgs = [self.grayChecker(self.imgs[idx]) for idx in matches_idxs]

    for match in match_imgs:
      kp1, kp2, matches = get_ORB(img, match)
      kps1_l.append(kp1)
      kps2_l.append(kp2)
      mts_l.append(matches)

    self.fig = plt.figure(constrained_layout=True, figsize=self.figsize)
    gs = GridSpec(self.rows, self.cols, figure=self.fig)
    for n, (i, j) in enumerate(zip(match_imgs, self.order)):
      im = self.fig.add_subplot(gs[j[0], j[1][0]:j[1][1]])
      feature.plot_matches(im, img, i, kps1_l[n], keypoints2_l[n], mts_l[n])
      if self.axis:
        im.axis('off')
      if self.titles:
        im.set_title(self.titles[n], fontsize= self.title_size)
      if autoTitles:
        if autoTitles == True:
          im.set_title('Matches: ' + str(mts_l[n].shape[0]), fontsize= self.title_size)
        else:
          im.set_title(autoTitles[n] + ' - Matches: ' + str(mts_l[n].shape[0]), fontsize= self.title_size)

  @staticmethod
  def grayChecker(color_img):
    if len(color_img.shape) == 2:
      gray_img = color_img
    elif color_img.shape[2] == 3:
      gray_img = rgb2gray(color_img)
    elif color_img.shape[2] == 4:
      gray_img = rgb2gray(rgba2rgb(color_img))

    return gray_img
