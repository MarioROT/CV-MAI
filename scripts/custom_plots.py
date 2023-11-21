from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple
import random
from utils import get_ORB
from skimage import feature, segmentation
from skimage.color import rgb2gray, rgba2rgb
from pylab import *
import numpy as np
import pandas as pd
import warnings
import itertools


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

  def add_plot(self, title=None, axis=None, position=None, last=False, row_last=False, projection=False, clear_ticks=None, axlabels=None):
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

    if projection:
        self.axs[nextr][nextc].remove()
        self.axs[nextr][nextc] = self.fig.add_subplot(self.rows,self.cols,self.len_imgs,projection='3d')
        if clear_ticks:
            self.axs[nextr][nextc].zaxis.set_ticklabels([])
        if axlabels:
            self.axs[nextr][nextc].set_zlabel(axlabels[2])        
        
    if self.rows <= 1 and self.cols<=1:
      if clear_ticks:
        self.axs.xaxis.set_ticklabels([])
        self.axs.yaxis.set_ticklabels([])
      if axlabels:
        self.axs.set_xlabel(axlabels[0])
        self.axs.set_ylabel(axlabels[1])
    elif self.rows <= 1:
      if clear_ticks:
        self.axs[position[1]].xaxis.set_ticklabels([])
        self.axs[position[1]].yaxis.set_ticklabels([])
      if axlabels:
        self.axs[position[1]].set_xlabel(axlabels[0])
        self.axs[position[1]].set_ylabel(axlabels[1])
    elif self.cols <=1:
      if clear_ticks:
        self.axs[position[0]].xaxis.set_ticklabels([])
        self.axs[position[0]].yaxis.set_ticklabels([])
      if axlabels:
        self.axs[position[0]].set_xlabel(axlabels[0])
        self.axs[position[0]].set_ylabel(axlabels[1])     
    else:
      if clear_ticks:
        self.axs[nextr][nextc].xaxis.set_ticklabels([])
        self.axs[nextr][nextc].yaxis.set_ticklabels([])
      if axlabels:
        self.axs[nextr][nextc].set_xlabel(axlabels[0])
        self.axs[nextr][nextc].set_ylabel(axlabels[1])
    
    
  
    if row_last and (nextc < self.cols):
      for e in range(self.cols-(nextc+1)):
        nextr = round(np.floor(self.len_imgs/self.cols))
        nextc = self.len_imgs%self.cols
        self.axs[nextr][nextc].axis("off")
        self.len_imgs += 1

    if last and (self.len_imgs < (self.rows*self.cols)):
      for e in range((self.rows*self.cols)-self.len_imgs):
        nextr = round(np.floor(self.len_imgs/self.cols))
        nextc = self.len_imgs%self.cols
        self.axs[nextr][nextc].axis("off")
        self.len_imgs += 1


    if self.rows <= 1 and self.cols<=1:
      if axis:
        self.axs.axis(axis)
      if title:
        self.axs.set_title(title, fontsize= self.title_size)
      return self.axs
    elif self.rows <= 1:
      if axis:
        self.axs[position[1]].axis(axis)
      if title:
        self.axs[position[1]].set_title(title, fontsize= self.title_size)
      return self.axs[position[1]]
    elif self.cols <= 1:
      if axis:
        self.axs[position[0]].axis(axis)
      if title:
        self.axs[position[0]].set_title(title, fontsize= self.title_size)
      return self.axs[position[0]]
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
      feature.plot_matches(im, img, i, kps1_l[n], kps2_l[n], mts_l[n])
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


def groupedBarPlot(data, xticks, title,legend=True,axislabels = False,width=0.35,figsize=(25,10), barLabel=False, png = False, pdf = False, colors = None, fsizes = False, axisLim = False, xtick_rot=False, bLconfs = ['%.2f', 14]):
    """Width recomendado para 2 barras agrupadas es 0.35, para 3 y 4 es 0.2
       Para usar el barLabel, debe ser una lista de listas por cada tipo,
       aun que sea solo una barra por paso en el eje x deber ser una lista contenida dentro de otra
       Las opciones para fsizes son:
            'font' --> controla el tamaño de los textos por defecto
            'axes' --> tamaño de fuente del titulo y las etiquetas del eje x & y
            'xtick' --> tamaño de fuente de los puntos en el eje x
            'ytick' --> tamaño de fuente en los puntos del eje y
            'legend --> controla el tamaño de fuente de la leyenda
            'figure' --> controla el tamaño de fuente del titulo de la figura
       """
    if fsizes:
        for key,size in fsizes.items():
            if key == 'font':
                plt.rc(key, size=size)
            elif key == 'axes':
                plt.rc(key, titlesize=size)
                plt.rc(key, labelsize=size)
            elif key in ['xtick','ytick']:
                plt.rc(key, labelsize=size)
            elif key == 'legend':
                plt.rc(key, fontsize=size)
            elif key == 'figure':
                plt.rc(key, titlesize=size)
    else:
        plt.rc('font', size=15)

    x = np.arange(len(xticks))
    if colors:
        cl = colors
    else:
        cl = clrs

    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    rects = {}
    if len(data) == 1:
        ldata = list(data.values())
        keys = list(data.keys())
        rects[keys[0]] = ax.bar(x, ldata[0], width, label=keys[0], color = cl)
    elif len(data) == 2:
        ldata = list(data.values())
        keys = list(data.keys())
        rects[keys[0]] = ax.bar(x + width/2, ldata[0], width, label=keys[0], color = cl[2])
        rects[keys[1]] = ax.bar(x - width/2, ldata[1], width, label=keys[1], color = cl[3])
    elif len(data) == 3:
        ldata = list(data.values())
        keys = list(data.keys())
        rects[keys[0]] = ax.bar(x, ldata[0], width, label=keys[0])
        rects[keys[1]] = ax.bar([i+width for i in x], ldata[1], width, label=keys[1])
        rects[keys[2]] = ax.bar([i+2*width for i in x], ldata[2], width, label=keys[2])
    elif len(data) == 4:
        ldata = list(data.values())
        keys = list(data.keys())
        rects[keys[0]] = ax.bar(x + width/2, ldata[0], width, label=keys[0], color = cl[0])
        rects[keys[1]] = ax.bar(x - width/2, ldata[1], width, label=keys[1], color = cl[1])
        rects[keys[2]] = ax.bar(x + 1.5*width, ldata[2], width, label=keys[2], color = cl[2])
        rects[keys[3]] = ax.bar(x - 1.5*width, ldata[3], width, label=keys[3], color = cl[3])

    # ax.patch.set_facecolor('red')
    ax.patch.set_alpha(0.0)

    if axislabels:
        ax.set_xlabel(axislabels[0])
        ax.set_ylabel(axislabels[1])

    ax.set_title(title)
    if len(data) == 3:
        ax.set_xticks(x+width)
    else:
        ax.set_xticks(x)
    if xtick_rot:
        ax.set_xticklabels(xticks, rotation = xtick_rot)
    else:
        ax.set_xticklabels(xticks)

    if legend:
        ax.legend(prop={"size":30})

    if barLabel:
#         error = ['Hola' for i in range(9)]
#         ax.bar_label(list(rects.values())[0], padding=3, labels=[ e for e in error])
        try:
            for j,i in enumerate(rects.values()):
                ax.bar_label(i, padding=3, labels=[barLabel[0][:].format(ldata[j][r], barLabel[j+1][r]) for r in range(len(ldata[0]))])
        except:
            for j,i in enumerate(rects.values()):
                ax.bar_label(i, padding=3, labels=['{}\n{:.2f}%'.format(ldata[j][r], barLabel[j][r]) for r in range(len(ldata[0]))])
    else:
        for i in rects.values():
            ax.bar_label(i, padding=3, fmt = bLconfs[0], fontsize = bLconfs[1])

    fig.tight_layout()

    if axisLim:
        for key,values in axisLim.items():
            if key == 'xlim':
                plt.xlim(values[0], values[1])
            elif key == 'ylim':
                plt.ylim(values[0], values[1])

    if png:
        plt.savefig(png + '.png', transparent=True)
    if pdf:
        plt.savefig(pdf + '.pdf', transparent=True)

    plt.show()
