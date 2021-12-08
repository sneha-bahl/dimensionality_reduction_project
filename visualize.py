#!/usr/bin/env python
# coding: utf-8

# In[7]:


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')


# ## 2D visualization

# In[2]:


def visualize_2d(red_data, data_label, dataset, labels, img_name, xlbl, ylbl ):
    N= 10
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    cmap = plt.cm.jet
    color_list = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', color_list, cmap.N)
    bins = np.arange(N+1)
    norm = colors.BoundaryNorm(bins, cmap.N)
    # PLOT
    scat = ax.scatter(red_data[:,0], red_data[:,1],c= data_label,cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', pad = 0.1)
    c_labels = np.arange(N)
    c_loc = bins + .5
    cb.set_ticks(c_loc)
    cb.set_ticklabels(labels)
    cb.set_label('Classes')
    ax.set_title(dataset)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.savefig(img_name,bbox_inches='tight',dpi=400)
    plt.show()


# ## 3D visualization

# In[3]:


def visualize_3d(red_data, data_label, dataset, labels, xlbl, ylbl, zlbl):
    # setup the plot
    N= 10
    fig, ax = plt.subplots(1,1, figsize=(8, 7))
    ax = plt.axes(projection ="3d")
    
    cmap = plt.cm.jet
    color_list = [cmap(i) for i in range(cmap.N)]
    # new mapping
    cmap = cmap.from_list('Custom cmap', color_list, cmap.N)
    
    bins = np.arange(N+1)
    norm = colors.BoundaryNorm(bins, cmap.N)

    # PLOT
    scat = ax.scatter(red_data[:,0], red_data[:,1], red_data[:,2],c= data_label,cmap=cmap, norm=norm)
    
    cb = plt.colorbar(scat, spacing='proportional', pad = 0.2)
    c_labels = np.arange(N)
    c_loc = bins + .5
    cb.set_ticks(c_loc)
    cb.set_ticklabels(labels)
    cb.set_label('Classes')
    ax.set_title(dataset)
    ax.set(xlabel=xlbl, ylabel=ylbl, zlabel=zlbl)
    plt.show()

