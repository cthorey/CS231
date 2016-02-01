import sys,os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import Figure_Tool as FST
import matplotlib.gridspec as gridspec


class Plot(object):

    ''' The aim of this class is to help making plot  !!
    Class parameter:
    Path_Save = Path where save the plots
    attribut :
        ...
        - fig_b : IF True, save the plot in Load_G.Path_Save
        - fig_name : Name of the figure if the plot is saved

    method :
    - color_map(N): Make a list that can be used in
    iteration plot, N in the size of the iteration

    - axes(ax): make beautiful axes :)

    - figure() : creer une figure

    - save_fig() : routine to save fig. '''
    
    def __init__(self):

        self.path_save = ''
        self.colormap = plt.cm.hsv
        self.x_lim = [0,3]
        self.y_lim = [0,10]
        self.x_label = ''
        self.y_label = ''
        self.colormap = plt.cm.gist_ncar
        self.markersize = 200
        self.figsize = 20,12
        self.fig_b = False
        self.fig_name = 'fig1'
        self.font = 12
        self.font_label = 12
        self.font_legend = 9
        self.loc = 'upper left'
        self.alpha = 0.5


    def m_color_map(self,N):
        
        c=[self.colormap(i) for i in np.linspace(0,1,N)]
        c = ['b','r','g','c','m','y','k',(0.2,0.1,0.5)]
        return c

    def m_axes(self,ax):
        ax.set_xlabel(self.x_label,size=self.font_label)
        ax.set_ylabel(self.y_label,size=self.font_label)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        FST.tick_1(ax,self.font)
        FST.legd_Scatter(ax,self.loc,self.font_legend)
        
    def m_axes_Point(self,ax):
        ax.set_xlabel(self.x_label,size=self.font_label)
        ax.set_ylabel(self.y_label,size=self.font_label)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        FST.tick_1(ax,self.font)
        FST.legd_1_Point(ax,self.loc,self.font_legend)
        
    def m_axes_Ss_L(self,ax):
        ax.set_xlabel(self.x_label,size=self.font)
        ax.set_ylabel(self.y_label,size=self.font)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        FST.tick_1(ax,self.font)
        
    def m_figure(self):
        print self.figsize
        fig, ax = plt.subplots(figsize=self.figsize)
        return fig,ax

    def m_sub_figure(self,n,m):

        fig = plt.figure(figsize=self.figsize)
        gs=gridspec.GridSpec(n,m) # create a GridSpec object
        kx=dict()
        ax = np.array([[plt.subplot(gs[i,j],**kx)  for i in np.arange(n)] for j in np.arange(m)])
        return fig,ax

    def m_save_fig(self,fig):
        
        if self.fig_b:
            if self.path_save == '':
                print 'Preciser le chemin pour sauvegarder la figure'
                sys.exit()
            else:
                fig.savefig(self.path_save+''+self.fig_name+'.eps',rasterized=True, dpi=100,bbox_inches='tight', pad_inches=0.1)
            
            
