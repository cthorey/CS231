##################################################################
# Lien Utiles
    
#http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes

##################################################################
# Banque de fonction utiles pour les plots
    
def font(family,weight,size):
    # Fonction qui definit les propriete up-level de la figure
    import matplotlib
    
    font = {'family' : family,
            'weight' : weight,
            'size'   : size}
    matplotlib.rc('font', **font)
    

def legd(ax,place,size) :
     # given the handle, return a legend
    import matplotlib.pyplot as plt
    
    legend = ax.legend(loc=place, frameon=False,numpoints =1)
    for label in legend.get_texts():
        label.set_fontsize(size)
    for label in legend.get_lines():
        label.set_linewidth(2)  # the legend line width

def cbar_p(mesh,axs,label,tick):
    # given the mesh, the handle, the label and the tick, return a colorbar
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pylab as plt
    
    # create an axes on the right side of ax. The width of cax will be 3%
    # of ax and the padding between cax and ax will be fixed at 0.1 inch.
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(mesh,ax=axs,cax=cax)
    cbar.ax.set_title(label,fontsize=18)
    cbar.set_ticks(tick)
    cbar.set_ticklabels(tick)

def tick(ax,size,xticks,yticks,xlabel,ylabel) :
    ax.set_xlim(0,6)
    ax.set_ylim(0 ,2)
    ax.set_xlabel(xlabel,fontsize=24)
    ax.set_ylabel(ylabel,fontsize=24)
    ax.set_xticks(xticks)
    #ax.set_xticklabels(xticks)
    ax.set_yticks(yticks)
    #ax.set_yticklabels(yticks)  
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(size)
        
def make_colormap(seq):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def tick_1(ax,size) :
    #Change le tick des axes pour un handle donne
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(size)


def legd_Scatter(ax,place,size) :
     # given the handle, return a legend
    legend = ax.legend(loc=place, frameon=False)
    for label in legend.get_texts():
        label.set_fontsize(size)
    for label in legend.get_lines():
        label.set_linewidth(3)  # the legend line width

def legd_1_Point(ax,place,size) :
     # given the handle, return a legend
    legend = ax.legend(loc=place, frameon=False,numpoints = 1,handlelength =0)
    for label in legend.get_texts():
        label.set_fontsize(size)
    for label in legend.get_lines():
        label.set_linewidth(3)  # the legend line width
        
