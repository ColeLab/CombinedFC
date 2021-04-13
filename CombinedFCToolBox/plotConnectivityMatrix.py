#plot a weighted connectivity matrix
#can use functional networks defined in Ji et al., (2019)
#https://www.sciencedirect.com/science/article/pii/S1053811918319657
#for the 360 cortical regions of Glasser et al., (2016)
#https://www.nature.com/articles/nature18933

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.colors as colors
import pkg_resources

def plotConnectivityMatrix(ConnMat,
                           methodTitle='correlation',
                           functionalNetworks = False,
                           networkColorBar_x = False,
                           networkColorBar_y = False,
                           networkLabels = False,
                           orientations = 'bottom'):  
    '''
    INPUT: 
        ConnMat : weighted connectivity matrix
        methodTitle : method used to compute the matrix, will be used in the title
        functionalNetworks : True if want to use the network order of Ji et al. (2019)
        networkColorBar_x : if True, display color bars indicating the networks in the x axis
        networkColorBar_y : if True, display color bars indicating the newtworks in the y axis
        networkLabels : if True, display networks labels in the Y axis
    OUTPUT:
        fig : a fig object containting a weighted connectivity matrix plot,
              with positive weights red and negative weights blue.
        ax : an axis objects for the fig, can use to manipulate the output graph if necessary
    *Usage: to visualize the plot, run the function with the proper input and save the output,
            then plot the output:
            fig,ax = plotConnectivityMatrix(...) 
            fig.show()
    '''
    
    if functionalNetworks == False:
        netOrder = np.arange(ConnMat.shape[0])
    
    #To apply the functional network order from Ji et al., (2019)
    elif functionalNetworks == True:
        #Glasser 360 cortex parcellation ordered into functional networks reported in Ji et al., (2019)
        #make as integer and subtract 1, so the indices start in 0, as Python requires.
        #path where the network file is: it contains labels (first column) and order (second column)
        netFilePath = pkg_resources.resource_filename('CombinedFC.CombinedFCToolBox','aux_files/networks_labels.txt')
        netFile = np.loadtxt(netFilePath,delimiter=',')
        #to assign each of the 360 nodes to its corresponding network
        netOrder = netFile[:,1].astype(int) - 1 

    
    #for the weights colorbar: red positive, blue negative
    v_min = np.min(ConnMat)
    v_max = np.max(ConnMat)
    v_mid = 0
    #define the figure and the axes
    fig,ax = plt.subplots()
    #plot the data
    img = ax.imshow(ConnMat[netOrder,:][:,netOrder],
                    origin = 'lower',
                    cmap='seismic',
                    clim=(v_min, v_max), 
                    norm=MidpointNormalize(midpoint=v_mid,vmin=v_min, vmax=v_max))
    plt.title(methodTitle,fontsize=16)
    #axes ticks labels
    a = np.round(ConnMat.shape[0]/2).astype(int)
    b = np.round(ConnMat.shape[0])
    plt.xticks([1,a,b])
    plt.yticks([1,a,b])
    #labels font size and length of the ticks
    #ax.tick_params(labelsize=14,length=0.01)
    #Thickness of the connectivity matrix border
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.1)
    #properties of the weights colorbar
    cbar = plt.colorbar(img)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_linewidth(0.01)
    #cbar.set_ticks([])
    
    
    if functionalNetworks == True:
        #To create the networks colorbars
        #network palette defined by the Cole lab:
        networkPalette = ['royalblue','slateblue','paleturquoise','darkorchid','limegreen',
                          'lightseagreen','yellow','orchid','r','peru','orange','olivedrab']
        #in case the networks labels want to be used in the plot
        orderedNetworks = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMM','VMM','ORA']
        #define the colormap as an independent graphical object
        cmap = mpl.colors.ListedColormap(networkPalette)
        #number of nodes (size) in each of the 12 networks
        size_networks=[]
        netLabels = netFile[:,0].astype(int)
        #loop through all the labels: 1 to 12 to count the number of nodes
        for i in range(np.max(netLabels)):
            size_networks.append(np.sum(netLabels==i+1))
        #the bounds of the bar are the cumulative sum for each network size starting at zero: 
        #ie, 0, 6, 6+54, 6+54+39, etc...
        su = 0
        #the first element of the networks bar bounds is zero
        bounds = [0]
        #this loop makes the cumulative sums
        for i in range(np.max(netLabels)):
            su += size_networks[i]
            bounds.append(su)
        #define the size of the color blocks according to the bounds (ie. the number of nodes)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        #get the size and position of the connectivity matrix as a reference 
        #to position the networks color bars
        l, b, w, h = ax.get_position().bounds
    
    if networkColorBar_x == True and functionalNetworks == True:
        ax1 = fig.add_axes([l,b/1.4,w,h/25])
        cbNet = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                        norm=norm,
                                        spacing='proportional',
                                        orientation='horizontal')
        #no border in the colorbar
        cbNet.outline.set_linewidth(0)
        #no ticks in the colorbar
        cbNet.set_ticks([])
    
    
    if networkColorBar_y == True and functionalNetworks == True:
        ax2 = fig.add_axes([l/1.105,b,w/25,h])

        cbNet = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                        norm=norm,
                                        spacing='proportional',
                                        orientation='vertical')
        cbNet.outline.set_linewidth(0)
        cbNet.set_ticks([])
    
    
    #make extra adjustment to the position of the matrix labels if functional networks colorbars are used
    if networkColorBar_x == True and functionalNetworks == True:
        ax.tick_params(axis='x',labelsize=14,pad=18,length=0.01)
    elif networkColorBar_x != True:
        ax.tick_params(axis='x',labelsize=14,length=0.01)

    if networkColorBar_y == True and functionalNetworks == True:
        ax.tick_params(axis='y',labelsize=14,pad=18,length=0.01)
    elif networkColorBar_y != True:
        ax.tick_params(axis='y',labelsize=14,length=0.01)
        
    if functionalNetworks == False:
        ax.tick_params(axis='y',labelsize=14,length=0.01)
        ax.tick_params(axis='x',labelsize=14,length=0.01)
        
    if networkLabels == True and functionalNetworks == True:
        #in case the networks labels are used in the plot
        orderedNetworks = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMM','VMM','ORA']
        #remove the number labels
        ax.get_yaxis().set_ticks([])
        #define styles for the text
        style = dict(va='top',transform=fig.transFigure,size=11)
        #position on the x axis of the figure
        axX = 0.15
        #include the network labels in the corresponding y-axis position of the figure
        plt.text(axX, 0.14, orderedNetworks[0],**style)
        plt.text(axX, 0.22, orderedNetworks[1],**style)
        plt.text(axX, 0.31, orderedNetworks[2],**style)
        plt.text(axX, 0.42, orderedNetworks[3],**style)
        plt.text(axX, 0.49, orderedNetworks[4],**style)
        plt.text(axX, 0.54, orderedNetworks[5],**style)
        plt.text(axX, 0.62, orderedNetworks[6],**style)
        plt.text(axX, 0.68, orderedNetworks[7],**style)
        plt.text(axX, 0.78, orderedNetworks[8],**style)
        plt.text(axX, 0.84, orderedNetworks[9],**style)
        plt.text(axX, 0.88, orderedNetworks[10],**style)
        plt.text(axX, 0.92, orderedNetworks[11],**style)
        
    #in case the origin of the matrix is in the upper left corner, invert the color bar order
    #todo: get the correct orientation for when both a
    #if img.origin == 'upper':
    #    ax1.invert_xaxis()
    #    ax2.invert_xaxis()
    
    return fig,ax


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
