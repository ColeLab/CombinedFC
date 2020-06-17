#Compute and plot average (with standard deviation) of precision and recall across the repetitions of the simulation
#It uses the results returned from simulation.py 
#this function determines automatically which parameter is being analyzed:
#this is the parameter with more than 1 value in its array when running simulation.py

import numpy as np
import matplotlib.pyplot as plt

def plotAvgPrecisionRecall(ax1, 
                           ax2,
                           nEdges, 
                           nNodes, 
                           nDatapoints,
                           alphaSig, 
                           equiv_bounds, 
                           precision, 
                           recall, 
                           method,
                           linestyle,
                           marker):
    '''
    INPUT:
        ax1, ax2 : axes from the figure defined outside the function to be able to overlap the three methods
        nEdges : number of edges in the true simulated model
        nNodes : the array of values for the number of nodes used in the simulation
        nDatapoints : the array of values for the number of datapoints used in the simulation
        alphaSig : the array of values for the alpha values of significance used for the methods
        equiv_bounds : the array of values for the bounds in the equivalence test if used
        precision : the values of precision for all the repetitions for the simulation
        recall : the values of recall for all the repetitions for the simulation
        method : a string indicating the functional connectivity method used: 
                'bivariate correlation', 'partial correlation', 'combinedFC'
    OUTPUT:
        a precision plot and a recall plot, where the X axes show the parameter being analyzed,
        results are averages with standard deviation bars across all the repetitions.
    '''
   
    #axis = 0 contains the number of repetitions, the average is across repetitions
    #squeeze the matrices since we are making an analysis on one parameter at a time while keeping the others fixed
    avg_precision = np.nanmean(np.squeeze(precision),axis=0) 
    std_precision = np.nanstd(np.squeeze(precision),axis=0)
    
    avg_recall = np.nanmean(np.squeeze(recall),axis=0) 
    std_recall = np.nanstd(np.squeeze(recall),axis=0)
    
    #computes the edge density of the true model simulated
    #average number of edges of the true model across the repetitions
    avg_nEdges = np.nanmean(np.squeeze(nEdges),axis=0)  
    #edge density = % of edges out of the total possible edges
    avg_density = avg_nEdges/(nNodes[0]*(nNodes[0]-1)/2)  
    
    
    #this line determines which parameter is being analyzed: the one with the maximum dimension
    paramAnalyzed = np.argwhere(precision[0,:,:,:,:,:].shape == np.amax(precision[0,:,:,:,:,:].shape))[0]
    
    #for reference this is the order of the parameters in the results arrays:
    #(len(nNodes),len(parameters),len(nDatapoints),len(equiv_bounds),len(alphaSig))
    if paramAnalyzed == 0:
        xlabel = 'Number of Regions'
        xdata = np.asarray(nNodes)
    elif  paramAnalyzed == 1:
        xlabel = 'Connectivity Density (%)'
        xdata = avg_density*100
    elif paramAnalyzed == 2:
        xlabel = 'Number of Datapoints'
        xdata = np.asarray(nDatapoints)
    elif paramAnalyzed == 3:
        xlabel = 'Min. Correlation of Interest'
        xdata = np.asarray(equiv_bounds)
    elif paramAnalyzed == 4:
        xlabel = r'$\alpha$'+' Cutoff (log)'
        xdata = np.asarray(alphaSig)
        
    #make the plot for precision average results    
    ax1.errorbar(xdata,avg_precision,yerr=std_precision,label=method,linewidth=2.5,marker=marker,linestyle=linestyle)
    ax1.set_title('Precision',fontsize=17)
    ax1.grid(True)
    ax1.set_ylim([0,1.1])
    ax1.tick_params(grid_alpha=0.3,labelsize=14,length=0.01)
    ax1.set_xlabel(xlabel,fontsize=14)
    #for alpha values, since they can go from very small, eg. 0.0001 to a larger eg. 0.1
    if paramAnalyzed == 4:
        ax1.set_xscale('log')
        
        
    #remove the axis lines
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    #make the plot for recall average results
    ax2.errorbar(xdata,avg_recall,yerr=std_recall,label=method,linewidth=2.5,marker=marker,linestyle=linestyle)
    ax2.set_title('Recall',fontsize=17)
    ax2.grid(True)
    ax2.set_xlabel(xlabel,fontsize=14)
    ax2.tick_params(grid_alpha=0.3,labelsize=14,length=0.01)
    ax2.set_ylim([0,1.1])
    #for alpha values, since they can go from very small, eg. 0.0001 to a larger eg. 0.1
    if paramAnalyzed == 4:
        ax2.set_xscale('log')
    
    ax2.legend(bbox_to_anchor=(2, 1),fontsize=12, borderaxespad=0, frameon=False)
    #remove the axis lines
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    #move the y axis to the right
    ax2.yaxis.tick_right()
    