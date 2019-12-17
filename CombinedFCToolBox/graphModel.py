#first install the next module, typing from terminal:
# "conda install -c conda-forge python-igraph"
# or "pip install python-igraph"
#documentation at, https://igraph.org/python/

import igraph as ig
import numpy as np


def graphModel(model, edgeDensity, nNodes):
    '''
    INPUT:
        model : graphical model to simulate the connectivity network, a string, 'StaticPowerLaw' or 'ErdosRenyi'
        edgeDensity : Parameter to control the connectivity density (#edges / total possible edges) of the graph
        nNodes : number of nodes in the network
    OUTPUT:
        C : a binary (1/0) directed connectivity network encoded as, column --> row direction Cij = 1 : j --> i
    '''

    if model == 'ErdosRenyi':
        """
        From igraph documentation:
        Erdos_Renyi(n, p, m, directed=False, loops=False)
        Generates a graph based on the Erdos-Renyi model.
        Parameters:
        n - the number of vertices.
        p - the probability of edges. If given, m must be missing. 
        m - the number of edges. If given, p must be missing. #Use this parameter in this application
        directed - whether to generate a directed graph.
        loops - whether self-loops are allowed.
        """ 
        g = ig.Graph.Erdos_Renyi(nNodes,m = edgeDensity, directed=True, loops=False)
    
    
    elif model == 'StaticPowerLaw':
        """
        From igraph documentation:
        Static_Power_Law(n, m, exponent_out, exponent_in, loops=False, multiple=False, finite_size_correction=True)
        Generates a non-growing graph with prescribed power-law degree distributions.
        Parameters:
        n - the number of vertices in the graph
        m - the number of edges in the graph
        exponent_out - the exponent of the out-degree distribution, which must be between 2 and infinity (inclusive). When exponent_in is not given or negative, the graph will be undirected and this parameter specifies the degree distribution. exponent is an alias to this keyword argument.
        exponent_in - the exponent of the in-degree distribution, which must be between 2 and infinity (inclusive) It can also be negative, in which case an undirected graph will be generated.
        loops - whether loop edges are allowed.
        multiple - whether multiple edges are allowed.
        finite_size_correction - whether to apply a finite-size correction to the generated fitness values for exponents less than 3. See the paper of Cho et al for more details.
        """
        g = ig.Graph.Static_Power_Law(nNodes, m = edgeDensity, exponent_out = 2, 
                                      exponent_in = 4, loops=False, multiple=False, 
                                      finite_size_correction=True)
    

    #transform the graph into a binary directed connectivity matrix
    C = g.get_adjacency() 
    #transform from the igraph matrix type to array type for numpy use
    C = np.asarray(C._get_data())  
    #transpose to get column --> row direction 
    C = C.T 
    
    #modify the networks to guarantee a majority of confounder and chains (ErdosRenyi) 
    #or colliders (StaticPowerLaw)
    if model == 'ErdosRenyi':
        C = np.tril(C,1)
            
    if model == 'StaticPowerLaw':
        C = C.T
        C = np.tril(C,1)
    
    return C