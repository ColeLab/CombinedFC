#Functions to generate simulated data from a given directed network model.
#Properly, simulate a dataset X, using the linear model X = WX + E,
#where W is a matrix of coefficients and E are noise terms.
#The W matrix encode the directed network model.

import numpy as np
from scipy import stats, linalg
import h5py


def pseudoEmpiricalData(C, mean_coeff = 0.8, std_coeff = 1, nDatapoints=1200):
    ''' 
    INPUT
        C : directed binary connectivity matrix. If entry Cij = 1 then Wij will be assigned a coefficient value.
        mean_coeff : mean for the normal distribution from where W coefficients will be sampled.
        std_coeff :  standard deviation for the normal distribution from where W coefficients will be sampled.
        nDatapoints : number of datapoints for dataset X. Default 1200 as one session of the HCP resting-state data
    OUTPUT
        X : pseudoEmpirical dataset with shape [nDatapoints x nNodes].
    '''

    #number of directed egdes in the network C
    nEdges = np.sum(C==1)
    W = C.copy()
    #coefficients sampled from a normal distribution with mean = mean_coeff and std = std_coeff
    coeff = np.random.normal(loc=mean_coeff, scale=std_coeff, size=(1,nEdges))
    #threshold to avoid edges with coefficient zero
    thresh = 0.5
    coeff[(coeff < thresh) & (coeff >= 0)] = thresh
    coeff[(coeff > -thresh) & (coeff < 0)] = -thresh
    
    #create the coefficient matrix W
    W[W.nonzero()] = coeff
    #zeroed the diagonal of W to guarantee that in the inversion, the coefficients for the self error E_ii = 1
    nNodes = W.shape[0]
    np.fill_diagonal(W,0)
    
    #randomize empirical data and subjects to create E values for the model X = WX + E
    datadir = 'CombinedFCToolBox/hcp_data'
    #150 subjects used to guarantee variability in the simulations
    subjects = [
        '100206','108020','117930','126325','133928','143224','153934','164636','174437',
        '183034','194443','204521','212823','268749','322224','385450','463040','529953',
        '587664','656253','731140','814548','877269','978578','100408','108222','118124',
        '126426','134021','144832','154229','164939','175338','185139','194645','204622',
        '213017','268850','329844','389357','467351','530635','588565','657659','737960',
        '816653','878877','987074','101006','110007','118225','127933','134324','146331',
        '154532','165638','175742','185341','195445','205119','213421','274542','341834',
        '393247','479762','545345','597869','664757','742549','820745','887373','989987',
        '102311','111009','118831','128632','135528','146432','154936','167036','176441',
        '186141','196144','205725','213522','285345','342129','394956','480141','552241',
        '598568','671855','744553','826454','896879','990366','102513','112516','118932',
        '129028'
    ]
    #use the four resting-state available sessions from the HCP
    restRuns = [
        'rfMRI_REST1_RL', 
        'rfMRI_REST1_LR',
        'rfMRI_REST2_RL',
        'rfMRI_REST2_LR'
    ]
    
    #randomly pick a subject from the list
    subj = np.random.randint(0,len(subjects))
    #open the h5 file for the random subject
    h5file = h5py.File(f'{datadir}/{subjects[subj]}_data.h5','r')
    #initialize the dataset
    rest_data = []
    #concatenate the four resting-state datasets
    for run in restRuns:
        #extract each resting-state data, [:] loads it as an array
        tmp = h5file[f'{run}/nuisanceReg_resid_24pXaCompCorXVolterra'][:]
        rest_data.extend(tmp.T)
    #save the 360 regions(nodes) x 4780 datapoints dataset
    rest_data = np.asarray(rest_data).T
    h5file.close()
    
    #allocate memory
    E = np.zeros((nNodes,nDatapoints))
    for node in range(nNodes):
        #randomly choose a region
        rand_region = np.random.randint(0,rest_data.shape[0])
        #randomize the datapoints
        shuf_datapoints = np.arange(nDatapoints)
        np.random.shuffle(shuf_datapoints)
        #define the error E terms by standardizing the shuffled datapoints
        E[node,:] = stats.zscore(rest_data[rand_region,shuf_datapoints],axis=0)
    
    
    #in the linear model the dataset X is defined as X = WX + E, 
    #so can be expressed as X = inv(I-W)*E to generate X dataset, where I is the identity matrix
    I = np.identity(nNodes)  
    X = np.dot(linalg.pinv(I-W),E)
    #transpose to get pseudo empirical dataset X ordered as [nNodes x nDatapoints]
    X = X.T 
    
    return X



def syntheticData(C, mean_coeff = 0.8, std_coeff = 1, nDatapoints=1200, distribution='Gaussian'):
    ''' 
    INPUT
        C : directed binary connectivity matrix. If entry Cij = 1 then Wij will be assigned a coefficient value.
        mean_coeff : mean for the normal distribution from where W coefficients will be sampled.
        std_coeff :  standard deviation for the normal distribution from where W coefficients will be sampled.
        nDatapoints : number of datapoints for dataset X. Default 1200 as one session of the HCP resting-state data
        distribution : a string 'Gaussian' for a normal with mean = 0 and std.dev = 1, or 'nonGaussian',
                       for a beta with parameters a = 1, b = 5 
    OUTPUT
        X : pseudoEmpirical dataset with shape [nDatapoints x nNodes].
    '''

    #number of directed egdes in the network C
    nEdges = np.sum(C==1)
    W = C.copy()
    #coefficients sampled from a normal distribution with mean = mean_coeff and std = std_coeff
    coeff = np.random.normal(loc=mean_coeff, scale=std_coeff, size=(1,nEdges))
    #threshold to avoid edges with coefficient zero
    thresh = 0.5
    coeff[(coeff < thresh) & (coeff >= 0)] = thresh
    coeff[(coeff > -thresh) & (coeff < 0)] = -thresh
    
    #create the coefficient matrix W
    W[W.nonzero()] = coeff
    #zeroed the diagonal of W to guarantee that in the inversion, the coefficients for the self error E_ii = 1
    nNodes = W.shape[0]
    np.fill_diagonal(W,0)
    
    #generate independent E noise values for the model X = WX + E
    if distribution == 'Gaussian':
        E = np.random.normal(0,1,size=(nNodes,nDatapoints))
    if distribution == 'nonGaussian':
        E = np.random.beta(1,5,size=(nNodes,nDatapoints))
    
    
    #in the linear model the dataset X is defined as X = WX + E, 
    #so can be expressed as X = inv(I-W)*E to generate X dataset, where I is the identity matrix
    I = np.identity(nNodes)  
    X = np.dot(linalg.pinv(I-W),E)
    #transpose to get pseudo empirical dataset X ordered as [nNodes x nDatapoints]
    X = X.T 
    
    return X
 
 


