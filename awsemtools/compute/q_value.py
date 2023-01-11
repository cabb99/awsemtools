import numpy as np
import scipy.spatial.distance as sdist
import prody

#Core funtions

def q_value(trajectory, reference, selection = 'name CA', min_seq_sep=3, contact_threshold = 9.5, sigma_exp=0.15, interface=False, internal=True):
    """ Core function """
    trajectory = trajectory.copy()
    reference = prody.parsePDB(reference).select(selection)

    # Prepare calculation
    trajectory.set_selection(selection)
    rN=sdist.squareform(sdist.pdist(reference.getCoords()))
    chix = trajectory.reference.getChindices()
    ix = trajectory.reference.getResindices()
    sigma = np.power(np.abs(ix[:,np.newaxis]-ix[np.newaxis,:]),sigma_exp)
    sigma[chix[:,np.newaxis] != chix[np.newaxis,:]] = np.power(len(ix),sigma_exp)
    
    #Create calculation mask
    mask = 1
    mask *= (chix[:,np.newaxis] == chix[np.newaxis,:]) * internal + (chix[:,np.newaxis] != chix[np.newaxis,:]) * interface # Consider the interface or internal residues
    mask *= (ix[:,np.newaxis] - ix[np.newaxis,:] >= min_seq_sep) # Don't consider aminoacids that are too close in the sequence
    if contact_threshold: 
        mask *= (rN <= contact_threshold) # Only consider aminoacids that are in contact in the reference structure


    coordset=trajectory.coords
    qs=[]
    for coord in coordset:
        r=sdist.squareform(sdist.pdist(coord))
        with np.errstate(divide='ignore', invalid='ignore'):
            q=mask*np.exp(-(r-rN)**2/(2*sigma**2))
        q=np.where(mask,q,np.zeros(q.shape))
        #qs+=[q.sum()/mask.sum()]
        qs +=[q/mask.sum()]
    return np.array(qs)


# Q value flavors
def q_wolynes(trajectory, reference):
    """ Add formula """
    return q_value(trajectory, reference).sum(axis=0)

def qonuchic(trajectory, reference):
    """ Add formula """
    raise NotImplementedError
    return qvalue(trajectory, reference, min_seq_sep=4)

def qinterface():
    """ Add formula """
    raise NotImplementedError



