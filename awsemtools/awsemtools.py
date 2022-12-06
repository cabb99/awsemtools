"""Provide the primary functions."""

import pyemma
import prody
from sklearn.decomposition import PCA as sklearn_PCA
from scipy.spatial import distance as sdist
import numpy as np
import nglview
from nglview.sandbox.interpolate import smooth as ngl_smooth
import numpy as np
import pandas as pd


from .compute import *


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


class Trajectory:
    
    def __init__(self, pdb_file, dcd_file, selection='all'):
        self.pdb_file=pdb_file
        self.dcd_file=dcd_file
        
        self.pdb = prody.parsePDB(pdb_file)
        self.dcd = prody.parseDCD(dcd_file)
        self.dcd.setCoords(self.pdb)
        
        self.reference=self.pdb.select(selection)
        self.dcd.setAtoms(self.reference)
        self.indices=self.reference.getIndices()
        
        self.frames=np.arange(0,len(self.dcd))
        self.features=pd.DataFrame(index=range(len(self.dcd)))

    def copy(self):
        cls=Trajectory(self.pdb_file,self.dcd_file)
        cls.set_selection(indices=self.indices)
        cls.set_frames(frames=self.frames)
        return cls
        
    def superpose(self):
        self.dcd.superpose()

    def set_selection(self,selection='all', indices=None):
        if indices is None:
            pdb_selection=self.pdb.select(selection)
            if pdb_selection:
                indices=self.pdb.select(selection).getIndices() 
            else:
                indices=[]
        self.indices=np.array(indices)
        self.reference=self.pdb.select(f"index {' '.join([str(i) for i in indices])}")

    def set_frames(self, start=0, stop=-1, stride=1, frames=None):
        if frames is None:
            frames=np.arange(0,len(self.dcd))[start:stop:stride]
        self.frames=np.array(frames)
    

    def show_trajectory(self, smooth=0):
        temp_dcd=self.dcd[0:1]
        smooth_trajectory=ngl_smooth(self.dcd.getCoordsets(), method='savgol_filter', window_length=smooth, polyorder=3)
        temp_dcd.addCoordset(smooth_trajectory)
        temp_dcd.delCoordset(0)
        temp_dcd._n_csets=len(temp_dcd.getCoordsets())
        return nglview.show_prody(temp_dcd)

    def show_structure(self):
        return nglview.show_prody(self.pdb)

    def compute_RMSD(self):
        return self.dcd.getRMSDs()

    def compute_RMSF(self):
        return self.dcd.getRMSFs()

    def compute_MSF(self):
        return self.dcd.getMSFs()

    def compute_qvalue(self, min_seq_sep=3, contact_threshold=9.5):
        raise NotImplementedError
        

    def vmd(self):
        import subprocess
        return subprocess.Popen(['vmd', self.pdb_file, self.dcd_file], stdin=subprocess.PIPE)

    @property
    def coords(self):
        return self.dcd[self.frames].getCoordsets()[:,list(self.indices)]


class PCA:
    def __init__(self,features,featurization_method,**args):
        self.pca=sklearn_PCA(**args)
        self.features=features
        self.mean=self.features.mean(axis=0)
        centered_features=self.features-self.mean
        self.pca.fit(centered_features)
        def featurization_method(features):
            return features
        self.featurization_method=featurization_method


    @property
    def eigenvalues(self):
        return self.pca.explained_variance_

    
    def eigenvalues(self):
        return self.pca.explained_variance_

    @property
    def eigenvectors(self):
        return self.pca.components_

    @property
    def projection(self):
        centered_features=self.features-self.mean
        return self.pca.transform(centered_features)

    def project(self, trajectory):
        features=self.featurization_method(trajectory)
        centered_features=features-self.mean
        return self.pca.transform(centered_features)

    def plot_projection(self,n_PC=5):
        import seaborn as sns
        import pandas as pd
        data=pd.DataFrame(self.projection)
        data.index.name='frames'
        data.columns.name='PC'
        data=data.reset_index()
        data=data.melt(id_vars='frames')
        data['PC']+=1
        return sns.relplot(data[data['PC']<=n_PC],x='frames',y='value',row='PC',kind="line",height=1,aspect=10, hue='PC', palette="crest")

    def plot_explained_variance(self, n_PC):
        raise NotImplemented
    
    @classmethod
    def fit_coordinates(cls, trajectory,**args):
        def featurization_method(trajectory):
            s=trajectory.coords.shape
            return trajectory.coords.reshape(s[0],-1)
        features=featurization_method(trajectory)
        pca=PCA(features,**args)
        pca.featurization_method=featurization_method
        return pca

    @classmethod
    def fit_strain(cls, trajectory,**args):
        def featurization_method(trajectory):
            s=trajectory.coords.shape
            return trajectory.coords.reshape(s[0],-1)
        features=featurization_method(trajectory)
        pca=PCA(features,**args)
        pca.featurization_method=featurization_method
        return pca

    @classmethod
    def fit_contact(cls, trajectory,**args):
        def featurization_method(trajectory):
            s=trajectory.coords.shape
            return trajectory.coords.reshape(s[0],-1)
        features=featurization_method(trajectory)
        pca=PCA(features,**args)
        pca.featurization_method=featurization_method
        return pca

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())


