"""Provide the primary functions."""

import pyemma
import prody
from sklearn.decomposition import PCA as sklearn_PCA
import numpy as np


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

class PCA():
    """Compute PCA on the alanine dipeptide trajectory."""

    def __init__(self, pdb, dcd, selection='protein and name CA', start=0, stop = -1, stride = 1):
        """Initialize the PCA object. """

        self.pdb = prody.parsePDB(pdb)
        self.dcd = prody.parseDCD(dcd)
        self.selection=self.pdb.select(selection)
        self.dcd.setAtoms(self.selection)

        #Coordinates
        self.coords=self.dcd[start:stop:stride].getCoordsets()

        #PCA
        s=self.coords.shape
        self.features=self.coords.reshape(s[0],-1)
        self.center=self.features.mean(axis=0)
        centered_features=self.features-self.center

        self.eigenvalues=pca.explained_variance_
        self.eigenvectors=pca.components_
        self.projection=pca.fit(centered_features)


    def get_eigenvalues(self):
        """Return the eigenvalues of the PCA object."""
        return self.pca.getEigvals()

    def get_eigenvectors(self):
        """Return the eigenvectors of the PCA object."""
        return self.pca.getEigvecs()

    def get_variance(self):
        """Return the variance of the PCA object."""
        return self.pca.getVariances()

    def get_mean(self):
        """Return the mean of the PCA object."""
        return self.traj.getMean()

    def get_coordinates(self):
        """Return the coordinates of the PCA object."""
        return self.dcd.getCoords()

    def get_covariance(self):
        """Return the covariance of the PCA object."""
        return self.pca.getCovariance()

    def get_projection(self):
        """Return the projection of the PCA object."""
        return self.pca.getProjection()

    # def get_transformed(self):
    #     """Return the transformed coordinates of the PCA object."""
    #     return self.pca.getTransformed()

    # def get_transformed_coordinates(self):
    #     """Return the transformed coordinates of the PCA object."""
    #     return self.tpcaraj.getTransformedCoords()

    # def get_transformed_variance(self):
    #     """Return the transformed variance of the PCA object."""
    #     return self.pca.getTransformedVariances()

    # def get_transformed_eigenvalues(self):
    #     """Return the transformed eigenvalues of the PCA object."""
    #     return self.pca.getTransformedEigvals()

    # def get_transformed_eigenvectors(self):
    #     """Return the transformed eigenvectors of the PCA object."""
    #     return self.pca.getTransformedEigvecs()

    # def get_transformed_covariance(self):
    #     """Return the transformed covariance of the PCA object."""
    #     return self.pca.getTransformedCovariance()

    # def get_transformed_projection(self):
    #     """Return the transformed projection of the PCA object."""
    #     return self.pca.getTransformedProjection()

    # def get_cumulative_variance(self):
    #     """Return the cumulative variance of the PCA object."""
    #     return self.pca.getCumvar()

class Structure():
    pass

class Trajectory():
    
    def __init__(self, pdb, dcd, selection='all', start=0, stop=-1, stride=1):
        self.pdb = prody.parsePDB(pdb)
        self.dcd = prody.parseDCD(dcd)
        self.dcd.setAtoms(self.pdb.select('all'))
        self.indices=self.pdb.select('all').getIndices()
        self.frames=np.arange(0,len(self.dcd))

    def set_selection(self,selection='all', indices=None):
        if indices is None:
            pdb_selection=self.pdb.select(selection)
            if pdb_selection:
                indices=self.pdb.select(selection).getIndices() 
            else:
                indices=[]
        self.indices=np.array(indices)

    def set_frames(self, start=0, stop=-1, stride=1, frames=None):
        if frames is None:
            frames=np.arange(0,len(self.dcd))[start:stop:stride]
        self.frames=np.array(frames)
    
    @property
    def coords(self):
        return self.dcd[self.frames].getCoordsets()[:,list(self.indices)]


class PCA():
    def __init__(self,features,**args):
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

    def visualize_inside_notebook():
        raise NotImplemented

    @classmethod
    def fit_strain(cls, trajectory,**args):
        def featurization_method(trajectory):
            s=trajectory.coords.shape
            return trajectory.coords.reshape(s[0],-1)
        features=featurization_method(trajectory)
        pca=PCA(features,**args)
        pca.featurization_method=featurization_method
        return pca

        

class ContactPCA():
    pass

class StrainPCA():
    pass



def compute_pca():
    """Compute PCA on the alanine dipeptide trajectory."""
    traj = prody.parsePDB('alanine_dipeptide.pdb')
    traj = prody.selectAtoms(traj, 'protein and name CA')
    traj = prody.calcCoordsets(traj)
    traj = prody.calcPCA(traj)
    return traj

def compute_rmsd():
    """Computes the RMSD of a pdb trajectory"""
    pdb = prody.parsePDB('alanine_dipeptide.pdb')
    traj = prody.parseDCD('alanine_dipeptide.dcd')
    traj = prody.selectAtoms(traj, 'protein and name CA')
    rmsd_traj = prody.calcRMSD(traj, pdb)
    return rmsd_traj
    



if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())


