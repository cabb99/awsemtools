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

# TODO: Priority qvalue

class Trajectory:
    """A class for parsing and visualizing molecular dynamics (MD) trajectories.

    This class provides methods for manipulating and analyzing MD trajectories, as well as visualizing them
    using the NGLView library. It is initialized with a PDB file and a DCD file, which contain the static
    structure of the protein and the trajectory data, respectively. The class also allows to specify a
    selection of atoms to focus on, and to select a subset of frames from the trajectory to analyze.

    Parameters:
    - pdb_file (str): The path to the PDB file containing the reference structure of the protein.
    - dcd_file (str): The path to the DCD file containing the trajectory data.
    - selection (str): An optional selection of atoms to focus on, specified using ProDy's atom selection syntax.
                     The default value is 'all', which includes all atoms in the protein.

    Attributes:
    - pdb_file (str): The path to the PDB file.
    - dcd_file (str): The path to the DCD file.
    - pdb (ProDy AtomGroup): The static structure of the protein, as parsed by ProDy.
    - dcd (ProDy AtomGroup): The trajectory data, as parsed by ProDy.
    - reference (ProDy AtomGroup): The subset of atoms in the protein specified by the selection.
    - indices (np.ndarray): The indices of the atoms in the reference AtomGroup.
    - frames (np.ndarray): The indices of the frames in the trajectory to analyze.
    - features (pd.DataFrame): A DataFrame for storing calculated features of the trajectory.
    """
    #TODO: use pyemma for features

    def __init__(self, pdb_file, dcd_file, selection='all'):
        """Initializes the Trajectory object using the pdb file and dcd file. The
           reference atoms can be specified through the 'selection' argument using 
           ProDy's atom selection syntax. 
           http://prody.csb.pitt.edu/manual/reference/atomic/select.html"""
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
        """Creates a copy of the Trajectory object.

        Returns:
        - cls (Trajectory): A copy of the Trajectory object.
        """
        cls=Trajectory(self.pdb_file,self.dcd_file)
        cls.set_selection(indices=self.indices)
        cls.set_frames(frames=self.frames)
        return cls

    def __repr__(self):
        fs,ps,_ = self.dcd.getCoordsets().shape
        return f'Trajectory: {len(self.indices)}/{ps} particles, {len(self.frames)}/{fs} frames selected '
    
    @property
    def coords(self):
        return self.dcd[self.frames].getCoordsets()[:,list(self.indices)]
    
    def superpose(self,reference=0):
        """Superposes the trajectory onto the static structure of the protein.
           reference: index of the reference coordinate used for the superposition (default=0)
            """
        #TODO: add reference frame
        cls = self.copy()
        cls.dcd.superpose(ref = reference)
        return cls

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
        #TODO add 3Dmol.js view option
        return nglview.show_prody(self.pdb)


    def compute_RMSD(self):
        return self.dcd.getRMSDs()

    def compute_RMSF(self):
        return self.dcd.getRMSFs()

    def compute_MSF(self):
        return self.dcd.getMSFs()

    def compute_qvalue(self, min_seq_sep=3, contact_threshold=9.5):
        raise NotImplementedError

    def contact_map(self):
        raise NotImplementedError

        
    def vmd(self):
        import subprocess
        return subprocess.Popen(['vmd', self.pdb_file, self.dcd_file], stdin=subprocess.PIPE)

    def compute_feature():
        raise NotImplementedError


class PCA:
    def __init__(self,features,**args):
        self.pca=sklearn_PCA(**args)
        self.features=features
        self.mean=self.features.mean(axis=0)
        centered_features=self.features-self.mean
        self.pca.fit(centered_features)

    @property
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

    def plot_projection(self, n_PC=5, projection=None):
        
        import seaborn as sns
        import pandas as pd
        if projection == None:
            projection = self.projection
        data=pd.DataFrame()
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
        #TODO
        def featurization_method(trajectory):
            s=trajectory.coords.shape
            return trajectory.coords.reshape(s[0],-1)
        features=featurization_method(trajectory)
        pca=PCA(features,**args)
        pca.featurization_method=featurization_method
        return pca

    @classmethod
    def fit_contact(cls, trajectory,**args):
        #TODO
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



class EnergyLandscape():
    pass