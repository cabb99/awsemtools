"""
Unit and regression test for the awsemtools package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import awsemtools as awt


def test_awsemtools_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "awsemtools" in sys.modules

def test_q_value():
    qvalue = awt.q_value('awsemtools/tests/data/p65p50.pdb','awsemtools/tests/data/p65p50.dcd', type='wolynes')

    q_value.monomer() #multiple values
    q_value.interface() #multiple interfaces
    q_relative('awsemtools/tests/data/p65p50.pdb')
    q_value.globalq()

def test_trajectory():
    trajectory=awt.Trajectory(pdb_file='awsemtools/tests/data/p65p50.pdb',dcd_file='awsemtools/tests/data/p65p50.dcd')
    assert trajectory.coords.shape == (1000, 3073, 3)


def test_trajectory_selection():
    trajectory=awt.Trajectory(pdb_file='awsemtools/tests/data/p65p50.pdb',dcd_file='awsemtools/tests/data/p65p50.dcd')
    trajectory.set_selection('name CA')
    print(trajectory.coords.shape) ==(1000, 615, 3)

def test_trajectory_frames():
    trajectory=awt.Trajectory(pdb_file='awsemtools/tests/data/p65p50.pdb',dcd_file='awsemtools/tests/data/p65p50.dcd')
    trajectory.set_frames(stride=5)
    print(trajectory.coords.shape) == (200, 3073, 3)

def test_trajectory_complex_selection():
    trajectory=awt.Trajectory(pdb_file='awsemtools/tests/data/p65p50.pdb',dcd_file='awsemtools/tests/data/p65p50.dcd')
    trajectory.set_selection('name CA')
    trajectory.set_frames(stride=5)
    print(trajectory.coords.shape) == (200, 615, 3)

def test_trajectory_empty_selection():
    trajectory=awt.Trajectory(pdb_file='awsemtools/tests/data/p65p50.pdb',dcd_file='awsemtools/tests/data/p65p50.dcd')
    assert trajectory.indices.shape==(3073,)
    assert trajectory.frames.shape==(1000,)
    assert trajectory.coords.shape==(1000, 3073, 3)
    trajectory.set_selection(indices=[])
    assert trajectory.indices.shape==(0,)
    assert trajectory.frames.shape==(1000,)
    assert trajectory.coords.shape==(1000, 0, 3)
    trajectory.set_frames(start=100000,stride=100000)
    assert trajectory.indices.shape==(0,)
    assert trajectory.frames.shape==(0,)
    assert trajectory.coords.shape==(0, 0, 3)

def test_pca():
    import awsemtools as awt

    pca = awt.PCA('awsemtools/tests/data/p65p50.pdb','awsemtools/tests/data/p65p50.dcd')
    pca.fit()

    pca.animate() # widget if possible.
    pca.project() # other

def test_mdtraj_wrappers():
    awt.rmsd()
    awt.rg()
    awt.contact_map()

def test_pyemma_wrappers():
    awt.free_energy_analysis(pdb, dcd, reaction_coordinate, reaction_coordinate2) #WHAM, pyemma?
    awt.clustering() #hierarchical
    awt.MSM = pyemma.msm.MSM #Markov state models

def test_colective_variables():
    #Test for collective variables
    #Intermolecular contacts

    qvalue=Force.define(f"(1/{normalization})*gamma_ij*exp(-(r-r_ijN)^2/(2*sigma_ij^2))")
    qvalue.addPerBondParameter("gamma_ij")
    qvalue.addPerBondParameter("r_ijN")
    qvalue.addPerBondParameter("sigma_ij")

    for structure_interaction in structure_interactions:
        qvalue.addBond(*structure_interaction)
    # qvalue.setForceGroup(forceGroup) Maybe implicit

    trajectories=[]
    for i in np.arange(0,1,20):
        system.addForce(qvalue,bias=0.1)
        trajectories+=[system.run_simulation()]

    wham = awt.WHAM(trajectories, qvalue)
    wham.compute_energy_landscape()



