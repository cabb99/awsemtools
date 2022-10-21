"""
Unit and regression test for the awsemtools package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import awsemtools


def test_awsemtools_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "awsemtools" in sys.modules

def test_q_value():
    awsemtools.q_value()

def test_strain_pca():
    awsemtools.strain_pca()

def test_coordinate_pca():
    awsemtools.coordinate_pca()
