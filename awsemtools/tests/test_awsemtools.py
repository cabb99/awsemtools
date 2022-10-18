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
