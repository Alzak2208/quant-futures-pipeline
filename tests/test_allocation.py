"""Smoke tests for allocation module."""

import numpy as np
import pytest

from src.allocation import pcaweights


class TestPCAWeights:
    def test_basic_2x2(self):
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        w = pcaweights(cov)
        assert w.shape == (2,)
        assert np.isfinite(w).all()

    def test_identity_covariance(self):
        cov = np.eye(3)
        risk = np.array([1 / 3, 1 / 3, 1 / 3])
        w = pcaweights(cov, riskDistr=risk, riskTarget=1.0)
        assert w.shape == (3,)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            pcaweights(np.ones((2, 3)))

    def test_bad_risk_dist_sum_raises(self):
        cov = np.eye(2)
        with pytest.raises(ValueError):
            pcaweights(cov, riskDistr=np.array([0.5, 0.3]))
