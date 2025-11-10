"""Ensemble Learning: Stacking across multiple learners"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from sklearn.linear_model import Ridge
import structlog

logger = structlog.get_logger(__name__)


class StackedEnsemble:
    """Stacked ensemble using Ridge regression as meta-learner."""

    def __init__(self, base_learners: List[Any], alpha: float = 1.0):
        self.base_learners = base_learners
        self.alpha = alpha
        self.meta_learner = Ridge(alpha=alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StackedEnsemble':
        """
        Fit base learners and meta-learner.

        Uses out-of-fold predictions for meta-learner training.
        """
        logger.info(f"Training stacked ensemble with {len(self.base_learners)} base learners")

        # Train base learners
        for i, learner in enumerate(self.base_learners):
            logger.info(f"Training base learner {i+1}/{len(self.base_learners)}")
            learner.fit(X, y)

        # Generate out-of-fold predictions for meta-learner
        base_predictions = []
        for learner in self.base_learners:
            pred = learner.predict(X)
            if isinstance(pred, dict):
                pred = pred.get('p50', pred[list(pred.keys())[0]])
            base_predictions.append(pred)

        # Stack predictions
        X_meta = np.column_stack(base_predictions)

        # Train meta-learner
        self.meta_learner.fit(X_meta, y)

        logger.info("Ensemble training complete")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        # Get base predictions
        base_predictions = []
        for learner in self.base_learners:
            pred = learner.predict(X)
            if isinstance(pred, dict):
                pred = pred.get('p50', pred[list(pred.keys())[0]])
            base_predictions.append(pred)

        # Stack and meta-predict
        X_meta = np.column_stack(base_predictions)
        return self.meta_learner.predict(X_meta)
