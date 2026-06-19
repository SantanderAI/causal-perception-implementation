# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

"""
Linear Additive Noise Model (ANM) with configurable DAG edges.

Fits OLS for mediators and logistic regression for the outcome Y.
Supports interventions via do-operator: propagate through the DAG
and return predicted risk probabilities for each individual.

Usage:
    scm = LinearANM(edges=CHIAPPA_EDGES)
    scm.fit(train_df)
    probs = scm.predict_y(test_df, interventions={"A": 1})
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression

# -- DAG definitions ----------------------------------------------------------
# Each DAG is a dict: node -> list of parent nodes.
# Root nodes have an empty parent list.
# Topological order: A, C (roots) -> S1, S2, S3, R1, R2 (mediators) -> Y (outcome)

CHIAPPA_FULL = {
    "A": [],
    "C": [],
    "S1": ["A", "C"],
    "S2": ["A", "C"],
    "S3": ["A", "C"],
    "R1": ["A", "C"],
    "R2": ["A", "C"],
    "Y": ["A", "C", "S1", "S2", "S3", "R1", "R2"],
}

CHIAPPA_NO_AY = {
    "A": [],
    "C": [],
    "S1": ["A", "C"],
    "S2": ["A", "C"],
    "S3": ["A", "C"],
    "R1": ["A", "C"],
    "R2": ["A", "C"],
    "Y": ["C", "S1", "S2", "S3", "R1", "R2"],  # A -> Y removed
}

CHIAPPA_NO_CY = {
    "A": [],
    "C": [],
    "S1": ["A", "C"],
    "S2": ["A", "C"],
    "S3": ["A", "C"],
    "R1": ["A", "C"],
    "R2": ["A", "C"],
    "Y": ["A", "S1", "S2", "S3", "R1", "R2"],  # C -> Y removed
}

# Legacy constants (kept for backward compatibility with existing scripts)
TOPO_ORDER = ["A", "C", "S1", "S2", "S3", "R1", "R2", "Y"]
MEDIATORS = ["S1", "S2", "S3", "R1", "R2"]
ROOTS = ["A", "C"]


def topological_sort(edges):
    """Kahn's algorithm: return nodes in topological order from a DAG dict."""
    in_degree = {node: len(parents) for node, parents in edges.items()}
    queue = [n for n in edges if in_degree[n] == 0]
    order = []
    while queue:
        queue.sort()  # deterministic tie-breaking
        node = queue.pop(0)
        order.append(node)
        for child, parents in edges.items():
            if node in parents:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    if len(order) != len(edges):
        raise ValueError("Graph has a cycle - not a valid DAG.")
    return order


class LinearANM:
    """
    Linear Additive Noise Model with logistic outcome.

    Mediators (S1, S2, S3, R1, R2) are modeled as linear regressions.
    Outcome (Y) is modeled as logistic regression returning P(Y=1).
    """

    def __init__(self, edges, outcome="Y", y_model=None):
        """
        Parameters
        ----------
        edges : dict
            DAG structure as {node: [parent_nodes]}.
        outcome : str
            Name of the outcome node (default: "Y").
        y_model : sklearn classifier, optional
            Custom classifier for the outcome. Must implement fit(X, y) and
            predict_proba(X). Defaults to LogisticRegression(max_iter=1000).
        """
        self.edges = edges
        self.outcome = outcome
        self.y_model = y_model
        self.models = {}  # fitted sklearn models per node
        self.root_stats = {}  # {node: (mean, std)} for root nodes
        # Derive topology from edges
        self.topo_order = topological_sort(edges)
        self.roots = [n for n in self.topo_order if len(edges[n]) == 0]
        self.mediators = [n for n in self.topo_order if n != outcome and len(edges[n]) > 0]

    def fit(self, df):
        """
        Fit structural equations from data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with columns A, C, S1, S2, S3, R1, R2, Y.
        """
        # Root nodes: store empirical distribution stats
        for node in self.roots:
            self.root_stats[node] = {
                "mean": df[node].mean(),
                "std": df[node].std(),
            }

        # Mediators: OLS
        for node in self.mediators:
            parents = self.edges[node]
            if not parents:
                continue
            X = df[parents].values
            y = df[node].values
            model = LinearRegression().fit(X, y)
            self.models[node] = model

        # Outcome: custom classifier or logistic regression
        parents_y = self.edges[self.outcome]
        X_y = df[parents_y].values
        y_y = df[self.outcome].values
        if self.y_model is not None:
            model_y = clone(self.y_model).fit(X_y, y_y)
        else:
            model_y = LogisticRegression(max_iter=1000).fit(X_y, y_y)
        self.models[self.outcome] = model_y

        return self

    def intervene(self, df, interventions):
        """
        Apply do-operator and propagate through the DAG.

        Parameters
        ----------
        df : pd.DataFrame
            Test data (original values used for non-intervened root nodes).
        interventions : dict
            {variable_name: value} for do-operator interventions.

        Returns
        -------
        pd.DataFrame
            Intervened data with all variables propagated.
        np.ndarray
            Predicted P(Y=1) for each individual.
        """
        n = len(df)
        result = pd.DataFrame(index=range(n))

        # Process nodes in topological order
        for node in self.topo_order:
            if node in interventions:
                # Intervened variable: set to fixed value
                result[node] = interventions[node]

            elif node in self.roots:
                # Non-intervened root: keep original values
                result[node] = df[node].values

            elif node == self.outcome:
                # Outcome: classifier -> probability
                parents = self.edges[self.outcome]
                X = result[parents].values
                probs = self.models[self.outcome].predict_proba(X)[:, 1]
                result[node] = probs

            else:
                # Mediator: OLS prediction
                parents = self.edges[node]
                X = result[parents].values
                result[node] = self.models[node].predict(X)

        y_probs = result[self.outcome].values
        return result, y_probs

    def abduct(self, df):
        """
        Abduction step: infer individual noise terms U from observed data.

        For mediators (linear): U_node = observed - predicted (residual).
        For Y (logistic): U_Y = observed Y (binary label, kept as-is for
        counterfactual logit adjustment).

        Parameters
        ----------
        df : pd.DataFrame
            Observed data with columns A, C, S1, S2, S3, R1, R2, Y.

        Returns
        -------
        dict
            {node: np.ndarray of noise terms} for each non-root node.
        """
        noise = {}
        for node in self.mediators:
            parents = self.edges[node]
            if not parents:
                noise[node] = np.zeros(len(df))
                continue
            X = df[parents].values
            predicted = self.models[node].predict(X)
            noise[node] = df[node].values - predicted

        # Outcome: no noise abduction - counterfactual probability is computed
        # from the classifier on counterfactual parent values.

        return noise

    def counterfactual(self, df, interventions):
        """
        Counterfactual inference (3rd rung): abduction + intervention + prediction.

        For each individual, infer their noise terms from the *observed* data,
        then re-propagate the SCM with the intervention applied and the
        individual-specific noise terms preserved.

        Parameters
        ----------
        df : pd.DataFrame
            Observed test data.
        interventions : dict
            {variable_name: value} for counterfactual intervention.

        Returns
        -------
        pd.DataFrame
            Counterfactual data with all variables.
        np.ndarray
            Counterfactual P(Y=1) for each individual.
        """
        # Step 1: Abduction - infer noise from observed data
        noise = self.abduct(df)

        n = len(df)
        result = pd.DataFrame(index=range(n))

        # Steps 2 & 3: Intervention + Prediction with preserved noise
        for node in self.topo_order:
            if node in interventions:
                result[node] = interventions[node]

            elif node in self.roots:
                result[node] = df[node].values

            elif node == self.outcome:
                parents = self.edges[self.outcome]
                X = result[parents].values
                # Counterfactual probability from classifier on CF parents
                cf_probs = self.models[self.outcome].predict_proba(X)[:, 1]
                result[node] = cf_probs

            else:
                parents = self.edges[node]
                X = result[parents].values
                # Counterfactual value = predicted + individual noise
                result[node] = self.models[node].predict(X) + noise[node]

        y_probs = result[self.outcome].values
        return result, y_probs


def compute_standard_errors(model, X):
    """
    Compute logistic regression coefficient SEs via Fisher information.

    Parameters
    ----------
    model : fitted LogisticRegression
        Must have .predict_proba() and .coef_ attributes.
    X : np.ndarray, shape (n, p)
        Design matrix (without intercept column).

    Returns
    -------
    np.ndarray, shape (p+1,)
        Standard errors for [intercept, coef_1, ..., coef_p].
    """
    probs = model.predict_proba(X)[:, 1]
    W = probs * (1 - probs)
    X_int = np.column_stack([np.ones(len(X)), X])
    fisher = X_int.T @ (X_int * W[:, None])
    cov = np.linalg.inv(fisher)
    return np.sqrt(np.diag(cov))


if __name__ == "__main__":
    # Quick sanity check
    from src.data_prep import load_data

    train, test = load_data()

    print("=== Chiappa Full DAG ===")
    scm1 = LinearANM(edges=CHIAPPA_FULL).fit(train)
    _, probs_do0 = scm1.intervene(test, {"A": 0})
    _, probs_do1 = scm1.intervene(test, {"A": 1})
    print(f"do(A=0): mean P(Y=1) = {probs_do0.mean():.4f}, std = {probs_do0.std():.4f}")
    print(f"do(A=1): mean P(Y=1) = {probs_do1.mean():.4f}, std = {probs_do1.std():.4f}")

    print("\n=== Chiappa Full DAG - Counterfactual ===")
    _, cf_probs_0 = scm1.counterfactual(test, {"A": 0})
    _, cf_probs_1 = scm1.counterfactual(test, {"A": 1})
    print(f"CF(A=0): mean P(Y=1) = {cf_probs_0.mean():.4f}, std = {cf_probs_0.std():.4f}")
    print(f"CF(A=1): mean P(Y=1) = {cf_probs_1.mean():.4f}, std = {cf_probs_1.std():.4f}")

    print("\n=== Chiappa No A->Y ===")
    scm2 = LinearANM(edges=CHIAPPA_NO_AY).fit(train)
    _, probs_do0_b = scm2.intervene(test, {"A": 0})
    _, probs_do1_b = scm2.intervene(test, {"A": 1})
    print(f"do(A=0): mean P(Y=1) = {probs_do0_b.mean():.4f}, std = {probs_do0_b.std():.4f}")
    print(f"do(A=1): mean P(Y=1) = {probs_do1_b.mean():.4f}, std = {probs_do1_b.std():.4f}")

    print("\n=== Chiappa No A->Y - Counterfactual ===")
    _, cf_probs_0_b = scm2.counterfactual(test, {"A": 0})
    _, cf_probs_1_b = scm2.counterfactual(test, {"A": 1})
    print(f"CF(A=0): mean P(Y=1) = {cf_probs_0_b.mean():.4f}, std = {cf_probs_0_b.std():.4f}")
    print(f"CF(A=1): mean P(Y=1) = {cf_probs_1_b.mean():.4f}, std = {cf_probs_1_b.std():.4f}")
