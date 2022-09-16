import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class Node(object):
    def __init__(self, val=None, d=None):
        self.val = val
        self.d = d
        self.children = []


def information_gain(X, y, truth_value='success'):
    """
    Args:
        X: Attributes
        y: Target attribute
        truth_value: Value of positive outcome of target
    Returns: float total information gain from attribute
    """
    information_gains = []

    for i in range(X.shape[1]):
        x = X.iloc[:,[i]]

        # create dict with number of distinct values of an attribute
        # ex {'Sunny': [0,0], 'Cloudy': [0,0]}

        x_entropies = {}
        for attribute in np.array(x.value_counts().keys()):
            x_entropies[attribute[0]] = [0, 0]

        # fill dict with counts of values corresponding to the binary output
        # ex {'Sunny': [2,3], 'Cloudy': [0,4]}
        for j in np.array(pd.concat([x,y], axis=1)):
            idx = 0 if j[1] == truth_value else 1
            x_entropies[j[0]][idx] += 1

        y_entropy = entropy(np.array(y.value_counts()))
        n_values = len(y)
        for v in x_entropies.values():
            y_entropy -= (sum(v)/n_values)*entropy(np.array(v))
        information_gains.append(y_entropy)
        print(f'{x.columns.values[0]} information gain: {y_entropy:.4f}')
    return information_gains


class DecisionTree:
    def __init__(self):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.root = None
        self.n = 0

    def fit(self, X, y):
        """
        Generates a decision tree for classification

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.n += 1
        print(f'\n___________________Node {self.n} created__________________ ')
        # Get the column of the largest information gain
        IG = np.array(information_gain(X, y))
        indices = X.columns.values
        largest = indices[IG.argmax()]
        #print(IG)
        print(indices)
        #print(largest)

        # create node, if its the first one add it as root
        n = Node(d=largest )
        if not self.root:
            self.root = n

        # If largest information gain is 0 we have a leaf node
        # y should only contain 1 value
        # Set the nodes value to the outcome and return
        if IG.max() == 0:
            n.val = y.iloc[-1]
            print(f'\nLeaf node value: {n.val}')
            return n

        # for each value in column of largest attribute, separate indices to individual lists
        new_dim = X[largest]
        new_indices = {k: [] for k in np.array(new_dim.value_counts().keys())}
        for idx, item in enumerate(new_dim):
            new_indices[item].append(idx)
        print(new_indices)
        # recurse the fit function for every list with corresponding indices
        nc = X.drop(columns=largest)
        print(nc)

        for vi in new_indices.values():
            nc_x = nc.drop(index=[x for x in range(nc.shape[0]) if x not in vi])
            nc_x.reset_index(drop=True, inplace=True)

            nc_y = y.drop(index=[x for x in range(nc.shape[0]) if x not in vi])
            nc_y.reset_index(drop=True, inplace=True)

            nn = self.fit(nc_x, nc_y)
            n.children.append(nn)
        return n

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement 
        raise NotImplementedError()


    def traverse(self, n):
        print(n.d)
        print(n.val)
        for c in n.children:
            self.traverse(c)
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        nodes = self.traverse(self.root)
        return []

# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



