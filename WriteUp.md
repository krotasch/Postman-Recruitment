Task 4: Decision Tree and Random Forest from Scratch

What I did:
The objective of this task is to build a decision tree from scratch, then use my implementation to create a random forest.
The implementation is being developed from scratch using Python and NumPy.

## 2.How I did it:

### Node Representation

A `Node` data class is used to represent each node of the decision tree. Each node stores these values:

1. The characteristic used for splitting an array 
2. The threshold of the split which I got from the characteristic I chose for the split
3. References to the left and right child nodes
4. The value stored at a leaf
5. The depth of the node

This provides a simple mini-structure for representing the tree.
### Decision Tree Initial Creation

The `DecisionTree` class is made up of:
* `task` — classification (`"clsf"`) or regression
* `max_depth` — maximum allowed tree depth
* `min_samples` — minimum number of samples intended for a split
The tree also maintains a `root` attribute, which will contain the root node after training of data.
## 3. Impurity and Loss Functions

### Gini Impurity
For classification, I used Gini impurity to measure how random, or rather non-unique the classes are within a node:
$$
Gini = 1-\sum_{i=1}^{k}p_i^2
$$

where \(p_i\) is the proportion of samples belonging to class \(i\).
The implementation calculates the frequency of every unique class using NumPy and converts these frequencies into probabilities.

### Mean Squared Error
For regression, MSE is used:

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\bar y)^2
$$
where \(\bar y\) is the mean target value of the node.

A node in which all target values are identical has an MSE of 0.
PLEASE NOTE THAT EVEN THOUGH I DID CREATE A FUNCTION TO INVOLVE REGRESSION USING MSE I COULD NOT IMPLEMENT IT COMPLETELY BY THE END , BY MYSELF.
## 4. Dataset Splitting

For a selected feature and threshold, the dataset is divided into two groups:
* Left: feature value \(\leq\) threshold
* Right: feature value \(>\) threshold

Boolean NumPy masks are used to perform the split.

An important property of this implementation is that the same mask is applied to both `X` and `y`. Therefore, each feature vector remains paired with its corresponding target value after splitting.

## 5. Split Evaluation

Each candidate split is evaluated using a weighted combination of the impurity/loss of the two resulting child nodes:

$$
Score =
\frac{n_L}{n}Score_L+
\frac{n_R}{n}Score_R
$$

where \(n_L\) and \(n_R\) are the numbers of samples in the left and right child nodes.
For classification, the child scores are Gini impurities. For regression, they are MSE values.
The split with the lowest weighted score is considered the best split.
## 6. Candidate Threshold Generation

For every feature, the implementation first obtains the sorted unique feature values.

Candidate thresholds are generated as the midpoints between consecutive unique values:

$$
t_i=\frac{x_i+x_{i+1}}{2}
$$

This avoids unnecessary duplicate thresholds and ensures that a threshold is considered between every pair of distinct feature values.
If a feature contains only one unique value, no valid threshold is generated.
The function therefore returns the feature index, threshold, and score of the best split found.

