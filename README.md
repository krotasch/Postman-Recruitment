# Postman-Recruitment
This is my attempt to solve the task for Postman Club's recruitment for the '26 Batch. I have chosen to attempt the 4 question in the document provided by them.

/x How to run it?
->I did try it on a few test cases at every point. Here are those and the expected outputs provided below each of them:-
/1. For testing Gini impurity:-
tree=DecisionTree() 
print(tree._gini(np.array([1, 1, 1, 1])))
Expected output:0.0
Since, the array has all elements equal the probability of 1 is 1. This means that Gini impurity is 1-(Sigma(Pi)^2)=1-1^2=0
/2.MSE
tree=DecisionTree()
print(tree._mse(np.array([5, 5, 5, 5])))
Expected output:2.6666666666666665
/3.tree=DecisionTree()
print(tree._split_score( np.array([0, 1]), np.array([0, 1]) ))
Expected output:0.5
/4. Characteristic choosing using NumPy Index
tree=DecisionTree()
X = np.array([ [1, 10], [2, 20], [3, 30] ])
X[:, 1]
Expected output:[10, 20, 30]
