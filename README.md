# SVM_python_code
The repository contains the python code to implement Support Vector Machine algorithm.

The input file mydata.txt contains the dataset as follows:
For age: Youth=1,Middle=2,Senior=3 For income: Low=1,Medium=2,High=3 For student: Yes=1,No=2
For credit rating: Fair=1,Excellent=2 For buys computer: Yes=1,No=2
DATASET:
age	income	student	credit_rating buys_computer
1	3	2	1	           2
1	3	2	2	           2
2	3	2	1	           1
3	2	2	1	           1
3	1	1	1	           1
3	1	1	2	           2
2	1	1	2	           1
1	2	2	1	           2
1	1	1	1	           1
3	2	1	1	           1
1	2	1	2	           1
2	2	2	2	           1
2	3	1	1	           1
3	2	2	2	           2



predict(X)[source]
Perform classification on samples in X.
For an one-class model, +1 or -1 is returned.
Parameters:	
X : {array-like, sparse matrix}, shape (n_samples, n_features)
For kernel=”precomputed”, the expected shape of X is [n_samples_test, n_samples_train]
Returns:	
y_pred : array, shape (n_samples,)
Class labels for samples in X.
