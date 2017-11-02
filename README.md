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






Support Vector Classification.
The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.
The multiclass support is handled according to a one-vs-one scheme.
Parameters:	
C : float, optional (default=1.0)
Penalty parameter C of the error term.
kernel : string, optional (default=’rbf’)
Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
degree : int, optional (default=3)
Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
gamma : float, optional (default=’auto’)
Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
coef0 : float, optional (default=0.0)
Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
probability : boolean, optional (default=False)
Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
shrinking : boolean, optional (default=True)
Whether to use the shrinking heuristic.
tol : float, optional (default=1e-3)
Tolerance for stopping criterion.
cache_size : float, optional
Specify the size of the kernel cache (in MB).
class_weight : {dict, ‘balanced’}, optional
Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
verbose : bool, default: False
Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
max_iter : int, optional (default=-1)
Hard limit on iterations within solver, or -1 for no limit.
decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’
Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
decision_function_shape is ‘ovr’ by default.
random_state : int, RandomState instance or None, optional (default=None)
The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.


5.predict(X)
Perform classification on samples in X.
For an one-class model, +1 or -1 is returned.
Parameters:	
X : {array-like, sparse matrix}, shape (n_samples, n_features)
For kernel=”precomputed”, the expected shape of X is [n_samples_test, n_samples_train]
Returns:	
y_pred : array, shape (n_samples,)
Class labels for samples in X.
