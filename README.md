# SVM_python_code
The repository contains the python code to implement Support Vector Machine algorithm.

The input file mydata.txt contains the dataset as follows:
For age: Youth=1,Middle=2,Senior=3 For income: Low=1,Medium=2,High=3 For student: Yes=1,No=2
For credit rating: Fair=1,Excellent=2 For buys computer: Yes=1,No=2

After classification the model would predict whether a person buys a computer based upon attributes age,income,student and credit rating.

DATASET:

age   	income	   student	   credit_rating   buys_computer
1  	   3      	   2     	   1	             2
1	      3         	2	         2	             2
2	      3	         2	         1	             1
3	      2	         2        	1	             1
3	      1	         1	         1	             1
3	      1     	   1         	2	             2 
2	      1	         1     	   2	             1
1	      2	         2     	   1	             2
1	      1	         1	         1	             1
3	      2	         1	         1	             1
1	      2	         1      	   2	             1
2      	2	         2	         2	             1
2	      3	         1	         1	             1
3	      2	         2	         2	             2
 
First we import the required libraries:

1.pandas:This is used to read the dataset from the file.

2.train_test_split from sklearn.model_selection: Both the training and testing dataset is taken from the input file,so it needs to be                                                    split into two parts training dataset and testing dataset.This is the purpose of                                                        train_test_split which splits the dataset into training and testing datasets in the                                                      ratio 3:1(approx).

3.svm from sklearn: To implement Support Vector Classsification(SVC).

Read the data in variable comps.

Print the original dataset.

Store the attributes/features in tuple X.

Store the class label in y.

Print X,print y.

Use the functions:

1.train_test_split(*arrays, ** options)
Split arrays or matrices into random train and test subsets
Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting (and optionally subsampling) data in a oneliner.

Parameters:	

*arrays : sequence of indexables with same length / shape[0]
Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
test_size : float, int, None, optional
If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. If None, the value is set to the complement of the train size. By default, the value is set to 0.25. The default will change in version 0.21. It will remain 0.25 only if train_size is unspecified, otherwise it will complement the specified train_size.
train_size : float, int, or None, default None
If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples. If None, the value is automatically set to the complement of the test size.
random_state : int, RandomState instance or None, optional (default=None)
If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
shuffle : boolean, optional (default=True)
Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
stratify : array-like or None (default is None)
If not None, data is split in a stratified fashion, using this as the class labels.
Returns:	
splitting : list, length=2 * len(arrays)
List containing train-test split of inputs.

Print X_train,y_train,X_test and y_test

2. svm.SVC

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

3. fit:

   Fits the SVM model according to the given training data.
   Parameters:	
X : array-like, dtype=float64, size=[n_samples, n_features]
Y : array, dtype=float64, size=[n_samples]
target vector
svm_type : {0, 1, 2, 3, 4}, optional
Type of SVM: C_SVC, NuSVC, OneClassSVM, EpsilonSVR or NuSVR respectively. 0 by default.
kernel : {‘linear’, ‘rbf’, ‘poly’, ‘sigmoid’, ‘precomputed’}, optional
Kernel to use in the model: linear, polynomial, RBF, sigmoid or precomputed. ‘rbf’ by default.
degree : int32, optional
Degree of the polynomial kernel (only relevant if kernel is set to polynomial), 3 by default.
gamma : float64, optional
Gamma parameter in rbf, poly and sigmoid kernels. Ignored by other kernels. 0.1 by default.
coef0 : float64, optional
Independent parameter in poly/sigmoid kernel. 0 by default.
tol : float64, optional
Numeric stopping criterion (WRITEME). 1e-3 by default.
C : float64, optional
C parameter in C-Support Vector Classification. 1 by default.
nu : float64, optional
0.5 by default.
epsilon : double, optional
0.1 by default.
class_weight : array, dtype float64, shape (n_classes,), optional
np.empty(0) by default.
sample_weight : array, dtype float64, shape (n_samples,), optional
np.empty(0) by default.
shrinking : int, optional
1 by default.
probability : int, optional
0 by default.
cache_size : float64, optional
Cache size for gram matrix columns (in megabytes). 100 by default.
max_iter : int (-1 for no limit), optional.
Stop solver after this many iterations regardless of accuracy. -1 by default.
random_seed : int, optional
Seed for the random number generator used for probability estimates. 0 by default.
Returns:	
support : array, shape=[n_support]
index of support vectors
support_vectors : array, shape=[n_support, n_features]
support vectors (equivalent to X[support]). Will return an empty array in the case of precomputed kernel.
n_class_SV : array
number of support vectors in each class.
sv_coef : array
coefficients of support vectors in decision function.
intercept : array
intercept in decision function
probA, probB : array
probability estimates, empty array for probability=False

4.score(X, y, sample_weight=None)
Returns the mean accuracy on the given test data and labels.
In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.
Parameters:	
X : array-like, shape = (n_samples, n_features)
Test samples.
y : array-like, shape = (n_samples) or (n_samples, n_outputs)
True labels for X.
sample_weight : array-like, shape = [n_samples], optional
Sample weights.
Returns:	
score : float
Mean accuracy of self.predict(X) wrt. y.

5.predict(X)
Perform classification on samples in X.
For an one-class model, +1 or -1 is returned.
Parameters:	
X : {array-like, sparse matrix}, shape (n_samples, n_features)
For kernel=”precomputed”, the expected shape of X is [n_samples_test, n_samples_train]
Returns:	
y_pred : array, shape (n_samples,)
Class labels for samples in X.

 Used different kernels,i.e.,kernel='linear',kernel='poly' and kernel='rbf' and tested the accuracy in each case, found that the accuracy for kernel='linear' and kernel='poly' was 0.5 and for kernel='rbf' it was 0.75.
