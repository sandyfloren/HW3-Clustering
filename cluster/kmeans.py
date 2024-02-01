import numpy as np
#from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # Type checking
        if not isinstance(k, int):
            raise TypeError('k must be of type int.')
        if not isinstance(tol, float):
            raise TypeError('tol must be of type float.')
        if not isinstance(max_iter, int):
            raise TypeError('max_iter must be of type int.')
        
        # Value checking
        if not k > 0:
            raise ValueError('k must be at least 1.')
        if not tol >= 0:
            raise ValueError('tol must be at least 0.')
        if not max_iter > 0:
            raise ValueError('max_iter must be at least 1.')
        
        # Initialize class attributes
        self.seed = None
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.mat = None
        self.n = None
        self.d = None
        self.centroids = None
        self.classes = None
        self.error = np.inf
        self.converged = False
        self.fitted = False

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        self.mat = mat
        shape = mat.shape
        
        # Handle 1-dimensional mat
        if len(mat.shape) == 1:
            self.n = shape[0]
            self.d = 1
        else:
            self.n, self.d = mat.shape

        # Value checking
        if not self.k < self.n:
            raise ValueError('number of labels (k) must be less than number of observations.')

        # Initialize class memberships matrix
        self.classes = np.zeros(shape=(self.n, self.k))

        # Initialize centroids
        self._init_centroids()

        i = 0
        # If maximum iterations are reached, or if error is below tolerance threshold, stop fitting.
        while i < self.max_iter and not self.converged:
          
            self._update_classes()
            self._update_centroids()
            i += 1

        self.fitted = True

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # Check if model has been fit
        if not self.fitted:
            raise Exception('KMeans must be fit to data before prediction.')

        n = mat.shape[0]

        d = 1
        if len(mat.shape) != 1:
            d = mat.shape[1]
        
        # Check if dimension of prediction data is the same as training data
        if not d == self.d:
            raise ValueError('dimension of data does not match data used for model fitting.')
        
        # Initialize predicted class memberships matrix
        pred_classes = np.zeros(shape=(n))

        # Iterate through data points
        for i in range(n):
            
            # Find centroid with minimal squared distance to current data point (mat[i])
            if d == 1:
                r = np.argmin((self.centroids - mat[i])**2)
            else:                 
                r = np.argmin(np.sum((self.centroids - mat[i])**2, axis=1))
            pred_classes[i] = r

        return pred_classes

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x d` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids
    
    def _init_centroids(self):
        """
        Initialize the centroid locations of the model. 
        """
        # Random initialization
        rng = np.random.default_rng(self.seed)
        self.centroids = self.mat[rng.choice(self.mat.shape[0], size=self.k, replace=False)]   

    def _update_classes(self):
        """
        Update class memberships matrix.
        """
        
        # Iterate through data points
        for i in range(self.n):
            squared_dist = (self.centroids - self.mat[i])**2
            # Find centroid with minimal squared distance to current data point (self.mat[i])
            if self.d == 1: 
                r = np.argmin(squared_dist)
            else:                 
                r = np.argmin(np.sum(squared_dist, axis=1))
            # Assign one-hot encoded class to ith row of class memberships matrix
            self.classes[i] = np.zeros(self.k)
            self.classes[i][r] = 1

            # Update error
            prev_error = self.error
            self.error = np.sum(squared_dist)

            # Check tolerance
            if np.abs(prev_error - self.error) <= self.tol:
                self.converged = True


    def _update_centroids(self):
        """
        Update centroids matrix.
        """
        # Normalize classes 
        norm = np.linalg.norm(self.classes, ord=1, axis=0, keepdims=True)
        class_weights = np.divide(self.classes, norm, out=np.zeros_like(self.classes), where=norm!=0) # handle divide by 0 if no predicted class members   
        # Compute new means
        prev_centroids = self.centroids.copy()
        self.centroids = class_weights.T @ self.mat

