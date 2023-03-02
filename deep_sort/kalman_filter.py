
import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of freedom (contains values for N = 1, ..., 9). 
Taken from MATLAB/Octave's chi2inv function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    
    The 8-dimensional state space vector is defined as follow:

        [ x, y, a, h, vx, vy, va, vh ]

    It contains the bounding box center position (x, y), aspect ratio a, height h, and their respective velocities.
    Object motion follows a constant velocity model. 
    The bounding box location (x, y, a, h) is taken as direct observation of the state space (linear observation model).

    """

    def __init__(self):

        ndim, dt = 4, 1.

        # Define state transition matrix:
        self.F = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = dt

        # Define observation matrix:
        self.H = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current state estimate. 
        # These weights control the amount of uncertainty in the model. 
        # This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y), aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            State's mean vector (8 dimensional vector) and covariance matrix (8x8 dimensional matrix) of the new track. 
            Unobserved velocities are initialized to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[0],   # the center point x
            2 * self._std_weight_position * measurement[1],   # the center point y
            1 * measurement[2],                               # the ratio of width/height
            2 * self._std_weight_position * measurement[3],   # the height
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))

        return mean, covariance

    def predict(self, x, P):
        """
        Run Kalman Filter prediction step.

        Parameters
        ----------
        x : ndarray
            State's mean vector at the previous time step (8 dimensional vector).
        P : ndarray
            State covariance matrix at the previous time step (8x8 dimensional matrix).

        Returns
        -------
        (ndarray, ndarray)
            State's mean vector and covariance matrix of the predicted state. 
            Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * x[0],
            self._std_weight_position * x[1],
            1 * x[2],
            self._std_weight_position * x[3]]

        std_vel = [
            self._std_weight_velocity * x[0],
            self._std_weight_velocity * x[1],
            0.1 * x[2],
            self._std_weight_velocity * x[3]]

        # Compute process noise covariance matrix:
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        # Compute state vector and covariance matrix:
        x = np.dot(self.F, x)
        P = np.linalg.multi_dot((self.F, P, self.F.T)) + Q

        return x, P

    def project(self, x, P):
        """
        Project state distribution to measurement space.

        Parameters
        ----------
        x : ndarray
            State's mean vector (8 dimensional vector).
        P : ndarray
            State's covariance matrix (8x8 dimensional matrix).

        Returns
        -------
        (ndarray, ndarray)
            Projected state's mean vector and covariance matrix of the given state estimate.

        """
        std = [
            self._std_weight_position * x[0],
            self._std_weight_position * x[1],
            0.1 * x[2],
            self._std_weight_position * x[3]]

        # Compute measurement noise covariance matrix:
        R = np.diag(np.square(std))

        # Compute state vector and covariance matrix:
        x = np.dot(self.H, x)
        P = np.linalg.multi_dot((self.H, P, self.H.T)) + R

        return x, P

    def update(self, x, P, y):
        """
        Run Kalman filter correction step.

        Parameters
        ----------
        x : ndarray
            Predicted state's mean vector (8 dimensional vector).
        P : ndarray
            State's covariance matrix (8x8 dimensional matrix).
        y : ndarray
            Measurement vector (x, y, a, h), where (x, y) is the center position, a the aspect ratio, 
            and h the height of the bounding box (4 dimensional vector)

        Returns
        -------
        (ndarray, ndarray)
            Measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(x, P)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower = True, check_finite = False)

        # Compute Kalman gain:
        K = scipy.linalg.cho_solve((chol_factor, lower), np.dot(P, self.H.T).T, check_finite = False).T

        # Compute the innovation term:
        z = y - projected_mean

        # Compute new state vector and new covariance matrix:
        x_new = x + np.dot(z, K.T)
        P_new = P - np.linalg.multi_dot((K, projected_cov, K.T))

        return x_new, P_new

    def gating_distance(self, mean, covariance, measurements, only_position = False):
        """
        Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. 
        If `only_position` is False, the chi-square distribution has 4 degrees of freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the squared Mahalanobis distance 
            between (mean, covariance) and `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower = True, check_finite = False, overwrite_b = True)
        squared_maha = np.sum(z * z, axis=0)

        return squared_maha