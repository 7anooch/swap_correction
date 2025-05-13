import numpy as np
import math


class KalmanFilter(object):
    '''
    A non-general Kalman filter designed to operate on trajectory data
    under the assumption of a Newtonian model with N derivatives
    Assumes no control inputs
    TODO: make a generalized version (there are probably better ones, but they're poorly-documented)
    NOTE: in comments, m = n-1 (x_m is more readable than x_(n-1))
    '''
    def __init__(self, dt : float, ndim : int = 2, derivatives : int = 2,
                  estimateCov : float = 1000, measurementCov : float = 1000, processCov : float = 1000):
        """
        dt: sampling time (time for 1 cycle)
        derivatives: number of derivatives to use
        estimateCov: magnitude of estimate covariance P
        measurementCov: magnitude of measurement covariance R
        processCov: magnitude of process covariance Q

        xstd: standard deviations of position measurements
        w: process noise vector
        """
        self.dt = dt
        self.dim = ndim # number of dimensions (should be 2, but ...)
        self.npts = self.dim * (1 + derivatives) # number of values to use in estimates

        # Intial estimate
        self.x = np.zeros((self.npts,))

        # generate transition matrix (organized as [x,y, vx,vy, ...])
        self.F = np.eye(self.npts)
        for i in range(self.npts):
            order = i//self.dim # derivative order
            for j in range(derivatives-order):
                k = j + 1
                self.F[i,i + k*self.dim] = dt**k / math.factorial(k)

        # initialize filter
        self.H = np.eye(self.dim,self.npts) # observation matrix (maps state vector -> measurement vector)
        self.P = np.eye(self.npts) * estimateCov # P is initialized with np.eye()
        self.Q = np.eye(self.npts) * processCov
        self.R = np.eye(self.dim) * measurementCov


    def predict(self) -> np.ndarray:
        '''Perform Time Updata / Prediction step'''
        # Extrapolate State
        # x_n = Fx_m + Gu_m
        self.x = np.dot(self.F, self.x)# + np.dot(self.G, self.u)

        # Extrapolate Uncertainty
        # P_n = A*P_m*A' + Q 
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.x[:self.dim]
    

    def update(self, z : list[float]) -> np.ndarray:
        '''Perform Measurement Update / Correction step'''
        # Compute Kalman Gain
        # S = H * P * H' + R
        # K = P * H' * inv(H*P*H'+R)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update estimate with measurement
        # x_n = x_m + K(z_n - Hx_m)
        self.x += np.dot(K, (z - np.dot(self.H, self.x))) # removed rounding

        # Update error covariance matrix
        # P_n = (1-K*H)P_m
        # NOTE: simplified computation -- numerically unstable, but faster
        # can generate nonsymmetric matrix due to floating-point errors
        # Full version: P_n = (1-K*H)P_m(1-K*H)' + K*R*K')
        I = np.eye(self.H.shape[1])
        self.P = (I - (K @ self.H)) @ self.P

        return self.x[:self.dim]
    

    def filter(self, data : np.ndarray, xo : list[float] | None = None) -> np.ndarray:
        '''
        Run the filter on the given data set
        '''
        # set initial estimate to position of first point
        self.x = np.zeros((self.npts,))
        self.x[:self.dim] = data[0,:] if xo is None else xo

        filtData = data.copy()
        for i in range(data.shape[0]):
            # get next data point
            z = data[i,:]

            # skip over NaN values
            # these (to our knowledge) only occur at overlaps,
            # so position data should not change much
            # TODO: research more robust methods for dealing with gaps
            if sum(np.isnan(z)) == 0:
                self.predict()
                filtData[i,:] = self.update(z)
        return filtData
                