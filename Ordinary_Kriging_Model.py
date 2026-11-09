import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


class ExactAcousticKriging:
    """
    Ordinary Kriging model that uses an exponential variogram and assumes 100% accurate phone readings (nugget = 0.0).
    The practical range is automatically adjusted based on the inputted sound map geometry to ensure that the correlation between points is meaningful.
    This model assumes that the inserted sound map is an outside in an open space with no obstacles or reflections, and that the sound field is smooth and continuous.
    Calculation is performed in the linear energy domain to avoid bias from dB averaging.
    """
    def __init__(self, practical_range=None, sill=1.0, range_ratio=0.25, safety_margin_db=0.0, peak_boost=False):
        """
        Inputs:
            sill: the variance between points (in linear energy units)
            
            practical_range: the range at which the correlation drops to ~5% of the sill (in meters). 
                                Will be automatically inferred from the sound map geometry if not provided.
            
            range_ratio: float, fraction of map diagonal to use when practical_range=None (default 0.25)
            
            safety_margin_db: float, maximum dB cushion added in unmeasured dead zones.
                          Scales smoothly with spatial uncertainty (0 dB at phones, 
                          up to safety_margin_db in large gaps).
        """
        self.practical_range = practical_range
        self.sill = sill
        self.range_ratio = range_ratio
        self.safety_margin_db = float(safety_margin_db)
        self.peak_boost = peak_boost

    def _variogram(self, h, eff_range):
        """
        Uses the standard exponential variogram model to compute semi-variance for distances h.
        The variogram is necessary to determine the weighting of known points when predicting unknown locations later on.
        """
        return self.sill * (1.0 - np.exp(-3.0 * h / eff_range))

    def fit_predict(self, phone_positions, phone_levels, query_positions):
        """
        Inputs:
            phone_positions: (N, 2)     #Coordinates of the N known microphones
            phone_levels:    (T, N)     #Sound levels at each time step in dB for all N microphones
            query_positions: (M, 2)     #Coordinates of the M query points where we want to predict sound levels
        Output:
            predicted_levels: (T, M)    #Estimated sound levels at each time step for all M query points in dB
        """
        #Ensure 2D shape (T, N) even if T=1
        if phone_levels.ndim == 1:
            phone_levels = phone_levels[np.newaxis, :]

        #Extract dimensions from input arrays
        N = phone_positions.shape[0]
        M = query_positions.shape[0]

        #Resolve practical range dynamically if not explicitly specified
        if self.practical_range is None:
            #Span covers the bounding box of both the phones and the target query grid
            all_pts = np.vstack([phone_positions, query_positions])
            span = np.ptp(all_pts, axis=0)  #[max_x - min_x, max_y - min_y]
            diagonal = np.sqrt(np.sum(span ** 2))
            eff_range = max(self.range_ratio * diagonal, 1e-6)
        else:
            eff_range = self.practical_range


        #Transform known dB levels to linear energy domain as to avoid bias from dB averaging
        linear_energy = 10.0 ** (phone_levels / 10.0)  # (T, N)

        #Calculate the Euclidean distance matrix between all known phone positions and then apply the variogram function to compute the covariance matrix
        D_known = distance_matrix(phone_positions, phone_positions)
        A = self._variogram(D_known, eff_range)

        #Create an augmented covariance matrix A_augmented of size (N+1, N+1) to include the Lagrange multiplier for the unbiasedness constraint
        A_aug = np.ones((N + 1, N + 1))
        A_aug[:N, :N] = A 
        A_aug[N, N] = 0.0 

        #Add a small value to the diagonal of A_augmented to prevent singularity issues during matrix inversion
        A_aug[:N, :N] += np.eye(N) * 1e-5 

        #Calculate the Euclidean distance matrix between all known phone positions and query positions and then apply the variogram function to compute the covariance matrix
        D_query = distance_matrix(phone_positions, query_positions)
        B = self._variogram(D_query, eff_range)

        #Create an augmented covariance matrix B_augmented of size (N+1, M) to include the Lagrange multiplier for the unbiasedness constraint
        B_aug = np.ones((N + 1, M))
        B_aug[:N, :] = B

        #Solve the linear system A_augmented * W_augmented = B_augmented to find the optimal weights for each query point, including the Lagrange multiplier
        W_aug = np.linalg.solve(A_aug, B_aug)
        weights = W_aug[:N, :]  #Remove the last row corresponding to the Lagrange multiplier, leaving only the weights for the known points

        #Compute the predicted linear energy at each query point by taking the dot product of the known linear energy and the weights
        pred_energy = np.dot(linear_energy, weights)
        pred_energy = np.maximum(pred_energy, 1e-12) #Clamp any negative or near-zero predicted energy values to avoid issues with logarithmic conversion

        #Convert back to dB and return the predicted sound levels at each query point for all time steps
        base_db = 10.0 * np.log10(pred_energy)

        #Apply Peak Headroom Boost if enabled (for point sources)
        if self.peak_boost:
            #Estimate the missing maximum by findeing the max sensor reading and accounting for average phone spacing
            max_measured_db = np.max(phone_levels, axis=1, keepdims=True)  # (T, 1)
            #Estimate the minimum distance from source to closest phone based on density
            mean_spacing = np.sqrt(np.prod(np.ptp(phone_positions, axis=0)) / N) 
            expected_headroom = 20.0 * np.log10(mean_spacing / 1.0)  #Assuming 1 meter reference distance

            #Add the expected headroom to the max measured dB to get the peak boost
            p90 = np.percentile(phone_levels, 90, axis=1, keepdims=True)  # (T, 1)
            peak_weights = np.clip((base_db - p90) / (np.max(base_db, axis=1, keepdims=True) - p90 + 1e-6), 0.0, 1.0)  # (T, M)
            base_db = base_db + (expected_headroom * (peak_weights)**2)

        #Apply hearing-protection safety buffer
        if self.safety_margin_db > 0.0:
            # Kriging estimation variance: sigma^2 = sum(W_aug * B_aug)
            kriging_var = np.sum(W_aug * B_aug, axis=0)
            # Normalized standard deviation: 0.0 at phone coordinates, approaching 1.0 in gaps
            sigma_norm = np.sqrt(np.clip(kriging_var / self.sill, 0.0, 1.0))
            # Broadcast across time steps: (T, M)
            base_db + (self.safety_margin_db * sigma_norm[np.newaxis, :])

        return base_db