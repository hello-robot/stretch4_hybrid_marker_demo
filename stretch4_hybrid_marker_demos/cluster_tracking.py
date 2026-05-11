import numpy as np
from scipy.optimize import linear_sum_assignment

class TrackedCluster:
    """
    Represents a persistently tracked cluster over time with an Alpha-Beta filter.
    """
    def __init__(self, track_id, initial_features, alpha=0.5, beta=0.1):
        self.track_id = track_id
        
        # features array: [x, y, z, radius, num_points]
        self.features = np.array(initial_features, dtype=np.float32)
        
        # velocity array: [vx, vy, vz]
        self.velocity = np.zeros(3, dtype=np.float32)
        
        # Track management
        self.staleness = 0
        
        # Filter parameters
        self.alpha = alpha
        self.beta = beta
        
        # Categorization Scores (Robot, Environment, Agent)
        # Uniform uncertainty initially
        self.scores = {'robot': 0.333, 'env': 0.333, 'agent': 0.334}

    def predict(self, dt):
        """
        Predict the next state of the cluster based on its velocity.
        """
        if dt > 0:
            # Update expected position [x, y, z] via constant velocity
            self.features[0:3] += self.velocity * dt

    def update(self, measurement_features, dt):
        """
        Update the cluster state using a new measurement observation.
        """
        measurement_features = np.array(measurement_features, dtype=np.float32)
        
        # Calculate residual (difference between prediction and measurement)
        # Position residual
        pos_res = measurement_features[0:3] - self.features[0:3]
        
        # Alpha-Beta Update for position and velocity
        self.features[0:3] = self.features[0:3] + (self.alpha * pos_res)
        
        if dt > 0:
            self.velocity = self.velocity + ((self.beta / dt) * pos_res)
            
        # For non-kinematic features (radius, num_points), we can just blend them
        # using a simple exponential moving average (using alpha or a smaller factor)
        self.features[3:] = self.features[3:] + (self.alpha * (measurement_features[3:] - self.features[3:]))
        
        # Reset staleness since we saw it
        self.staleness = 0

    def update_scores(self, l_robot, l_env, l_agent, smooth_alpha=0.2):
        """
        Update categorical confidence scores using a low-pass filter and normalize.
        """
        self.scores['robot'] = (1.0 - smooth_alpha) * self.scores['robot'] + smooth_alpha * l_robot
        self.scores['env'] = (1.0 - smooth_alpha) * self.scores['env'] + smooth_alpha * l_env
        self.scores['agent'] = (1.0 - smooth_alpha) * self.scores['agent'] + smooth_alpha * l_agent
        
        # Normalize
        total = sum(self.scores.values())
        if total > 0:
            for k in self.scores:
                self.scores[k] /= total
        else:
            self.scores = {'robot': 0.333, 'env': 0.333, 'agent': 0.334}

class ClusterTrackerAlphaBetaHungarian:
    """
    Modular tracker that uses an Alpha-Beta predictive filter and the
    Hungarian algorithm (Munkres) for global optimal assignment.
    """
    def __init__(self, 
                 weights_pos=1.0, 
                 weights_rad=0.0, 
                 weights_pts=0.01,
                 max_match_distance=1.0, 
                 max_staleness=3,
                 alpha=0.5,
                 beta=0.1):
        self.tracks = []
        self.next_track_id = 1
        
        # Tuning parameters
        self.weights_pos = weights_pos
        self.weights_rad = weights_rad
        self.weights_pts = weights_pts
        self.max_match_distance = max_match_distance
        self.max_staleness = max_staleness
        
        self.alpha = alpha
        self.beta = beta

    def _cluster_to_features(self, cluster):
        # [x, y, z, radius, num_points]
        return [
            float(cluster['centroid'][0]),
            float(cluster['centroid'][1]),
            float(cluster['centroid'][2]),
            float(cluster['radius']),
            float(len(cluster['points']))
        ]

    def _calculate_distance(self, track_features, cluster_features):
        tf = np.array(track_features)
        cf = np.array(cluster_features)
        
        # Distance components
        pos_dist = np.linalg.norm(tf[0:3] - cf[0:3])
        rad_dist = abs(tf[3] - cf[3])
        pts_dist = abs(tf[4] - cf[4])
        
        total_dist = (self.weights_pos * pos_dist) + \
                     (self.weights_rad * rad_dist) + \
                     (self.weights_pts * pts_dist)
                     
        return total_dist

    def update(self, detected_clusters, dt):
        """
        Updates internal tracks given new detected clusters and time delta.
        Returns a list of paired internal TrackedCluster objects representing
        the confirmed active tracks for this frame.
        """
        # 1. Predict new positions for all existing tracks
        for track in self.tracks:
            track.predict(dt)
            track.staleness += 1 # Assume missed until proven otherwise

        if not detected_clusters:
            # Drop very stale tracks
            self.tracks = [t for t in self.tracks if t.staleness <= self.max_staleness]
            return []

        # Convert list of clusters to feature vectors for matching
        det_features = [self._cluster_to_features(c) for c in detected_clusters]

        # 2. Build Cost Matrix if we have existing tracks
        # Rows = Tracks, Cols = Detections
        if self.tracks and det_features:
            n_tracks = len(self.tracks)
            n_dets = len(det_features)
            cost_matrix = np.zeros((n_tracks, n_dets))
            
            for i, track in enumerate(self.tracks):
                for j, feat in enumerate(det_features):
                    cost_matrix[i, j] = self._calculate_distance(track.features, feat)
                    
            # 3. Solve Global Optimal Assignment (Hungarian Alg)
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            
            # 4. Apply validations and update
            matched_det_indices = set()
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] <= self.max_match_distance:
                    self.tracks[r].update(det_features[c], dt)
                    
                    # Attach the raw cluster data into the track for downstream use
                    self.tracks[r].latest_cluster = detected_clusters[c] 
                    matched_det_indices.add(c)
                # Else: Rejects match (distance too far). Track remains stale, Detection becomes new.
        else:
            matched_det_indices = set()
            
        # 5. Spawn new tracks for unmatched detections
        for j, feat in enumerate(det_features):
            if j not in matched_det_indices:
                new_track = TrackedCluster(self.next_track_id, feat, self.alpha, self.beta)
                new_track.latest_cluster = detected_clusters[j]
                self.tracks.append(new_track)
                self.next_track_id += 1
                
        # 6. Clean up dead tracks
        self.tracks = [t for t in self.tracks if t.staleness <= self.max_staleness]
        
        # Return all tracks that were successfully updated (or just created) this frame
        active_returns = [t for t in self.tracks if t.staleness == 0]
        return active_returns

# Map of available trackers for easy switching
AVAILABLE_TRACKERS = {
    'alpha_beta_hungarian': ClusterTrackerAlphaBetaHungarian
}
