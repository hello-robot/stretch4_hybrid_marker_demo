import numpy as np
from scipy.spatial import cKDTree
import networkx as nx

def cluster_points_kdtree(points, connectivity_distance, min_points, max_radius):
    """
    Clusters sparse high-intensity 3D points using KD-Tree Radius Search.
    Filters clusters based on min_points and max_radius constraints.
    Returns a list of dicts containing centroid, radius, point array, and indices.
    """
    if len(points) == 0:
        return []
        
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=connectivity_distance, p=2.0)
    
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    G.add_edges_from(pairs)
    
    clusters = []
    
    for cc in nx.connected_components(G):
        if len(cc) < min_points:
            continue
            
        cluster_indices = list(cc)
        cluster_points = points[cluster_indices]
        
        centroid = np.mean(cluster_points, axis=0)
        
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        radius = np.max(distances)
        
        if radius > max_radius:
            continue
            
        clusters.append({
            'centroid': centroid,
            'radius': radius,
            'points': cluster_points,
            'indices': cluster_indices
        })
        
    return clusters

def cluster_points_iterative_sphere(points, max_sphere_radius, min_points, min_density):
    """
    Clusters sparse high-intensity 3D points using Iterative Seed-Based Spherical Clustering.
    (Method 1 from plan: Mean Shift Variant).
    
    Note: min_density is points per cubic meter.
    Returns a list of dicts containing centroid, radius, point array, and indices.
    """
    if len(points) == 0:
        return []
        
    clusters = []
    unclustered_indices = set(range(len(points)))
    
    # Optional parameters for convergence
    max_iterations = 20
    convergence_threshold = 0.01 # 1 cm
    
    while unclustered_indices:
        # 1. Select an unclustered point as a seed
        seed_index = next(iter(unclustered_indices))
        current_center = points[seed_index]
        
        for _ in range(max_iterations):
            # 2. Find all points within this sphere (from all points, not just unclustered)
            distances = np.linalg.norm(points - current_center, axis=1)
            in_sphere_indices = np.where(distances <= max_sphere_radius)[0]
            
            if len(in_sphere_indices) == 0:
                # Should not happen since seed is always included, but theoretically
                break
                
            in_sphere_points = points[in_sphere_indices]
            
            # 3. Compute centroid
            new_center = np.mean(in_sphere_points, axis=0)
            
            # Check convergence
            if np.linalg.norm(new_center - current_center) < convergence_threshold:
                current_center = new_center
                break
                
            current_center = new_center
            
        # Convergence reached for this sphere.
        
        # Re-fetch points within final converged sphere to get final set
        final_distances = np.linalg.norm(points - current_center, axis=1)
        final_indices_array = np.where(final_distances <= max_sphere_radius)[0]
        final_indices = set(final_indices_array.tolist())
        
        # Check constraints
        valid_cluster = True
        
        if len(final_indices) < min_points:
            valid_cluster = False
            
        # Density check
        sphere_volume = (4.0/3.0) * np.pi * (max_sphere_radius ** 3)
        density = len(final_indices) / sphere_volume if sphere_volume > 0 else 0
        
        if density < min_density:
            valid_cluster = False
            
        if valid_cluster:
            cluster_points = points[final_indices_array]
            
            # Calculate actual radius (distance to furthest point in sphere)
            actual_radius = np.max(np.linalg.norm(cluster_points - current_center, axis=1)) if len(cluster_points) > 0 else 0.0
            
            clusters.append({
                'centroid': current_center,
                'radius': actual_radius, # Note: this will be <= max_sphere_radius
                'points': cluster_points,
                'indices': final_indices_array.tolist()
            })
            
        # 5. Hard Membership: Remove the points in the convergence sphere from unclustered pool
        # We remove them even if not valid cluster, to avoid infinite loop on sparse noise
        unclustered_indices -= final_indices
        unclustered_indices.discard(seed_index)
        
    return clusters
