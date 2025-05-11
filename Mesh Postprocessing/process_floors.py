import numpy as np
import matplotlib.pyplot as plt
import copy 
from scipy.spatial import Delaunay
import tqdm

# Simple Gaussian filter 
def gaussian_filter1d(input_array, sigma=2, truncate=4.0):
    # Determine filter radius based on sigma and create Gaussian kernel
    radius = int(truncate * sigma + 0.5)
    kernel_size = 2 * radius + 1
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize to ensure sum is 1
    
    # Convolve 
    padded = np.pad(input_array, radius, mode='edge')
    result = np.zeros_like(input_array, dtype=float)
    for i in range(len(input_array)):
        result[i] = np.sum(padded[i:i+kernel_size] * kernel)
    return result

def find_peaks(x, height=None, threshold=None, distance=None):
    """
    Find peaks in a 1D array as local maxima.
    
    Parameters:
    - x: 1D array in which to find peaks
    - height: Minimum height required to be considered a peak
    - threshold: Minimum height difference to neighboring samples
    - distance: Minimum distance between peaks
    
    Returns:
    - Tuple (peaks, properties) where peaks is array of peak indices
    """
    candidates = np.zeros_like(x, dtype=bool)
    for i in range(1, len(x) - 1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            candidates[i] = True
    peak_indices = np.where(candidates)[0]
    
    # Apply minimum height filter
    if height is not None:
        height_mask = x[peak_indices] >= height
        peak_indices = peak_indices[height_mask]
    if threshold is not None and len(peak_indices) > 0:
        threshold_mask = []
        for idx in peak_indices:
            # Find valleys to left and right
            left_idx = idx
            while left_idx > 0 and x[left_idx-1] < x[left_idx]:
                left_idx -= 1
            
            right_idx = idx
            while right_idx < len(x)-1 and x[right_idx+1] < x[right_idx]:
                right_idx += 1
            left_min = np.min(x[left_idx:idx+1])
            right_min = np.min(x[idx:right_idx+1])
            prominence = x[idx] - max(left_min, right_min)
            threshold_mask.append(prominence >= threshold)
        peak_indices = peak_indices[np.array(threshold_mask)]
    
    # Apply distance filter
    if distance is not None and len(peak_indices) > 1:
        # Sort peaks by height
        sorted_idxs = np.argsort(x[peak_indices])[::-1]
        sorted_peaks = peak_indices[sorted_idxs]
        keep = np.ones(len(sorted_peaks), dtype=bool)
        for i, peak in enumerate(sorted_peaks):
            if keep[i]:
                # Suppress peaks within distance
                dist_mask = np.abs(sorted_peaks - peak) <= distance
                dist_mask[i] = False  # Don't suppress self
                keep[dist_mask] = False
        peak_indices = sorted_peaks[keep]
        peak_indices = np.sort(peak_indices)
    
    return peak_indices

def plot_histogram(bins, hist, title):
    plt.plot(bins[:-1], hist, label='Original Histogram')
    plt.xlabel('Y values')
    plt.ylabel('Frequency')
    plt.title('Original Histogram of Y values')
    plt.legend()
    # plt.show()
    plt.savefig(title + ".png")
    plt.clf() 

# Pseudocode for floor detection
def detect_floor(vertices):
    # histogram of y-values with appropriate bin size
    y_values = [v[1] for v in vertices]  # Assuming y is up
    hist, bins = np.histogram(y_values, bins=100)
    plot_histogram(bins, hist, "Original_Histogram")

    # Smoothing the histogram
    smoothed_hist = gaussian_filter1d(hist)
    plot_histogram(bins, smoothed_hist, "Smoothed_Histogram")
    
    # find floor, which is the first peak (as they are sorted)
    peaks = find_peaks(smoothed_hist, height=np.max(smoothed_hist)*0.3)
    return bins[peaks[0]]  # Return the y-value of the first peak
    
def process_floors(vertices, faces, y_value=None):
    # Make all vertices below the y_value have the y_value
    vertices = copy.deepcopy(vertices)
    if y_value is None:
        y_value = detect_floor(vertices)
    vertices[:, 1] = np.maximum(vertices[:, 1], y_value)
    return vertices, faces

def fill_mesh_holes(vertices, faces, colors=None):
    """
    Fill holes in a 3D mesh represented by vertices and face indices.
    
    Parameters:
    - vertices: numpy array of shape (N, 3) containing vertex coordinates
    - faces: numpy array of shape (M, 3) containing vertex indices for triangular faces
    
    Returns:
    - updated_faces: numpy array containing the original faces plus the new filling faces
    """
    edges = []
    for face in faces:
        edges.append((face[0], face[1]))
        edges.append((face[1], face[2]))
        edges.append((face[2], face[0]))

    edge_counts = {}
    for a, b in edges:
        edge = tuple(sorted([a, b]))
        edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    # Boundary edges appear exactly once
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not boundary_edges:
        return faces  
    holes = []
    remaining_edges = set(boundary_edges)
    
    # find all holes, which include all boundary vertices 
    for edge in tqdm.tqdm(list(remaining_edges), desc="Finding holes"):
        if edge not in remaining_edges:
            continue  # Skip edges that have already been processed
        # Start a new hole loop
        current_hole = []  # vertex indices
        remaining_edges.remove(edge)
        start_vertex = edge[0]
        current_vertex = edge[1]
        current_hole.append(start_vertex)
        
        while current_vertex != start_vertex:
            # KEY: add the current vertex of the hole to the list
            current_hole.append(current_vertex)
            # Find the next edge in the boundary loop
            next_edge = None
            for potential_edge in list(remaining_edges):
                if current_vertex in potential_edge:
                    next_edge = potential_edge
                    remaining_edges.remove(potential_edge)
                    # Get the other vertex of the edge
                    current_vertex = next_edge[0] if next_edge[1] == current_vertex else next_edge[1]
                    break
            
            if next_edge is None:
                # Failed to close the loop, which means there's an issue with the mesh
                break
        if len(current_hole) >= 3:  
            holes.append(current_hole)
    # Fill each hole
    new_faces = []
    for hole in tqdm.tqdm(holes):
        # Get coordinates of hole vertices
        hole_vertices = np.array([vertices[i] for i in hole])
        centroid = np.mean(hole_vertices, axis=0)
        u, s, vh = np.linalg.svd(hole_vertices - centroid)
        
        # Project vertices onto the best-fit plane
        # In a floor, the first two singular vectors define the plane parallel to the floor (xz-plane)
        u1 = vh[0] 
        u2 = vh[1] 
        
        # Project hole vertices onto the plane
        projected_2d = np.array([(np.dot(v - centroid, u1), np.dot(v - centroid, u2)) for v in hole_vertices])
        
        # Triangulate the projected 2D polygon
        try:
            tri = Delaunay(projected_2d) # each simplex is an index of the vertices 
            for simplex in tri.simplices:
                new_faces.append([hole[simplex[0]], hole[simplex[1]], hole[simplex[2]]])
        except:
            # Fall back to a simpler fan triangulation if Delaunay fails
            for i in range(1, len(hole) - 1):
                new_faces.append([hole[0], hole[i], hole[i + 1]])
    updated_faces = np.vstack([faces, np.array(new_faces)])
    return updated_faces
