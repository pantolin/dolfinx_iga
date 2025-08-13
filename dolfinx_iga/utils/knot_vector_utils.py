"""
Knot vector utilities for B-spline and NURBS computations.

This module provides functions for generating, validating, and manipulating
knot vectors used in isogeometric analysis.
"""

import numpy as np


def generate_uniform_knot_vector(n_control_points: int, degree: int) -> np.ndarray:
    """
    Generate a uniform knot vector.
    
    Parameters
    ----------
    n_control_points : int
        Number of control points
    degree : int
        Degree of the B-spline
        
    Returns
    -------
    np.ndarray
        Uniform knot vector
    """
    n_knots = n_control_points + degree + 1
    
    # Create uniform knot vector with clamped ends
    if n_knots < 2 * (degree + 1):
        raise ValueError("Not enough knots for the given degree and control points")
    
    # Number of internal knots
    n_internal = n_knots - 2 * (degree + 1)
    
    knots = np.zeros(n_knots)
    
    # Set clamped ends
    knots[:degree + 1] = 0.0
    knots[-(degree + 1):] = 1.0
    
    # Set internal knots uniformly
    if n_internal > 0:
        internal_knots = np.linspace(0, 1, n_internal + 2)[1:-1]
        knots[degree + 1:degree + 1 + n_internal] = internal_knots
    
    return knots


def generate_open_knot_vector(n_control_points: int, degree: int) -> np.ndarray:
    """
    Generate an open (clamped) knot vector.
    
    Parameters
    ----------
    n_control_points : int
        Number of control points
    degree : int
        Degree of the B-spline
        
    Returns
    -------
    np.ndarray
        Open knot vector
    """
    return generate_uniform_knot_vector(n_control_points, degree)


def generate_periodic_knot_vector(n_control_points: int, degree: int) -> np.ndarray:
    """
    Generate a periodic knot vector for closed curves.
    
    Parameters
    ----------
    n_control_points : int
        Number of control points
    degree : int
        Degree of the B-spline
        
    Returns
    -------
    np.ndarray
        Periodic knot vector
    """
    n_knots = n_control_points + degree + 1
    knots = np.linspace(0, 1, n_knots)
    return knots


def validate_knot_vector(
    knot_vector: np.ndarray, 
    n_control_points: int, 
    degree: int
) -> None:
    """
    Validate a knot vector for consistency.
    
    Parameters
    ----------
    knot_vector : np.ndarray
        Knot vector to validate
    n_control_points : int
        Number of control points
    degree : int
        Degree of the B-spline
        
    Raises
    ------
    ValueError
        If the knot vector is invalid
    """
    expected_length = n_control_points + degree + 1
    
    if len(knot_vector) != expected_length:
        raise ValueError(
            f"Knot vector length {len(knot_vector)} does not match expected "
            f"length {expected_length} for {n_control_points} control points "
            f"and degree {degree}"
        )
    
    # Check if knot vector is non-decreasing
    if not np.all(np.diff(knot_vector) >= 0):
        raise ValueError("Knot vector must be non-decreasing")
    
    # Check for sufficient range
    if knot_vector[0] == knot_vector[-1]:
        raise ValueError("Knot vector must have non-zero range")


def normalize_knot_vector(knot_vector: np.ndarray) -> np.ndarray:
    """
    Normalize a knot vector to the range [0, 1].
    
    Parameters
    ----------
    knot_vector : np.ndarray
        Input knot vector
        
    Returns
    -------
    np.ndarray
        Normalized knot vector
    """
    knot_vector = np.asarray(knot_vector)
    min_val = knot_vector[0]
    max_val = knot_vector[-1]
    
    if max_val == min_val:
        raise ValueError("Cannot normalize constant knot vector")
    
    return (knot_vector - min_val) / (max_val - min_val)


def insert_knot(
    knot_vector: np.ndarray,
    control_points: np.ndarray,
    degree: int,
    u: float,
    multiplicity: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert a knot into the knot vector and update control points.
    
    Parameters
    ----------
    knot_vector : np.ndarray
        Original knot vector
    control_points : np.ndarray
        Original control points
    degree : int
        Degree of the B-spline
    u : float
        Parameter value of the knot to insert
    multiplicity : int
        Number of times to insert the knot (default: 1)
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        New knot vector and updated control points
    """
    # This is a simplified implementation
    # A full implementation would use the knot insertion algorithm
    raise NotImplementedError("Knot insertion not yet implemented")


def remove_knot(
    knot_vector: np.ndarray,
    control_points: np.ndarray, 
    degree: int,
    u: float,
    tolerance: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Remove a knot from the knot vector.
    
    Parameters
    ----------
    knot_vector : np.ndarray
        Original knot vector
    control_points : np.ndarray
        Original control points
    degree : int
        Degree of the B-spline
    u : float
        Parameter value of the knot to remove
    tolerance : float
        Tolerance for knot removal (default: 1e-10)
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, bool]
        New knot vector, updated control points, and success flag
    """
    # This is a simplified implementation
    # A full implementation would use the knot removal algorithm
    raise NotImplementedError("Knot removal not yet implemented")


def degree_elevate(
    knot_vector: np.ndarray,
    control_points: np.ndarray,
    degree: int,
    elevate_by: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Elevate the degree of a B-spline.
    
    Parameters
    ----------
    knot_vector : np.ndarray
        Original knot vector
    control_points : np.ndarray
        Original control points
    degree : int
        Original degree
    elevate_by : int
        How much to elevate the degree (default: 1)
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        New knot vector and control points for elevated degree
    """
    # This is a simplified implementation
    # A full implementation would use degree elevation algorithms
    raise NotImplementedError("Degree elevation not yet implemented")


def find_knot_multiplicity(knot_vector: np.ndarray, u: float, tolerance: float = 1e-12) -> int:
    """
    Find the multiplicity of a knot value in the knot vector.
    
    Parameters
    ----------
    knot_vector : np.ndarray
        Knot vector
    u : float
        Knot value to find multiplicity for
    tolerance : float
        Tolerance for comparing knot values (default: 1e-12)
        
    Returns
    -------
    int
        Multiplicity of the knot
    """
    return np.sum(np.abs(knot_vector - u) < tolerance)
