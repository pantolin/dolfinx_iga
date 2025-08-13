"""
B-spline basis function implementations.

This module provides efficient computation of B-spline basis functions
and their derivatives using the Cox-de Boor recursion formula.
"""

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None


def find_span(u: float, degree: int, knot_vector: np.ndarray) -> int:
    """
    Find the knot span index for a given parameter value.
    
    Uses binary search to find the span [u_i, u_{i+1}) containing u.
    
    Parameters
    ----------
    u : float
        Parameter value
    degree : int
        Degree of the B-spline
    knot_vector : np.ndarray
        Knot vector
        
    Returns
    -------
    int
        Knot span index
    """
    n = len(knot_vector) - degree - 1
    
    # Special cases
    if u >= knot_vector[n]:
        return n - 1
    if u <= knot_vector[degree]:
        return degree
    
    # Binary search
    low = degree
    high = n
    mid = (low + high) // 2
    
    while u < knot_vector[mid] or u >= knot_vector[mid + 1]:
        if u < knot_vector[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    
    return mid


def bspline_basis(
    u: float, 
    degree: int, 
    knot_vector: np.ndarray, 
    n_basis: int
) -> np.ndarray:
    """
    Compute B-spline basis functions using Cox-de Boor recursion.
    
    Parameters
    ----------
    u : float
        Parameter value where to evaluate basis functions
    degree : int
        Degree of the B-spline
    knot_vector : np.ndarray
        Knot vector
    n_basis : int
        Number of basis functions
        
    Returns
    -------
    np.ndarray
        Array of basis function values
    """
    basis = np.zeros(n_basis)
    
    # Find the knot span
    span = find_span(u, degree, knot_vector)
    
    # Compute non-zero basis functions
    N = np.zeros(degree + 1)
    N[0] = 1.0
    
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    
    for j in range(1, degree + 1):
        left[j] = u - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - u
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved
    
    # Copy to output array
    for j in range(degree + 1):
        basis[span - degree + j] = N[j]
    
    return basis


def bspline_basis_derivatives(
    u: float,
    degree: int, 
    knot_vector: np.ndarray,
    n_basis: int,
    n_derivs: int
) -> np.ndarray:
    """
    Compute B-spline basis functions and their derivatives.
    
    Parameters
    ----------
    u : float
        Parameter value where to evaluate
    degree : int
        Degree of the B-spline
    knot_vector : np.ndarray
        Knot vector
    n_basis : int
        Number of basis functions
    n_derivs : int
        Number of derivatives to compute
        
    Returns
    -------
    np.ndarray
        Array of shape (n_derivs + 1, n_basis) containing basis functions
        and their derivatives
    """
    ders = np.zeros((n_derivs + 1, n_basis))
    
    # Find the knot span
    span = find_span(u, degree, knot_vector)
    
    # Compute basis functions and derivatives
    ndu = np.zeros((degree + 1, degree + 1))
    ndu[0, 0] = 1.0
    
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    
    for j in range(1, degree + 1):
        left[j] = u - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - u
        saved = 0.0
        
        for r in range(j):
            # Lower triangle
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            
            # Upper triangle  
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        ndu[j, j] = saved
    
    # Load the basis functions
    for j in range(degree + 1):
        ders[0, span - degree + j] = ndu[j, degree]
    
    # Compute derivatives
    for r in range(degree + 1):
        s1 = 0
        s2 = 1
        a = np.zeros((2, degree + 1))
        a[0, 0] = 1.0
        
        for k in range(1, min(n_derivs + 1, degree + 1)):
            d = 0.0
            rk = r - k
            pk = degree - k
            
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else degree - r
            
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            
            ders[k, span - degree + r] = d
            
            # Switch rows
            s1, s2 = s2, s1
    
    # Multiply through by the correct factors
    r = degree
    for k in range(1, min(n_derivs + 1, degree + 1)):
        for j in range(degree + 1):
            ders[k, span - degree + j] *= r
        r *= (degree - k)
    
    return ders


# Numba-accelerated versions (if available)
if HAS_NUMBA and numba is not None:
    find_span = numba.jit(find_span, nopython=True)
    bspline_basis = numba.jit(bspline_basis, nopython=True)
    bspline_basis_derivatives = numba.jit(bspline_basis_derivatives, nopython=True)
