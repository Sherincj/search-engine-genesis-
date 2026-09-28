"""
Query weighting — assigns mathematical weights to query terms based on frequency.
"""

import math

def query_term_weight(frequency: int, max_weight: float = 3.0) -> float:
    """
    Calculate a sublinear weight for a query term based on its frequency.
    
    Uses a logarithmic scale: weight = 1 + log(frequency)
    Capped at max_weight to prevent extreme score inflation.
    
    Parameters
    ----------
    frequency : int
        The number of times the term appears in the query.
    max_weight : float
        The maximum allowed weight (default 3.0).
        
    Returns
    -------
    float
        The calculated weight (minimum 1.0).
    """
    if frequency <= 0:
        raise ValueError("Frequency must be a positive integer.")
    
    weight = 1.0 + math.log(frequency)
    return min(weight, max_weight)
