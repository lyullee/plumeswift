"""Scalar and batched QES three-component vector operations."""
import numpy as np


def _vector(values):
    values = np.asarray(values)
    if values.shape[-1:] != (3,):
        raise ValueError('Expected three vector components on the final axis')
    return values


def _dot32(left, right):
    x, y, z = left*right
    return x+y+z


def length32(vector):
    """Fast path for a single float32 vector already validated by its caller."""
    return np.sqrt(_dot32(vector, vector))


def reflect32(vector, normal):
    """Fast path for a single float32 vector and normal."""
    return vector-np.float32(2)*_dot32(vector, normal)*normal


def _dot(left, right):
    products = left*right
    return products[..., 0]+products[..., 1]+products[..., 2]


def dot(left, right):
    """Compute the dot product in the original x, y, z summation order."""
    return _dot(_vector(left), _vector(right))


def length(vector):
    vector = _vector(vector)
    return np.sqrt(_dot(vector, vector))


def distance(left, right):
    return length(_vector(left)-_vector(right))


def reflect(vector, normal):
    """Reflect across a supplied unit normal, preserving the input precision."""
    vector, normal = _vector(vector), _vector(normal)
    projection = _dot(vector, normal)
    return vector-(projection.dtype.type(2)*projection)[..., None]*normal
