"""Batched three-dimensional matrix operations and QES storage layouts."""
import numpy as np


_UPPER = ((0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2))
_SYMMETRIC = ((0, 1, 2), (1, 3, 4), (2, 4, 5))


def _matrix(values):
    values = np.asarray(values)
    if values.shape[-2:] != (3, 3):
        raise ValueError('Expected trailing matrix dimensions (3, 3)')
    return values


def from_column_major(values):
    """Expand nine QES column-major entries into ordinary row/column indexing."""
    values = np.asarray(values)
    if values.shape[-1:] != (9,):
        raise ValueError('Expected nine column-major matrix entries')
    return values.reshape((*values.shape[:-1], 3, 3)).swapaxes(-2, -1)


def to_column_major(values):
    """Return QES column-major entries on the final axis of each matrix."""
    values = _matrix(values)
    return values.swapaxes(-2, -1).reshape((*values.shape[:-2], 9))


def from_symmetric(values):
    """Expand the QES order xx, xy, xz, yy, yz, zz."""
    values = np.asarray(values)
    if values.shape[-1:] != (6,):
        raise ValueError('Expected six symmetric matrix entries')
    return values[..., _SYMMETRIC]


def to_symmetric(values):
    """Pack the upper triangle in QES symmetric order."""
    values = _matrix(values)
    return values[..., _UPPER[0], _UPPER[1]]


def symmetric_invariants(values):
    """Compute trace, principal-minor sum and determinant in QES GLE order."""
    values = _matrix(values)
    xx, yy, zz = (values[..., i, i] for i in range(3))
    xy, xz, yz = values[..., 0, 1], values[..., 0, 2], values[..., 1, 2]
    trace = xx+yy+zz
    minor_sum = xx*yy+xx*zz+yy*zz-(xy*xy+xz*xz+yz*yz)
    det = xx*(yy*zz-yz*yz)-xy*(xy*zz-yz*xz)+xz*(xy*yz-yy*xz)
    return np.stack((trace, minor_sum, det), axis=-1)


def inverse(values):
    """Invert each 3×3 matrix with a pivoted solve and reject singular input."""
    return np.linalg.inv(_matrix(values))


def native_inverse(values):
    """Replay Matrix3::invert's stored cofactors and near-zero determinant fallback."""
    packed = to_column_major(values)
    if packed.dtype.type not in (np.float32, np.float64):
        raise TypeError('The native inverse requires a float32 or float64 matrix')
    a, b, c, d, e, f, g, h, i = np.moveaxis(packed, -1, 0)
    first = e*i-f*h
    determinant = a*first-b*(d*i-f*g)+c*(b*h-e*g)
    small = np.abs(determinant).astype(np.float64, copy=False) < 1e-10
    determinant = np.where(small, packed.dtype.type(1e11), determinant)
    cofactors = np.stack((
        first, -(b*i-c*h), b*f-e*c,
        -(b*i-f*g), a*i-c*g, -(a*f-c*b),
        b*h-g*e, -(a*h-b*g), a*e-b*d,
    ), axis=-1)
    return from_column_major(cofactors/determinant[..., None])
