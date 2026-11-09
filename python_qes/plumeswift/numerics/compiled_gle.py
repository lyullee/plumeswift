"""Optional float64 3×3 GLE solve with partial pivoting and LAPACK fallback."""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def stress_needs_invariants(matrices, tolerance):
    # A positive Gershgorin bound bounds every eigenvalue from below.
    # Keep a margin from the invariant threshold and from cancellation.
    minimum = 2*max(tolerance, np.sqrt(tolerance), np.cbrt(tolerance))
    needs_check = np.ones(len(matrices), dtype=np.bool_)
    for row in range(len(matrices)):
        lower = np.inf
        scale = 0.
        for i in range(3):
            diagonal = matrices[row, i, i]
            radius = 0.
            for j in range(3):
                value = abs(matrices[row, i, j])
                scale = max(scale, value)
                if i != j:
                    radius += value
            lower = min(lower, diagonal-radius)
        if scale < 1e75 and lower > max(minimum, 1e-3*scale):
            needs_check[row] = False
    return needs_check


@njit(cache=True, nogil=True)
def solve_three_by_three(matrices, right_hand_side):
    result = np.empty_like(right_hand_side)
    fallback = np.zeros(len(matrices), dtype=np.bool_)
    work = np.empty((3, 4))
    for row in range(len(matrices)):
        scale = 0.
        for i in range(3):
            for j in range(3):
                value = matrices[row, i, j]
                work[i, j] = value
                scale = max(scale, abs(value))
            work[i, 3] = right_hand_side[row, i]
        # Let the reference solver handle weak or non-finite pivots.
        threshold = max(1e-8*scale, np.finfo(np.float64).tiny)
        for column in range(3):
            pivot = column
            for i in range(column+1, 3):
                if abs(work[i, column]) > abs(work[pivot, column]):
                    pivot = i
            value = work[pivot, column]
            if not np.isfinite(value) or abs(value) <= threshold:
                fallback[row] = True
                break
            if pivot != column:
                for j in range(column, 4):
                    work[column, j], work[pivot, j] = work[pivot, j], work[column, j]
            for i in range(column+1, 3):
                factor = work[i, column]/value
                for j in range(column+1, 4):
                    work[i, j] -= factor*work[column, j]
        if fallback[row]:
            continue
        for i in range(2, -1, -1):
            value = work[i, 3]
            for j in range(i+1, 3):
                value -= work[i, j]*result[row, j]
            result[row, i] = value/work[i, i]
            fallback[row] |= not np.isfinite(result[row, i])
    return result, fallback
