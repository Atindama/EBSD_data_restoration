# TODO should we have assertions that ensure the input objects weren't
# unintentionally modified?

from typing import Callable
import pytest
import tvflow as tv
import numpy as np

def orient_propagates_nans(orient_func: Callable, array: np.ndarray,
                           *args, **kwargs) -> bool:
    if not np.any(np.isnan(array)):
        raise ValueError("`array` must contain NaN values to test.")
    last_dim = array.ndim - 1
    out_array = orient_func(array, *args, **kwargs)
    return(
        np.all(
            np.all(np.isnan(out_array), axis=last_dim) == 
            np.any(np.isnan(array), axis=last_dim)
        )
    )

def all_unit_quats(q: np.ndarray) -> bool:
    unit_quat_tolerance = 1e-10 # TODO is this tolerance small enough?
    last_dim = q.ndim - 1
    return(
        not np.any(
            np.abs(np.linalg.norm(q, axis=last_dim) - 1) > unit_quat_tolerance
        )
    )

def test_orient_unit_quats():
    rng = np.random.default_rng(23459213) # arbitrary seed
    
    # range was chosen arbitrarily
    q = -54.3 + (103.7 - (-54.3)) * rng.random((20, 20, 4))
    # add zero quaternions
    q[(1, 3), (1, 3), :] = 0
    # add zero parts to some quaternions
    q[(2, 5), (4, 7), :] = 0
    # add NaN quaternions
    q[(4, 7), (3, 5), :] = np.nan
    # add NaN parts to quaternions
    q[(6, 8, 8), (5, 2, 2), (3, 0, 2)] = np.nan
    
    unit_q = tv.orient.unit_quats(q)

    # quaternions with NaN parts must become fully NaN
    assert orient_propagates_nans(tv.orient.unit_quats, q)
    # must preserve zeros and zero quaternions
    assert(
        np.all(
            (q == 0) == (unit_q == 0)
        )
    )
    # must be unit quaternions
    partly_nan_quats = np.any(np.isnan(q), axis=2)
    zero_quats = np.all(q == 0, axis=2)
    should_be_one = ~zero_quats & ~partly_nan_quats
    assert all_unit_quats(unit_q[should_be_one, :])
    # should work on volumetric images
    qv = rng.random((10, 10, 10, 4))
    unit_qv = tv.orient.unit_quats(qv)
    assert all_unit_quats(unit_qv)

def test_orient_eulers_to_quats():
    rng = np.random.default_rng(432632) # arbitrary seed
    e = -np.pi + (np.pi - (-np.pi)) * rng.random((10, 10, 3))
    e[(1, 4, 4), (4, 2, 2), (1, 0, 2)] = np.nan
    assert orient_propagates_nans(tv.orient.eulers_to_quats, e, "xyz")
    # TODO unfinished. Add more tests
    # TODO test that it works on arrays in an arbitrary number of dimensions
    # TODO test that conversion is correct

def test_orient_quats_to_eulers():
    rng = np.random.default_rng(52346) # arbitrary seed
    q = -1 + (1 - (-1)) * rng.random((10, 10, 4))
    q[(1, 4, 4), (4, 2, 2), (1, 0, 2)] = np.nan
    assert orient_propagates_nans(tv.orient.quats_to_eulers, q, "xyz")
    # TODO unfinished. Add more tests.
    # TODO test that it works on arrays in an arbitrary number of dimensions
    # TODO how do we test that conversion is correct? The corresponding Euler
    # angles might not be unique.

def test_misc_range_map():
    input = np.array([0, -1, 3, 1, np.nan, 2])
    nanmask = np.isnan(input)
    output = tv.misc.range_map(input, in_range=(-1, 3), out_range=(0, 1))
    expected = np.array([0.25, 0, 1, 0.5, np.nan, 0.75])

    # should preserve NaN positions
    assert(
        np.all(
            np.isnan(output) == nanmask
        )
    )
    # should map correctly
    assert(
        np.all(
            output[~nanmask] == expected[~nanmask]
        )
    )
    # should raise if input range start and end are the same but input array or
    # output range contain more than one value
    with pytest.raises(ValueError):
        _ = tv.misc.range_map(input, (1, 1), (0, 1))

# TODO outputs must be unit quaternions; what if there are zero quaternions? Do
# we just leave them as zero?
# TODO volumetric test
def test_tvflow_inpaint():
    rng = np.random.default_rng(46532146) # arbitrary seed
    q = rng.random((100, 100, 4))
    nan_inds_dim1 = 99*rng.random((10)).astype(int)
    nan_inds_dim2 = 99*rng.random((10)).astype(int)
    q[nan_inds_dim1, nan_inds_dim2, :] = np.nan
    known_mask = ~np.isnan(q)
    u = tv.inpaint(q, max_iters=300)

    # output must not contain NaNs
    assert(
        not np.any(
            np.isnan(u)
        )
    )
    # known values must be unchanged (but normalized)
    assert(
        np.all(
            tv.orient.unit_quats(q)[known_mask] == u[known_mask]
        )
    )
    # outputs must be unit quaternions
    assert all_unit_quats(u)

# TODO Outputs must be unit quaternions; what if there are zero quaternions? Do
# we just leave them as zero?
def test_tvflow_denoise():
    rng = np.random.default_rng(231536234) # arbitrary seed
    q = rng.random((100, 100, 4))
    u_unweighted = tv.denoise(q, max_iters=300, weighted=False)
    u_weighted   = tv.denoise(q, max_iters=300, weighted=True)

    # outputs must be unit quaternions
    assert all_unit_quats(u_unweighted)
    assert all_unit_quats(u_weighted)
    # should raise exception on NaN inputs
    q[2, 3, 1] = np.nan
    with pytest.raises(ValueError):
        _ = tv.denoise(q, max_iters=300, weighted=False)
    with pytest.raises(ValueError):
        _ = tv.denoise(q, max_iters=300, weighted=True)

def test_tvflow_shift():
    from tvflow._tvflow import _shift
    
    input = np.array(
        [[[ 1,  2,  3], [ 4,      5,  6], [ 7,  8,  9]],
         [[10, 11, 12], [13,     14, 15], [16, 17, 18]],          
         [[19, 20, 21], [22, np.nan, 24], [25, 26, 27]]]
    )
    output = _shift(input, axis=2, direction="+")
    expected = np.array(
        [[[ 1,  1,  2], [ 4,  4,      5], [ 7,  7,  8]],
         [[10, 10, 11], [13, 13,     14], [16, 16, 17]],
         [[19, 19, 20], [22, 22, np.nan], [25, 25, 26]]]
    )
    nanmask = np.isnan(expected)

    assert(
        np.all(
            np.isnan(output) == nanmask
        )
    )
    assert(
        np.all(
            output[~nanmask] == expected[~nanmask]
        )
    )

    output = _shift(input, axis=0, direction="-")
    expected = np.array(
        [[[10, 11, 12], [13,     14, 15], [16, 17, 18]],          
         [[19, 20, 21], [22, np.nan, 24], [25, 26, 27]],
         [[19, 20, 21], [22, np.nan, 24], [25, 26, 27]]]
    )
    nanmask = np.isnan(expected)
    
    assert(
        np.all(
            np.isnan(output) == nanmask
        )
    )
    assert(
        np.all(
            output[~nanmask] == expected[~nanmask]
        )
    )

# TODO what should _diff() do on NaNs?
def test_tvflow_diff():
    from tvflow._tvflow import _diff
    
    input = np.array(
        [[0, 0, 1, 3, 6],
         [5, 4, 6, 2, 3],
         [7, 2, 1, 7, 9]]
    )
    output = _diff(input, axis=1, type="+")
    expected = np.array(
        [[ 0,  1,  2,  3,  0],
         [-1,  2, -4,  1,  0],
         [-5, -1,  6,  2,  0]]
    )

    assert(
        np.all(
            output == expected
        )
    )

    output = _diff(input, axis=0, type="-")
    expected = np.array(
        [[ 0,  0,  0,  0,  0],
         [ 5,  4,  5, -1, -3],
         [ 2, -2, -5,  5,  6]]
    )
    assert(
        np.all(
            output == expected
        )
    )

    output = _diff(input, axis=1, type="0")
    expected = np.array(
        [[ 0  ,  0.5,  1.5,  2.5,  1.5],
         [-0.5,  0.5, -1  , -1.5,  0.5],
         [-2.5, -3  ,  2.5,  4  ,  1  ]]
    )
    assert(
        np.all(
            output == expected
        )
    )