import numpy as np
import tvflow as tv
import skimage.morphology as morph
import matplotlib.pyplot as plt
from skimage.filters import gaussian, farid
from numpy import pi, deg2rad

# `in_range` intentionally comes before `out_range`. We must not forget that in
# many cases we should specify the input range manually, because we cannot
# always assume that the input data contains datapoints at the extreme ends of
# its range. To keep conscious of this fact, specifying out_range in this
# function requires either specifying `in_range` or explicitly using a named
# argument.
def range_map(x: np.ndarray, in_range: tuple=None, out_range: tuple=(0, 1)):
    """
    Returns M(x), where M is the linear map that takes the interval in_range to
    out_range.
    NaN values remain as NaN.
    If in_range is unspecified, it is taken as [min(x), max(x)].

    Raises
    ------
    ValueError
        If `in_range` start and end are the same, but `x` or `out_range` contain
        more than one unique value.
    """
    if in_range == None:
        in_range = np.nanmin(x), np.nanmax(x)
    
    _, unique_inds = np.unique(x, return_index=True)
    if in_range[0] == in_range[1] and \
        (out_range[0] != out_range[1] or len(unique_inds) != 1):
        raise ValueError("If start and end of `in_range` are the same, then "
                         "`x` and `out_range` must each contain exactly 1 "
                         "unique element.")

    return out_range[0] + (x - in_range[0]) * \
           (out_range[1] - out_range[0]) / (in_range[1] - in_range[0])
           





def edges_to_weight(edge_mask, sigma=1):
    w = gaussian( 1.0 - edge_mask, sigma )
    min_w = 0.5
    w = (w - min_w) / (1.0 - min_w)
    w[ w<0 ] = 0
    return w

def highlight_edges_on_image(image, edges):
    image2 = image.copy() / image.max()
    image2[ edges ] = 1
    return image2

def weight_from_TV_solution(e, edge_strength_threshold=0.15, min_obj_size=64):
    if e.max() > 2*pi:
        e = (pi/180) * e.copy()
    e_tv = tv.denoise( e, weighted=False, on_quats=False )
    edges_tv = np.sum([farid(e_tv[:,:,k]) for k in range(e.shape[2])], 0)
    edges_tv = edges_tv > edge_strength_threshold
    edges_tv = morph.remove_small_objects( edges_tv, min_obj_size )
    edges_tv = morph.skeletonize( edges_tv )
    w = edges_to_weight( edges_tv, 1 )

    return w

def weight_from_TV_and_edge_detector(e, edge_strength_threshold=0.25,
                                     filter_min_obj_size=36, TV_min_obj_size=64):
    if e.max() > 2*pi:
        e = (pi/180) * e.copy()

    edges_farid = np.sum([farid(e[:,:,k]) for k in range(e.shape[2])], 0)
    edges_farid = edges_farid > edge_strength_threshold

    edges_d = morph.binary_dilation( edges_farid, morph.disk(2) )
    edges_de = morph.binary_erosion( edges_d, morph.disk(5) )
    edges_derm = morph.remove_small_objects( (~edges_de & edges_farid),
                                             filter_min_obj_size )
    edges_dermsk = morph.skeletonize( edges_derm )
    edges_farid = edges_dermsk # after dilation, erosion, rm small obj, skeletonize

    # plt.figure(); plt.imshow( highlight_edges_on_image( e, edges_farid ) )

    e_tv = tv.denoise( e, weighted=False, on_quats=False )

    edges_tv = np.sum([farid(e_tv[:,:,k]) for k in range(e.shape[2])], 0)
    edges_tv = edges_tv > edge_strength_threshold
    edges_tv = morph.remove_small_objects( edges_tv, TV_min_obj_size )
    edges_tv = morph.skeletonize( edges_tv )

    edges_tv[ ~edges_de ] = False # Accept edges_tv only btw the few grains
                                  # where edges_farid failed

    all_edges = edges_farid | edges_tv

    # plt.figure(); plt.imshow( highlight_edges_on_image( e, edges_farid ) )

    w = edges_to_weight( all_edges, 1 )
    return w
