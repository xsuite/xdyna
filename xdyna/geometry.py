import numpy as np


def closest_polygon_vertex_2D(x, y, vertices_x, vertices_y):
    """Numpy-based algorithm to return, for a set of points, the polygon vertex that is closest.

    Parameters
    ----------
    x : single value or numpy.array with x-coordinates of points
    y : single value or numpy.array with y-coordinates of points
    vertices_x : numpy.array with x-coordinates of vertices of polygon
    vertices_y : numpy.array with y-coordinates of vertices of polygon

    Caveat
    ------
    Depending on the size of your array, this function can use a lot of memory. Internally,
    it stores the product of the points and the vertices during the calculation.
    For instance, using this algorithm with 50k points and 2k vertices (all double precision)
    uses over 0.8GB of RAM

    Returns
    -------
    c_x, c_y, d
    c_x : single value or numpy.array with the x-coordinates of the closest vertices
    c_y : single value or numpy.array with the y-coordinates of the closest vertices
    d :   single value or numpy.array with the distances to the closest vertices
    """
    one_point = False
    if not hasattr(x,'__iter__') and not hasattr(y,'__iter__'):
        one_point = True
        x = np.array([x])
        y = np.array([y])
    d = np.sqrt( (vertices_x-x[:,None])**2 + (vertices_y-y[:,None])**2 )
    arg = d.argmin(axis=1)
    full = [range(len(x))]
    if one_point:
        return vertices_x[arg][0], vertices_y[arg][0], d[full,arg][0][0]
    else:
        return vertices_x[arg], vertices_y[arg], d[full,arg][0]



def farthest_polygon_vertex_2D(x, y, vertices_x, vertices_y):
    """Numpy-based algorithm to return, for a set of points, the polygon vertex that is farthest.

    Parameters
    ----------
    x : single value or numpy.array with x-coordinates of points
    y : single value or numpy.array with y-coordinates of points
    vertices_x : numpy.array with x-coordinates of vertices of polygon
    vertices_y : numpy.array with y-coordinates of vertices of polygon

    Caveat
    ------
    Depending on the size of your array, this function can use a lot of memory. Internally,
    it stores the product of the points and the vertices during the calculation.
    For instance, using this algorithm with 50k points and 2k vertices (all double precision)
    uses over 0.8GB of RAM

    Returns
    -------
    c_x, c_y, d
    c_x : single value or numpy.array with the x-coordinates of the farthest vertices
    c_y : single value or numpy.array with the y-coordinates of the farthest vertices
    d :   single value or numpy.array with the distances to the farthest vertices
    """
    one_point = False
    if not hasattr(x,'__iter__') and not hasattr(y,'__iter__'):
        one_point = True
        x = np.array([x])
        y = np.array([y])
    d = np.sqrt( (vertices_x-x[:,None])**2 + (vertices_y-y[:,None])**2 )
    arg = d.argmax(axis=1)
    full = [range(len(x))]
    if one_point:
        return vertices_x[arg][0], vertices_y[arg][0], d[full,arg][0][0]
    else:
        return vertices_x[arg], vertices_y[arg], d[full,arg][0]


# TODO: this is only approximate, as it doesn't calculate the distance to the polygon, but
# to its closest vertex. In our case this is good enough, as the border has many vertices.
# If ever needed, the exact distance can be calculated by:
#    1) interpolating the vertices (1D or spline) in function of a parameter u = 0..1
#    2) the distance to a point on the curve is given by
#       D = np.sqrt( (x(u) - x1)**2 + (y(u) - y1)**2 )
#    3) the distance is the minimum distance to any of the points. To avoid the square
#       root, we can minimize the square (because if dD/dx = 0 the also dD^2/dx = 2D dD/dx = 0)
#    4) the caveat is that the interpolated polygon is a periodic function, with which some
#       optimisers have issues
def distance_to_polygon_2D(x, y, vertices_x, vertices_y):
    """Numpy-based algorithm to give the approximated distance to a polygon.

    Parameters
    ----------
    x : single value or numpy.array with x-coordinates of points
    y : single value or numpy.array with y-coordinates of points
    vertices_x : numpy.array with x-coordinates of vertices of polygon
    vertices_y : numpy.array with y-coordinates of vertices of polygon

    Caveat
    ------
    Depending on the size of your array, this function can use a lot of memory. Internally,
    it stores 3 times the product of the points and the vertices during the calculation.
    For instance, using this algorithm with 50k points and 2k vertices (all double precision)
    uses over 2.4GB of RAM

    Returns
    -------
    Single value or numpy.array with the distances, negative if inside the polygon0
    """
    one_point = False
    if not hasattr(x,'__iter__') and not hasattr(y,'__iter__'):
        one_point = True
        x = np.array([x])
        y = np.array([y])
    _, _, distances = closest_polygon_vertex_2D(x, y, vertices_x, vertices_y)
    wn = winding_number_2D(x, y, vertices_x, vertices_y)
    if one_point:
        return np.where(wn==0, distances, -distances)[0]
    else:
        return np.where(wn==0, distances, -distances)
    return distances



def in_polygon_2D(x, y, vertices_x, vertices_y):
    """Numpy-based algorithm to test which points are inside a polygon.

    Parameters
    ----------
    x : single value or numpy.array with x-coordinates of points
    y : single value or numpy.array with y-coordinates of points
    vertices_x : numpy.array with x-coordinates of vertices of polygon
    vertices_y : numpy.array with y-coordinates of vertices of polygon

    Caveat
    ------
    Depending on the size of your array, this function can use a lot of memory. Internally,
    it stores 3 times the product of the points and the vertices during the calculation.
    For instance, using this algorithm with 50k points and 2k vertices (all double precision)
    uses over 2.4GB of RAM
    
    Returns
    -------
    Boolean single value or numpy.array
    """
    one_point = False
    if not hasattr(x,'__iter__') and not hasattr(y,'__iter__'):
        one_point = True
        x = np.array([x])
        y = np.array([y])
    wn = winding_number_2D(x, y, vertices_x, vertices_y)
    result = np.full_like(wn, True)
    result[wn==0] = False
    if one_point:
        return result[0]
    else:
        return result



def winding_number_2D(x, y, vertices_x, vertices_y):
    """Numpy-based algorithm to return winding numbers of a set of points w.r.t. a polygon.

    Parameters
    ----------
    x : single value or numpy.array with x-coordinates of points
    y : single value or numpy.array with y-coordinates of points
    vertices_x : numpy.array with x-coordinates of vertices of polygon
    vertices_y : numpy.array with y-coordinates of vertices of polygon
    
    Caveat
    ------
    Depending on the size of your array, this function can use a lot of memory. Internally,
    it stores 3 times the product of the points and the vertices during the calculation.
    For instance, using this algorithm with 50k points and 2k vertices (all double precision)
    uses over 2.4GB of RAM

    Returns
    -------
    Single value or numpy.array with the winding number of each point, i.e. the number of times the polygon
    wraps around the point, positive if the curve is defined anti-clockwise,
    negative otherwise. If the curve does not contain the point, the winding
    number is zero.

    Source
    ------
    Adapted from https://community.esri.com/t5/python-blog/point-in-polygon-geometry-mysteries/ba-p/893890
    """
    one_point = False
    if not hasattr(x,'__iter__') and not hasattr(y,'__iter__'):
        one_point = True
        x = np.array([x])
        y = np.array([y])
    if vertices_x[-1] == vertices_x[0] and vertices_y[-1] == vertices_y[0]:
        # polygon already cyclic
        x0 = vertices_x[:-1] # polygon `from` coordinates
        y0 = vertices_y[:-1]
        x1 = vertices_x[1:]  # polygon `to` coordinates
        y1 = vertices_y[1:]
    else:
        # make polygon cyclic
        x0 = vertices_x    # polygon `from` coordinates
        y0 = vertices_y
        x1 = np.append(vertices_x[1:], vertices_x[0])  # polygon `to` coordinates
        y1 = np.append(vertices_y[1:], vertices_y[0])
#     y_y0 = y[:, None] - y0
#     x_x0 = x[:, None] - x0
#     diff_ = (x1 - x0) * y_y0 - (y1 - y0) * x_x0  # diff => einsum in original
    chk1 = y[:,None]-y0 >= 0.0
    chk2 = np.less(y[:,None], y1)
    chk3 = np.sign( (x1-x0)*(y[:,None]-y0) - (y1-y0)*(x[:,None]-x0) ).astype(np.int)
    pos = (chk1 & chk2 & (chk3 > 0)).sum(axis=1, dtype=int)
    neg = (~chk1 & ~chk2 & (chk3 < 0)).sum(axis=1, dtype=int)
    result = pos - neg
    if one_point:
        return result[0]
    else:
        return result


# Function to give a bleed value (to overflow a border) in function of the equivalent circle curves
def _bleed(vertices_x, vertices_y, margin=0.1, minimum_bleed=0.5):
    if not in_polygon_2D(0, 0, vertices_x, vertices_y):
        raise ValueError("The function _bleed only has meaning if the curve wraps the origin!")
    _, _, inner_circle = closest_polygon_vertex_2D(0, 0, vertices_x, vertices_y)
    _, _, outer_circle = farthest_polygon_vertex_2D(0, 0, vertices_x, vertices_y)
    return inner_circle, outer_circle, max([margin*(outer_circle+inner_circle)/2, minimum_bleed])



