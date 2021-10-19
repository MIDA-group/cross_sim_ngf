
import numpy as np
import transformations

def make_rot_matrix_3d(dim, rad, xyz_mode=True):
    a = np.zeros((3,3))
    for i in range(3):
        a[i, i] = 1.0
    cr = np.cos(rad)
    sr = np.sin(rad)
    # dim: 0 -> rotate around x
    # dim: 1 -> rotate around y
    # dim: 2 -> rotate around z
    
    if xyz_mode:
        xdim = 0
        ydim = 1
        zdim = 2
    else:
        xdim = 2
        ydim = 1
        zdim = 0

    if dim == 0:
        yy = cr
        zz = cr
        yz = -sr
        zy = sr

        a[ydim, ydim] = yy
        a[ydim, zdim] = yz
        a[zdim, ydim] = zy
        a[zdim, zdim] = zz
    elif dim == 1:
        xx = cr
        zz = cr
        xz = -sr
        zx = sr

        a[xdim, xdim] = xx
        a[xdim, zdim] = xz
        a[zdim, xdim] = zx
        a[zdim, zdim] = zz
    elif dim == 2:
        xx = cr
        yy = cr
        xy = -sr
        yx = sr

        a[xdim, xdim] = xx
        a[xdim, ydim] = xy
        a[ydim, xdim] = yx
        a[ydim, ydim] = yy

    return a

def make_empty(hom=True):
    if hom:
        m = np.zeros((4, 4))
        for i in range(4):
            m[i, i] = 1.0
    else:
        m = np.zeros((3, 4))
        for i in range(3):
            m[i, i] = 1.0
    return m

def make_translation_matrix(translation, hom=True):
    m = make_empty(hom)
    if translation is not None:
        m[:3, 3] = translation
    return m

def make_scaling_matrix(scale, hom=True, inv=False):
    m = make_empty(hom)
    if scale is not None:
        if inv:
            scale = 1.0 / scale
        m[0, 0] = scale[0]
        m[1, 1] = scale[1]
        m[2, 2] = scale[2]
    return m

def combine_rot_and_trans(rot, trans):
    res = trans.copy()
    res[:3, :3] = rot
    return res

def make_rigid3d_matrix(rads, shift, cp1, cp2, xyz_mode=False, hom=False):
    rot1 = make_rot_matrix_3d(0, rads[0], xyz_mode=xyz_mode)
    rot2 = make_rot_matrix_3d(1, rads[1], xyz_mode=xyz_mode)
    rot3 = make_rot_matrix_3d(2, rads[2], xyz_mode=xyz_mode)
    rot = rot3.dot(rot2.dot(rot1))
    t1 = make_translation_matrix(-cp1, hom=True)
    t2 = make_translation_matrix(cp2 + shift, hom=True)
    tf = combine_rot_and_trans(rot, t2).dot(t1)
    if hom == False:
        tf = tf[:3, :]
    return tf

def make_affine_transformation_from_matrix(matrix):
    tf = transformations.AffineTransform(3)
    for i in range(3):
        for j in range(3):
            tf.set_param(i * 3 + j, matrix[i, j])
        tf.set_param(9 + i, matrix[i, 3])
    return tf
