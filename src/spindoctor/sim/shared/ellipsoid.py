"""Shared ellipsoid geometry and shading for the simulator's two sides.

The image-side ellipsoid renderer (``spindoctor.sim.render.bodies.ellipsoid``) and the
navigator's simulated body and haze-envelope models (``spindoctor.nav_model.simulated``)
place an ellipsoid on a pixel grid through one projection, :func:`project_ellipsoid`,
and take its surface normals from :func:`ellipsoid_image_normals`.  With shared
conventions the planted scene error is the only error in a recovery measurement;
independent implementations would each carry their own conventions and any delta
between them would contaminate the measurement as an unknown systematic.

The navigator's predicted body is the smooth Lambert ellipsoid that
:func:`render_lambert_ellipsoid` draws.  The image side composes its own image from
the same projection and normals, adding relief, craters, photometric laws, and
textures, so the two sides never share a finished limb.

These helpers take explicit geometry arguments only.  They never read a
scene parameter mapping, so they cannot carry truth-side information across
the information boundary (see ``spindoctor.sim.scene``).
"""

from dataclasses import dataclass, replace
from typing import TypeVar, cast

import numpy as np

from spindoctor.sim.shared.inplace import result_array
from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'EllipsoidGeometry',
    'EllipsoidProjection',
    'ellipsoid_coverage',
    'ellipsoid_image_normals',
    'ellipsoid_lit',
    'lambert_from_normals',
    'outline_half_extents',
    'outline_semi_a',
    'project_ellipsoid',
    'render_lambert_ellipsoid',
    'supersampling_scale',
    'surface_normals',
    'surface_unit_coords',
]

# A vector component: one value, or one per pixel.
_Component = TypeVar('_Component', float, NDArrayFloatType)


@dataclass(frozen=True)
class EllipsoidProjection:
    """An ellipsoid projected onto one pixel grid.

    Produced by :func:`project_ellipsoid`.  Every array covers the whole grid, or the
    box that :meth:`crop` selects from it, and every length is in the grid's pixels.
    The grid is the one the caller stated the geometry on, supersampled by
    ``aa_scale``.

    The body frame has semi-axis a along its pole, semi-axis b across, and semi-axis
    c toward the observer when the body is seen equator-on.  The rotated frame is the
    image turned by the in-plane rotation, so its first axis is the direction the
    pole points in the image.  The north pole, where latitudes are positive, is the
    pole on the positive first axis.  A sub-observer latitude tips the pole about
    the second axis, the north pole toward the observer for a positive latitude, so
    the outline is the ellipse of semi-axes :attr:`semi_a_outline` and ``semi_b`` in
    the rotated frame.

    Parameters:
        aa_scale: Pixels of this grid per pixel of the grid the caller stated the
            geometry on, along each axis.
        center: Body center (v, u) in the whole grid's pixel corner coordinates.  A
            crop keeps it unchanged, so ``v_coords + center[0]`` and
            ``u_coords + center[1]`` are always a pixel's position on the whole grid.
        semi_a: Semi-axis a, along the pole, in this grid's pixels.
        semi_b: Semi-axis b in this grid's pixels.
        semi_c: Semi-axis c in this grid's pixels.
        semi_a_outline: The outline's semi-axis along the rotated first axis,
            ``hypot(semi_a * cos(lat), semi_c * sin(lat))`` for the sub-observer
            latitude ``lat``.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.
        cos_lat: Cosine of the sub-observer latitude.
        sin_lat: Sine of the sub-observer latitude.
        v_coords: The v coordinate of each pixel center relative to the body center,
            before rotation.  It is a read-only view of one value per row.
        u_coords: The u coordinate of each pixel center relative to the body center,
            before rotation.  It is a read-only view of one value per column.
        v_rot: The coordinate of each pixel along the rotated first axis.
        u_rot: The coordinate of each pixel along the rotated second axis.
        ellipse_dist_sq: The squared normalized ellipse distance of each pixel,
            ``(v_rot / semi_a_outline)**2 + (u_rot / semi_b)**2``, which is 1 on the
            limb.
    """

    aa_scale: int
    center: tuple[float, float]
    semi_a: float
    semi_b: float
    semi_c: float
    semi_a_outline: float
    cos_rz: float
    sin_rz: float
    cos_lat: float
    sin_lat: float
    v_coords: NDArrayFloatType
    u_coords: NDArrayFloatType
    v_rot: NDArrayFloatType
    u_rot: NDArrayFloatType
    ellipse_dist_sq: NDArrayFloatType

    def crop(self, rows: slice, cols: slice) -> 'EllipsoidProjection':
        """The part of the projection inside a box on its grid.

        The coordinates keep their origin at the body center, so each pixel keeps
        the values it has in the whole projection.  The center stays in the whole
        grid's coordinates, so adding it to the cropped coordinates still gives each
        pixel's position on the whole grid.  The image side's crater carving needs
        both, since it places craters by those positions.

        Parameters:
            rows: The box's rows.
            cols: The box's columns.

        Returns:
            The cropped projection, whose arrays are views into this one's.
        """
        return replace(
            self,
            v_coords=self.v_coords[rows, cols],
            u_coords=self.u_coords[rows, cols],
            v_rot=self.v_rot[rows, cols],
            u_rot=self.u_rot[rows, cols],
            ellipse_dist_sq=self.ellipse_dist_sq[rows, cols],
        )

    def surface_unit_coords(
        self,
    ) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
        """The body-frame point each pixel sees, on the unit sphere.

        See :func:`surface_unit_coords`, which this applies to every pixel.

        Returns:
            The ``(x, y, z)`` arrays.
        """
        return surface_unit_coords(
            self.v_rot, self.u_rot, self.ellipse_dist_sq, geometry=self.geometry
        )

    @property
    def geometry(self) -> 'EllipsoidGeometry':
        """The body's shape and orientation, without the grid."""
        return EllipsoidGeometry(
            semi_a=self.semi_a,
            semi_b=self.semi_b,
            semi_c=self.semi_c,
            cos_lat=self.cos_lat,
            sin_lat=self.sin_lat,
        )


@dataclass(frozen=True)
class EllipsoidGeometry:
    """An ellipsoid's semi-axes and sub-observer latitude, in one grid's pixels.

    Parameters:
        semi_a: Semi-axis a, along the pole.
        semi_b: Semi-axis b.
        semi_c: Semi-axis c.
        cos_lat: Cosine of the sub-observer latitude.
        sin_lat: Sine of the sub-observer latitude.
    """

    semi_a: float
    semi_b: float
    semi_c: float
    cos_lat: float
    sin_lat: float

    def to_view(
        self, x: _Component, y: _Component, z: _Component
    ) -> tuple[_Component, _Component, _Component]:
        """Turn body-frame vector components into the rotated frame.

        The rotated frame's third axis points toward the observer.

        Parameters:
            x: Components along the pole.
            y: Components along semi-axis b.
            z: Components along semi-axis c.

        Returns:
            The components along the rotated first axis, the second, and toward the
            observer.
        """
        if self.sin_lat == 0.0:
            return x, y, z
        return (
            x * self.cos_lat - z * self.sin_lat,
            y,
            x * self.sin_lat + z * self.cos_lat,
        )

    def to_body(
        self, first: _Component, second: _Component, toward: _Component
    ) -> tuple[_Component, _Component, _Component]:
        """Turn rotated-frame vector components into the body frame.

        Parameters:
            first: Components along the rotated first axis.
            second: Components along the rotated second axis.
            toward: Components toward the observer.

        Returns:
            The components along the pole, semi-axis b, and semi-axis c.
        """
        if self.sin_lat == 0.0:
            return first, second, toward
        return (
            first * self.cos_lat + toward * self.sin_lat,
            second,
            -first * self.sin_lat + toward * self.cos_lat,
        )


def supersampling_scale(anti_aliasing: float) -> int:
    """Pixels per input pixel, along each axis, of the grid :func:`project_ellipsoid` builds.

    Parameters:
        anti_aliasing: Anti-aliasing amount in [0, 1].

    Returns:
        ``int(1 + 3 * anti_aliasing)`` above 0, up to 4, and 1 at 0.
    """
    return int(1 + 3 * anti_aliasing) if anti_aliasing > 0 else 1


def outline_semi_a(semi_a: float, semi_c: float, sub_observer_lat: float) -> float:
    """The outline's semi-axis along the direction the pole points in the image.

    Tipping the pole toward or away from the observer by the sub-observer latitude
    foreshortens semi-axis a and brings semi-axis c into view, so the outline reaches
    ``hypot(semi_a * cos(lat), semi_c * sin(lat))``.  A sphere's outline is the same
    circle at every latitude.

    Parameters:
        semi_a: Semi-axis a, along the pole.
        semi_c: Semi-axis c.
        sub_observer_lat: The sub-observer latitude in radians.

    Returns:
        The outline semi-axis, in the semi-axes' unit.
    """
    if sub_observer_lat == 0.0:
        return semi_a
    return float(stable_hypot(semi_a * np.cos(sub_observer_lat), semi_c * np.sin(sub_observer_lat)))


def outline_half_extents(
    semi_a_px: float,
    *,
    semi_b_px: float,
    semi_c_px: float,
    rotation_z: float,
    sub_observer_lat: float,
) -> tuple[float, float]:
    """How far an ellipsoid's outline reaches from its center along v and along u.

    The outline is the ellipse of semi-axes :func:`outline_semi_a` and ``semi_b_px``
    turned in the image by ``rotation_z``, the outline :func:`project_ellipsoid`
    draws.

    Parameters:
        semi_a_px: Semi-axis a, along the pole, in pixels.
        semi_b_px: Semi-axis b in pixels.
        semi_c_px: Semi-axis c in pixels.
        rotation_z: The in-plane rotation in radians.
        sub_observer_lat: The sub-observer latitude in radians.

    Returns:
        The half-extents along v and along u, in pixels.
    """
    semi_a = outline_semi_a(semi_a_px, semi_c_px, sub_observer_lat)
    cos_rz = float(np.cos(rotation_z))
    sin_rz = float(np.sin(rotation_z))
    half_v = float(stable_hypot(semi_a * cos_rz, semi_b_px * sin_rz))
    half_u = float(stable_hypot(semi_a * sin_rz, semi_b_px * cos_rz))
    return half_v, half_u


def surface_unit_coords(
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    ellipse_dist_sq: NDArrayFloatType,
    *,
    geometry: EllipsoidGeometry,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """The body-frame point under each pixel, scaled to the unit sphere.

    The point is where the line of sight through the pixel first meets the
    ellipsoid, divided axis by axis by the semi-axes, so ``x`` is the sine of its
    latitude on a sphere.  Outside the limb the line of sight meets the ellipsoid
    nowhere.  There the point is the limb point at the pixel's azimuth on the
    outline, where the surface faces sideways, so every point has unit length.

    Parameters:
        v_rot: Each pixel's coordinate along the rotated first axis, in pixels.
        u_rot: Each pixel's coordinate along the rotated second axis, in pixels.
        ellipse_dist_sq: Each pixel's squared normalized outline distance.
        geometry: The ellipsoid's shape and orientation.

    Returns:
        ``(x, y, z)``: the components along the pole, semi-axis b, and semi-axis c.
    """
    # A pixel outside the limb moves in along its azimuth onto the outline.
    radial = np.sqrt(np.maximum(ellipse_dist_sq, 1.0))
    depth_fraction = np.sqrt(np.maximum(0.0, 1.0 - ellipse_dist_sq))
    if geometry.sin_lat == 0.0:
        return v_rot / geometry.semi_a / radial, u_rot / geometry.semi_b / radial, depth_fraction
    v_rot = v_rot / radial
    u_rot = u_rot / radial
    cos_lat, sin_lat = geometry.cos_lat, geometry.sin_lat
    inv_a2 = 1.0 / geometry.semi_a**2
    inv_c2 = 1.0 / geometry.semi_c**2
    # The line of sight through (v_rot, u_rot) meets the tilted ellipsoid where a
    # quadratic in the depth toward the observer vanishes; its discriminant is
    # proportional to 1 - ellipse_dist_sq, so the nearer root is the mid-point plus
    # the depth fraction over the square root of the leading coefficient.
    leading = sin_lat * sin_lat * inv_a2 + cos_lat * cos_lat * inv_c2
    toward = -v_rot * (cos_lat * sin_lat * (inv_a2 - inv_c2)) / leading + depth_fraction / np.sqrt(
        leading
    )
    x_body, y_body, z_body = geometry.to_body(v_rot, u_rot, toward)
    return x_body / geometry.semi_a, y_body / geometry.semi_b, z_body / geometry.semi_c


def surface_normals(
    v_rot: NDArrayFloatType,
    u_rot: NDArrayFloatType,
    ellipse_dist_sq: NDArrayFloatType,
    *,
    geometry: EllipsoidGeometry,
    cos_rz: float,
    sin_rz: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Unit surface normals in image coordinates at the surface points under the pixels.

    The points are those :func:`surface_unit_coords` gives.

    The normal at body-frame point ``(a x, b y, c z)`` is ``(x / a, y / b, z / c)``,
    normalized.  It is turned into the rotated frame by the sub-observer latitude,
    then into image coordinates through the inverse of the in-plane rotation.  At or
    outside the limb the point's surface faces sideways, so the normal lies in the
    image plane with ``normal_z`` 0.

    Parameters:
        v_rot: Each pixel's coordinate along the rotated first axis, in pixels.
        u_rot: Each pixel's coordinate along the rotated second axis, in pixels.
        ellipse_dist_sq: Each pixel's squared normalized outline distance.
        geometry: The ellipsoid's shape and orientation.
        cos_rz: Cosine of the in-plane rotation.
        sin_rz: Sine of the in-plane rotation.

    Returns:
        ``(normal_v, normal_u, normal_z)`` in image coordinates.
    """
    # The arrays are built in place wherever this call owns them: the image side
    # evaluates the normals over a large body on an oversampled grid, where a fresh
    # temporary costs more than its arithmetic.  Every step is the IEEE operation of
    # the plain expression, in the same operand order, so the normals are unchanged
    # to the last bit.
    if geometry.sin_lat == 0.0:
        # semi_c * sqrt(maximum(0, 1 - ellipse_dist_sq)) / semi_c**2
        normal_z_local = np.subtract(1.0, ellipse_dist_sq, out=result_array(ellipse_dist_sq))
        np.maximum(0.0, normal_z_local, out=normal_z_local)
        np.sqrt(normal_z_local, out=normal_z_local)
        normal_z_local *= geometry.semi_c
        normal_z_local /= geometry.semi_c**2
        normal_v_local = v_rot / (geometry.semi_a**2)
        normal_u_local = u_rot / (geometry.semi_b**2)
    else:
        x, y, z = surface_unit_coords(v_rot, u_rot, ellipse_dist_sq, geometry=geometry)
        normal_v_local, normal_u_local, normal_z_local = geometry.to_view(
            x / geometry.semi_a, y / geometry.semi_b, z / geometry.semi_c
        )
        # At or outside the limb the normal's component toward the observer is 0
        # in exact arithmetic; the rounding of the closest-approach depth leaves a
        # last-bit residue, which a renderer would read as a lit sliver of sky.
        normal_z_local = np.where(ellipse_dist_sq >= 1.0, 0.0, normal_z_local)
    components = (normal_v_local, normal_u_local, normal_z_local)
    if not _arrays_of_one_kind(components):
        # Scalars, components that broadcast against each other, or components of
        # different floating types: the plain expressions, which give the normals
        # their shape and type.  They are the expressions the steps below build in
        # place.
        magnitude = np.maximum(
            np.sqrt(normal_v_local**2 + normal_u_local**2 + normal_z_local**2), 1e-10
        )
        unit_v, unit_u, unit_z = (component / magnitude for component in components)
        return unit_v * cos_rz + unit_u * sin_rz, -unit_v * sin_rz + unit_u * cos_rz, unit_z

    # Normalize the normal vectors: sqrt(v**2 + u**2 + z**2), floored at 1e-10 to
    # avoid a division by zero.
    normal_mag = np.square(normal_v_local)
    scratch = np.square(normal_u_local)
    normal_mag += scratch
    np.square(normal_z_local, out=scratch)
    normal_mag += scratch
    np.sqrt(normal_mag, out=normal_mag)
    np.maximum(normal_mag, 1e-10, out=normal_mag)
    normal_v_local /= normal_mag
    normal_u_local /= normal_mag
    normal_z_local /= normal_mag

    # Rotate normal back to image coordinates (only v and u components)
    # The z component stays in the depth direction
    # Use inverse rotation (negate sin) to match the coordinate transformation:
    # normal_v = v * cos_rz + u * sin_rz and normal_u = -v * sin_rz + u * cos_rz.
    normal_v = np.multiply(normal_v_local, cos_rz, out=scratch)
    np.multiply(normal_u_local, sin_rz, out=normal_mag)
    normal_v += normal_mag
    normal_u = np.negative(normal_v_local, out=normal_v_local)
    normal_u *= sin_rz
    normal_u_local *= cos_rz
    normal_u += normal_u_local
    return normal_v, normal_u, normal_z_local


def _arrays_of_one_kind(components: tuple[NDArrayFloatType, ...]) -> bool:
    """Whether every component is an array of one shape and one type, not 0-d.

    Parameters:
        components: The components.

    Returns:
        True when they can be combined in place, with the plain expressions' result.
    """
    first = components[0]
    return all(
        isinstance(component, np.ndarray)
        and component.ndim > 0
        and component.shape == first.shape
        and component.dtype == first.dtype
        for component in components
    )


def project_ellipsoid(
    size: tuple[int, int],
    center: tuple[float, float],
    semi_a_px: float,
    *,
    semi_b_px: float,
    semi_c_px: float,
    rotation_z: float = 0.0,
    sub_observer_lat: float = 0.0,
    anti_aliasing: float = 0.0,
) -> EllipsoidProjection:
    """Project an ellipsoid onto a pixel grid.

    The navigator's predicted body is projected onto its supersampled frame here, and
    the image side's ellipsoid renderer projects each body onto its working grid and
    its detector grid.  The body's pole, semi-axis a, points along v before the
    in-plane rotation ``rotation_z`` turns it, with the north pole, where latitudes
    are positive, on +v.  The sub-observer latitude then tips the north pole toward
    the observer (positive) or away (negative) about semi-axis b, a true rotation of
    the ellipsoid, so the outline foreshortens along the pole to
    :func:`outline_semi_a`.

    Parameters:
        size: Tuple of (size_v, size_u) giving the image dimensions in pixels.
        center: Tuple of (v, u) giving the center position in floating-point
            pixels.  (0.0, 0.0) is the top-left corner of pixel (0,0),
            (0.5, 0.5) is the center of pixel (0,0).
        semi_a_px: Semi-axis a of the ellipsoid in pixels, half its length along the
            pole.
        semi_b_px: Semi-axis b of the ellipsoid in pixels, half its width across.
        semi_c_px: Semi-axis c of the ellipsoid in pixels, half its depth toward the
            observer when it is seen equator-on.
        rotation_z: Rotation angle around the viewing axis in radians.
        sub_observer_lat: The latitude of the point on the body directly below the
            observer, in radians, from -pi/2 to pi/2.
        anti_aliasing: Anti-aliasing amount in [0, 1].  Above 0 the grid is
            supersampled by ``int(1 + 3 * anti_aliasing)`` along each axis, up to 4;
            at 0 it is the ``size`` grid itself.

    Returns:
        The :class:`EllipsoidProjection` on the supersampled grid.
    """
    size_v, size_u = size
    aa_scale = supersampling_scale(anti_aliasing)

    center_v = center[0] * aa_scale
    center_u = center[1] * aa_scale
    semi_a = semi_a_px * aa_scale
    semi_b = semi_b_px * aa_scale
    semi_c = semi_c_px * aa_scale
    semi_a_outline = outline_semi_a(semi_a, semi_c, sub_observer_lat)

    # Pixel-center coordinates relative to the body center.  The grid starts as
    # array rows and columns and the projection works in the geometry layer's pixel
    # corner coordinates, where the caller states the center, so the half pixel goes
    # on here; that alignment holds at every supersampling scale because the center
    # is scaled with the grid.  Each coordinate varies along one axis only, so it is
    # computed once per row or column and broadcast over the grid.
    shape = (size_v * aa_scale, size_u * aa_scale)
    v_axis = (np.arange(shape[0], dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX) - center_v
    u_axis = (np.arange(shape[1], dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX) - center_u

    # Apply rotation_z (in-plane rotation around z-axis, clockwise)
    cos_rz = float(np.cos(rotation_z))
    sin_rz = float(np.sin(rotation_z))
    v_rot = (v_axis * cos_rz)[:, np.newaxis] - (u_axis * sin_rz)[np.newaxis, :]
    u_rot = (v_axis * sin_rz)[:, np.newaxis] + (u_axis * cos_rz)[np.newaxis, :]

    # The projected outline is the ellipse (v_rot/a_outline)^2 + (u_rot/b)^2 = 1.
    # Built in place, which gives the same bits as the plain expression with three
    # fewer grid-sized temporaries.
    ellipse_dist_sq = v_rot / semi_a_outline
    np.square(ellipse_dist_sq, out=ellipse_dist_sq)
    across = u_rot / semi_b
    np.square(across, out=across)
    ellipse_dist_sq += across
    del across

    return EllipsoidProjection(
        aa_scale=aa_scale,
        center=(center_v, center_u),
        semi_a=semi_a,
        semi_b=semi_b,
        semi_c=semi_c,
        semi_a_outline=semi_a_outline,
        cos_rz=cos_rz,
        sin_rz=sin_rz,
        cos_lat=float(np.cos(sub_observer_lat)),
        sin_lat=float(np.sin(sub_observer_lat)),
        v_coords=np.broadcast_to(v_axis[:, np.newaxis], shape),
        u_coords=np.broadcast_to(u_axis[np.newaxis, :], shape),
        v_rot=v_rot,
        u_rot=u_rot,
        ellipse_dist_sq=ellipse_dist_sq,
    )


def ellipsoid_image_normals(
    projection: EllipsoidProjection,
) -> tuple[NDArrayFloatType, NDArrayFloatType, NDArrayFloatType]:
    """Unit surface normals of the ellipsoid in image coordinates, at every pixel.

    The normals :func:`surface_normals` gives at each pixel of the projection.
    Outside the limb a pixel carries the normal of the limb point the line of sight
    passes closest to.  That normal lies in the image plane, with ``normal_z`` 0, so
    Lambert shading leaves the pixel dark.  A renderer that masks the body by its own
    outline can still read the normals there, and its shading continues smoothly
    across the limb.

    Parameters:
        projection: The ellipsoid's projection onto the grid.

    Returns:
        Tuple of (normal_v, normal_u, normal_z) unit-normal component arrays in
        image coordinates, on the projection's grid.
    """
    return surface_normals(
        projection.v_rot,
        projection.u_rot,
        projection.ellipse_dist_sq,
        geometry=projection.geometry,
        cos_rz=projection.cos_rz,
        sin_rz=projection.sin_rz,
    )


def lambert_from_normals(
    normal_v: NDArrayFloatType,
    normal_u: NDArrayFloatType,
    normal_z: NDArrayFloatType,
    *,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayFloatType:
    """Lambertian illumination strength for image-frame unit surface normals.

    The normals must already be unit length (or zero outside the body).

    Parameters:
        normal_v: V component of the unit surface normal in image coordinates.
        normal_u: U component of the unit surface normal in image coordinates.
        normal_z: Z (toward-observer) component of the unit surface normal.
        illumination_angle: In-plane light direction in radians; 0 is from the
            top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        Illumination strength array in [0, 1]; 0 on the night side and wherever
        ``normal_z`` is 0 or less.
    """
    illum_v_3d, illum_u_3d, illum_z_norm = illumination_vector(
        illumination_angle=illumination_angle, phase_angle=phase_angle
    )

    # Only illuminate points on the visible hemisphere (facing toward observer)
    # The z-component of the normal should be positive (pointing toward observer)
    visible_hemisphere = normal_z > 0

    # Compute cosine of incidence angle (Lambertian shading)
    # cos(incidence) = dot(normal, illumination_direction)
    cos_incidence = normal_v * illum_v_3d + normal_u * illum_u_3d + normal_z * illum_z_norm

    # Lambertian shading: I = I_0 * max(0, cos(incidence)) on the visible hemisphere.
    # The night side is black, as it is on a real airless body.
    illum_strength = np.where(visible_hemisphere, np.clip(cos_incidence, 0.0, 1.0), 0.0)

    return cast(NDArrayFloatType, illum_strength)


def render_lambert_ellipsoid(
    projection: EllipsoidProjection,
    *,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayFloatType:
    """Render a smooth ellipsoid with Lambert shading.

    This is the navigator's predicted body.  Each pixel of the projection's grid is
    shaded by :func:`lambert_from_normals` with the normals from
    :func:`ellipsoid_image_normals`.  The result is then averaged over blocks of
    ``aa_scale`` by ``aa_scale`` pixels.  Before averaging, a pixel whose center lies
    outside the limb or past the terminator is 0.  The night side is therefore 0 too,
    so the brightness does not mark the silhouette; :func:`ellipsoid_coverage` does.

    Parameters:
        projection: The ellipsoid's projection onto its grid, whose dimensions must
            be multiples of ``aa_scale``, as :func:`project_ellipsoid` builds them.
        illumination_angle: In-plane light direction in radians; 0 is from the
            top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        The intensity in [0, 1], on the grid the caller stated the geometry on: the
        projection's grid divided by ``aa_scale`` along each axis.
    """
    # Outside the limb every normal lies in the image plane, so the Lambert term is
    # already 0 there and the image needs no outline mask.
    intensity = lambert_from_normals(
        *ellipsoid_image_normals(projection),
        illumination_angle=illumination_angle,
        phase_angle=phase_angle,
    )
    scale = projection.aa_scale
    if scale > 1:
        rows, cols = intensity.shape
        intensity = intensity.reshape(rows // scale, scale, cols // scale, scale).mean(axis=(1, 3))
    return np.clip(intensity, 0.0, 1.0)


def ellipsoid_coverage(projection: EllipsoidProjection) -> NDArrayBoolType:
    """The pixels an ellipsoid's outline touches, lit or not.

    A pixel is covered when the center of any of its ``aa_scale`` by ``aa_scale``
    subpixels lies strictly inside the limb.  Those are exactly the pixels where
    :func:`render_lambert_ellipsoid` averages at least one subpixel of the visible
    hemisphere, so this is the silhouette that render draws, night side included.

    Parameters:
        projection: The ellipsoid's projection onto its grid, whose dimensions must
            be multiples of ``aa_scale``, as :func:`project_ellipsoid` builds them.

    Returns:
        The boolean silhouette on the grid the caller stated the geometry on: the
        projection's grid divided by ``aa_scale`` along each axis.
    """
    inside = projection.ellipse_dist_sq < 1.0
    scale = projection.aa_scale
    if scale > 1:
        rows, cols = inside.shape
        inside = inside.reshape(rows // scale, scale, cols // scale, scale).any(axis=(1, 3))
    covered: NDArrayBoolType = np.asarray(inside, dtype=bool)
    return covered


def ellipsoid_lit(
    projection: EllipsoidProjection,
    *,
    illumination_angle: float,
    phase_angle: float,
) -> NDArrayBoolType:
    """The pixels whose mean angle of incidence is below 90 deg.

    The mean is taken over the pixel's ``aa_scale`` by ``aa_scale`` subpixels whose
    centers lie strictly inside the limb, the subpixels :func:`ellipsoid_coverage`
    counts.  It is the rule the SPICE-backed body model uses for its lit region: a
    pixel straddling the terminator is lit only when most of its angle lies on the
    sunlit side, not whenever any part of it does.

    Parameters:
        projection: The ellipsoid's projection onto its grid, whose dimensions must
            be multiples of ``aa_scale``, as :func:`project_ellipsoid` builds them.
        illumination_angle: In-plane light direction in radians; 0 is from the
            top of the image, pi/2 from the right.
        phase_angle: Phase angle in radians; 0 is fully lit, pi is backlit.

    Returns:
        The boolean lit mask on the grid the caller stated the geometry on.
    """
    normal_v, normal_u, normal_z = ellipsoid_image_normals(projection)
    illum_v, illum_u, illum_z = illumination_vector(
        illumination_angle=illumination_angle, phase_angle=phase_angle
    )
    cos_incidence = normal_v * illum_v + normal_u * illum_u + normal_z * illum_z
    inside = projection.ellipse_dist_sq < 1.0
    angle = np.where(inside, np.arccos(np.clip(cos_incidence, -1.0, 1.0)), 0.0)
    scale = projection.aa_scale
    rows, cols = inside.shape
    angle_sum = angle.reshape(rows // scale, scale, cols // scale, scale).sum(axis=(1, 3))
    count = inside.reshape(rows // scale, scale, cols // scale, scale).sum(axis=(1, 3))
    lit: NDArrayBoolType = np.asarray(
        (count > 0) & (angle_sum < 0.5 * np.pi * np.maximum(count, 1)), dtype=bool
    )
    return lit
