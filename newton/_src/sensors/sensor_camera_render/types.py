# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass

import warp as wp

from ...utils.color import ColorSpace


class LightType(enum.IntEnum):
    """Light types supported by the Warp raytracer."""

    SPOTLIGHT = 0
    """Spotlight."""

    DIRECTIONAL = 1
    """Directional Light."""


class RenderOrder(enum.IntEnum):
    """Render Order"""

    PIXEL_PRIORITY = 0
    """Render the same pixel of every view before continuing to the next one"""
    VIEW_PRIORITY = 1
    """Render all pixels of a whole view before continuing to the next one"""
    TILED = 2
    """Render pixels in tiles, defined by tile_width x tile_height"""


class WorldRenderFlag(enum.IntEnum):
    """Negative disable sentinels for the per-view ``world_indices`` array.

    Each entry of ``world_indices`` is either a non-negative world index to
    render for that view, or one of these negative sentinels to skip the view.
    """

    DISABLE_PRESERVE = -101
    """Skip rendering and leave output pixels unchanged."""

    DISABLE_CLEAR = -102
    """Skip rendering and write clear values to output pixels."""


class GaussianRenderMode(enum.IntEnum):
    """Gaussian Render Mode"""

    FAST = 0
    """Fast Render Mode"""

    QUALITY = 1
    """Quality Render Mode, collect hits until minimum transmittance is reached"""


class TextureProjectionMode(enum.IntEnum):
    """Projection mode for texture-mapped shapes without authored UVs."""

    CUBIC = 0
    """Project from the dominant local axis and sample once."""

    TRIPLANAR = 1
    """Blend samples from all three local axes using normal-based weights."""


@dataclass(unsafe_hash=True)
class RenderConfig:
    """Raytrace render settings shared across all worlds."""

    enable_global_world: bool = True
    """Include shapes that belong to no specific world."""

    enable_textures: bool = False
    """Enable texture-mapped rendering for shapes."""

    texture_projection_mode: int = TextureProjectionMode.CUBIC
    """Projection mode for texture-mapped shapes without UVs."""

    enable_shadows: bool = False
    """Enable shadow rays for directional lights."""

    enable_dome_lighting: bool = True
    """Enable dome lighting for the ambient term.

    Uses the HDRI set via :meth:`~newton.sensors.SensorCamera.set_dome_light`
    if one was provided, otherwise falls back to a built-in two-tone sky/ground
    gradient. Set to ``False`` to disable ambient shading entirely.
    """

    dome_shadow_samples: int = 0
    """Shadow-ray samples for HDRI dome lighting (soft shadows / directional occlusion).

    ``0`` uses the fast unshadowed spherical-harmonic irradiance. A value ``N > 0``
    casts ``N`` shadow rays per pixel, importance-sampled toward the bright parts
    of the environment (e.g. the sun), and accumulates the radiance from the
    unoccluded directions. Occluders therefore cast soft shadows from concentrated
    sources as well as diffuse sky, without the fireflies that uniform sampling
    produces. Directions are drawn stochastically with a per-pixel, per-frame seed:
    the estimate is unbiased and converges under temporal accumulation, but each
    frame carries Monte-Carlo noise that varies frame to frame. Raise ``N`` to
    reduce it. Only used when :attr:`enable_dome_lighting` is set.
    """

    dome_shadow_max_distance: float = 0.0
    """Maximum length [m] of dome shadow rays; ``0`` uses :attr:`max_distance`.

    Dome occlusion is usually local (contact/ambient shadows), so capping the
    shadow-ray length lets the BVH traversal terminate early and skips distant
    geometry, which can substantially speed up rendering on open scenes. Smaller
    values are faster but ignore occluders beyond the cap. Only used when
    :attr:`dome_shadow_samples` > 0.
    """

    enable_dome_background: bool = False
    """Show the HDRI dome environment map as the background for camera rays that miss all geometry.

    Requires an environment set via
    :meth:`~newton.sensors.SensorCamera.set_dome_light`; the built-in sky/ground
    ambient has no per-direction image to display. Only used when
    :attr:`enable_dome_lighting` is set.
    """

    enable_particles: bool = True
    """Enable standalone particle rendering.

    Particles referenced by rendered triangle or tetrahedral deformable topology
    are rendered by the triangle mesh path and are not emitted as particle
    spheres.
    """

    enable_backface_culling: bool = True
    """Cull back-facing triangles."""

    enable_fast_math: bool = True
    """Compile render kernels with CUDA fast math."""

    output_color_space: ColorSpace = ColorSpace.SRGB
    """Color space for packed color and albedo outputs.

    Use ``ColorSpace.SRGB`` for display-encoded bytes or
    ``ColorSpace.LINEAR`` for linear RGB bytes.
    """

    render_order: int = RenderOrder.PIXEL_PRIORITY
    """Render traversal order (see :class:`RenderOrder`)."""

    tile_width: int = 16
    """Tile width [px] for ``RenderOrder.TILED`` traversal."""

    tile_height: int = 8
    """Tile height [px] for ``RenderOrder.TILED`` traversal."""

    max_distance: float = 1000.0
    """Maximum ray distance [m]."""

    gaussians_mode: int = GaussianRenderMode.FAST
    """Gaussian splatting render mode (see :class:`GaussianRenderMode`)."""

    gaussians_min_transmittance: float = 0.49
    """Minimum transmittance before early-out during Gaussian rendering."""

    gaussians_max_num_hits: int = 20
    """Maximum Gaussian hits accumulated per ray."""


@dataclass(unsafe_hash=True)
class ClearData:
    """Default values written to output images before rendering."""

    clear_color: int = 0
    """Packed RGBA value written to the color output."""
    clear_depth: float = 0.0
    """Depth value written to the depth and forward-depth outputs [m]."""
    clear_shape_index: int = 0xFFFFFFFF
    """Shape-index sentinel written to the shape-index output."""
    clear_normal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Normal vector written to the normal output."""
    clear_albedo: int = 0
    """Packed RGBA value written to the albedo output."""


@wp.struct
class MeshData:
    """Per-mesh auxiliary vertex data for texture mapping and smooth shading.

    Attributes:
        uvs: Per-vertex UV coordinates, shape ``[vertex_count, 2]``, dtype ``vec2f``.
        normals: Per-vertex normals for smooth shading, shape ``[vertex_count, 3]``, dtype ``vec3f``.
    """

    uvs: wp.array[wp.vec2f]
    normals: wp.array[wp.vec3f]


@wp.struct
class TextureData:
    """Texture image data for surface shading during raytracing.

    Uses a hardware-accelerated ``wp.Texture2D`` with bilinear filtering.

    Attributes:
        texture: 2D Texture as ``wp.Texture2D``.
        repeat: UV tiling factors along U and V axes.
    """

    texture: wp.Texture2D
    repeat: wp.vec2f


@wp.struct
class DomeLight:
    """HDRI dome (environment) lighting data for diffuse ambient and shadowed sampling.

    See :func:`~newton._src.sensors.sensor_camera_render.dome.compute_dome_sh9` and
    :func:`~newton._src.sensors.sensor_camera_render.dome.equirect_frame` for how
    ``spherical_harmonics``/``forward``/``bitangent``/``up`` are derived, and
    :func:`~newton._src.sensors.sensor_camera_render.dome.sample_env` for how
    ``environment_map``/``row_cumulative_distribution``/``column_cumulative_distribution``/
    ``probability_density_scale`` are consumed.

    Attributes:
        spherical_harmonics: Order-2 diffuse-irradiance SH coefficients, shape ``[9]``, dtype ``vec3f``.
        intensity: Scalar multiplier applied to the dome contribution.
        environment_map: Equirectangular radiance map, shape ``[height, width]``, dtype ``vec3f``.
        forward: Equirectangular basis forward axis (azimuth 0).
        bitangent: Equirectangular basis bitangent axis (azimuth pi/2).
        up: Equirectangular basis up axis (the pole).
        row_cumulative_distribution: Marginal row CDF for importance sampling, shape ``[height]``.
        column_cumulative_distribution: Per-row conditional column CDF, shape ``[height, width]``.
        probability_density_scale: Scale converting sampled luminance to a solid-angle pdf.
    """

    spherical_harmonics: wp.array[wp.vec3f]
    intensity: wp.float32
    environment_map: wp.array2d[wp.vec3f]
    forward: wp.vec3f
    bitangent: wp.vec3f
    up: wp.vec3f
    row_cumulative_distribution: wp.array[wp.float32]
    column_cumulative_distribution: wp.array2d[wp.float32]
    probability_density_scale: wp.float32
