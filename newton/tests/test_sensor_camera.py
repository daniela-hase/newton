# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import math
import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
import newton._src.sensors.sensor_camera_render as internal_render
import newton.geometry as geometry
from newton._src.sensors.sensor_camera_render import dome
from newton._src.sensors.sensor_camera_render.utils import Utils
from newton._src.utils.texture import load_hdr_image
from newton.sensors import (
    SensorCamera,
)

# Transform placing a camera at the origin looking down -Z (identity pose). A
# camera with this transform sees a sphere placed at z = -2.
_IDENTITY_XFORM = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class TestSensorCamera(unittest.TestCase):
    @staticmethod
    def _rays(width: int, height: int, fov: float = math.radians(45.0), device: str = "cpu") -> wp.array3d[wp.vec3f]:
        """Camera-space pinhole rays, shape ``(height, width, 2)``."""
        return SensorCamera.compute_camera_rays_pinhole(width, height, fov, device=device)

    @staticmethod
    def _sphere_world_builder() -> newton.ModelBuilder:
        """A single-world scene with a sphere in front of an identity camera."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(sphere_body, radius=0.75, color=(0.25, 0.5, 0.75))
        return builder

    @classmethod
    def _build_sphere_scene(
        cls,
        *,
        world_count: int = 1,
        assign_render_context: bool = True,
    ) -> tuple[newton.Model, SensorCamera]:
        if world_count == 1:
            builder = cls._sphere_world_builder()
        else:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            for _ in range(world_count):
                builder.add_world(cls._sphere_world_builder())

        model = builder.finalize(device="cpu")
        camera = SensorCamera(model if assign_render_context else None)
        return model, camera

    @staticmethod
    def _identity_transforms(view_count: int, device: str = "cpu") -> wp.array[wp.transformf]:
        """World-space identity camera poses, shape ``(view_count,)``."""
        return wp.array(np.tile(_IDENTITY_XFORM, (view_count, 1)), dtype=wp.transformf, device=device)

    @staticmethod
    def _camera_with_model(model: newton.Model, **camera_kwargs) -> SensorCamera:
        """A SensorCamera that renders ``model`` (owns an internal render context)."""
        return SensorCamera(model, **camera_kwargs)

    def test_sensor_camera_public_imports_resolve_to_same_class(self) -> None:
        """Verify public SensorCamera imports and removed site-attachment helpers."""
        # SensorCamera lives in ``newton.sensors`` like every other sensor, not at
        # the top level or in ``newton.geometry``.
        self.assertIs(newton.sensors.SensorCamera, SensorCamera)
        self.assertFalse(hasattr(newton, "SensorCamera"))
        # The camera model spec classes were removed along with USD/MJCF camera import.
        for spec_name in (
            "CameraSpec",
            "CameraPinholeSpec",
            "CameraFisheyeOpenCVSpec",
            "CameraFisheyeFThetaSpec",
            "CameraFisheyeKannalaBrandtSpec",
        ):
            self.assertFalse(hasattr(newton, spec_name), spec_name)
            self.assertFalse(hasattr(newton.sensors, spec_name), spec_name)
        # RenderContext is an internal implementation detail owned by SensorCamera;
        # it is not part of the public API.
        self.assertFalse(hasattr(newton, "RenderContext"))
        self.assertNotIn("RenderContext", internal_render.__all__)
        self.assertFalse(hasattr(internal_render, "RenderContext"))
        # The render config/enum types are exposed as SensorCamera nested attributes,
        # not on the top-level namespace, and the ``newton.render`` module is gone.
        self.assertFalse(hasattr(newton, "render"))
        render_types = (
            "ClearData",
            "GaussianRenderMode",
            "LightType",
            "RenderConfig",
            "RenderOrder",
            "TextureProjectionMode",
            "WorldRenderFlag",
        )
        for type_name in render_types:
            self.assertFalse(hasattr(newton, type_name), type_name)
            self.assertNotIn(type_name, newton.__all__)
            self.assertTrue(hasattr(SensorCamera, type_name), type_name)
            self.assertIs(getattr(SensorCamera, type_name), getattr(internal_render, type_name))
        # The post-processing Utils and the gray clear preset are nested on SensorCamera.
        self.assertIs(SensorCamera.Utils, Utils)
        self.assertFalse(hasattr(geometry, "SensorCamera"))

        # The caller owns the rays, transforms, and output buffers; the sensor holds
        # none of them, and is not attached to model sites.
        camera = SensorCamera()
        for attr in (
            "rays",
            "view_count",
            "width",
            "height",
            "world_indices",
            "camera_transforms",
            "shape_indices",
            "_shape_index_by_world",
            "_camera_transforms",
        ):
            self.assertFalse(hasattr(camera, attr), attr)
        for attr in (
            "_compute_shape_index_by_world",
            "_update_transforms",
            "_ensure_shape_index_by_world",
            "_ensure_render_buffers",
        ):
            self.assertFalse(hasattr(SensorCamera, attr), attr)
        self.assertFalse(hasattr(internal_render, "_compute_camera_transforms"))

        self.assertFalse(hasattr(newton.ModelBuilder, "add_shape_camera"))
        self.assertFalse(hasattr(newton.ModelBuilder, "set_site_camera"))
        self.assertNotIn("camera", inspect.signature(newton.ModelBuilder.add_site).parameters)

        # Negative disable sentinels for the per-view world_indices array; no ENABLE.
        self.assertFalse(hasattr(SensorCamera.WorldRenderFlag, "ENABLE"))
        self.assertEqual(int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE), -101)
        self.assertEqual(int(SensorCamera.WorldRenderFlag.DISABLE_CLEAR), -102)

    def test_constructor_without_model_is_inert(self) -> None:
        """Verify a model-less SensorCamera owns only render settings and rejects rendering."""
        camera = SensorCamera()

        # Only render settings; no render context, rays, dimensions, or buffers.
        self.assertFalse(hasattr(camera, "render_context"))
        self.assertIsInstance(camera.default_render_config, SensorCamera.RenderConfig)
        self.assertIsInstance(camera.default_clear_data, SensorCamera.ClearData)

        state = self._sphere_world_builder().finalize(device="cpu").state()
        camera_transforms = self._identity_transforms(1)
        rays = self._rays(4, 4)
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.update(state, camera_transforms, rays)
        with self.assertRaisesRegex(RuntimeError, "no model"):
            _ = camera.device
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.create_color_image_output(1, 4, 4)
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.utils(1)

    def test_camera_ray_helpers_live_on_sensor_camera(self) -> None:
        """Verify camera ray helpers live on SensorCamera."""
        sensor_helper_names = (
            "compute_camera_rays_pinhole",
            "compute_camera_rays_usd_pinhole",
            "compute_camera_rays_fisheye_opencv",
            "compute_camera_rays_fisheye_ftheta",
            "compute_camera_rays_fisheye_kannala_brandt",
        )
        for helper_name in sensor_helper_names:
            self.assertTrue(hasattr(SensorCamera, helper_name))
            self.assertFalse(hasattr(Utils, helper_name))
        self.assertFalse(hasattr(Utils, "compute_pinhole_camera_rays"))
        self.assertFalse(hasattr(Utils, "compute_camera_transforms_usd"))
        self.assertFalse(hasattr(Utils, "create_default_light"))
        self.assertFalse(hasattr(Utils, "assign_checkerboard_material"))
        self.assertFalse(hasattr(Utils, "assign_checkerboard_material_to_all_shapes"))
        for helper_name in (
            "_create_image_output",
            "create_color_image_output",
            "create_depth_image_output",
            "create_forward_depth_image_output",
            "create_shape_index_image_output",
            "create_normal_image_output",
            "create_albedo_image_output",
            "create_hdr_color_image_output",
        ):
            self.assertFalse(hasattr(Utils, helper_name))

        width, height = 3, 3
        rays = [
            SensorCamera.compute_camera_rays_pinhole(width, height, math.radians(45.0), device="cpu"),
            SensorCamera.compute_camera_rays_pinhole(
                width,
                height,
                focal_length=1.0,
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            ),
            SensorCamera.compute_camera_rays_fisheye_opencv(
                width, height, fx=1.0, fy=1.0, cx=1.5, cy=1.5, device="cpu"
            ),
            SensorCamera.compute_camera_rays_fisheye_ftheta(
                width, height, optical_center_x=1.5, optical_center_y=1.5, device="cpu"
            ),
            SensorCamera.compute_camera_rays_fisheye_kannala_brandt(
                width, height, optical_center_x=1.5, optical_center_y=1.5, device="cpu"
            ),
        ]

        for ray_bundle in rays:
            self.assertEqual(ray_bundle.shape, (height, width, 2))
            self.assertEqual(ray_bundle.dtype, wp.vec3f)

    def test_camera_ray_helpers_support_preallocated_output(self) -> None:
        """Verify camera ray helpers can write into caller output arrays."""
        width, height = 4, 3
        out_rays = wp.zeros((height, width, 2), dtype=wp.vec3f, device="cpu")

        rays = SensorCamera.compute_camera_rays_pinhole(
            width, height, math.radians(45.0), out_rays=out_rays, device="cpu"
        )

        self.assertIs(rays, out_rays)
        self.assertFalse(np.allclose(rays.numpy(), 0.0))

    def test_camera_ray_helpers_reject_batched_inputs(self) -> None:
        """Verify camera ray helpers accept only single-camera parameters."""
        width, height = 4, 3

        with self.assertRaisesRegex(ValueError, "camera_fov cannot be provided with aperture parameters"):
            SensorCamera.compute_camera_rays_pinhole(
                width,
                height,
                math.radians(45.0),
                focal_length=1.0,
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            )

        with self.assertRaises(TypeError):
            SensorCamera.compute_camera_rays_pinhole(width, height, [math.radians(45.0)], device="cpu")

        with self.assertRaises(TypeError):
            SensorCamera.compute_camera_rays_pinhole(
                width,
                height,
                focal_length=wp.array([1.0], dtype=wp.float32, device="cpu"),
                horizontal_aperture=2.0,
                vertical_aperture=2.0,
                device="cpu",
            )

        out_rays = wp.zeros((1, height, width, 2), dtype=wp.vec3f, device="cpu")
        with self.assertRaisesRegex(ValueError, "out_rays must have shape"):
            SensorCamera.compute_camera_rays_pinhole(width, height, math.radians(45.0), out_rays=out_rays)

    def test_pinhole_rays_reject_out_of_range_parameters(self) -> None:
        """Verify pinhole ray generation rejects non-positive focal length and out-of-range fov."""
        width, height = 4, 3
        for bad_fov in (0.0, math.pi, -0.1, math.pi + 0.1):
            with self.assertRaisesRegex(ValueError, r"camera_fov must be in \(0, pi\)"):
                SensorCamera.compute_camera_rays_pinhole(width, height, bad_fov, device="cpu")
        with self.assertRaisesRegex(ValueError, "must be positive"):
            SensorCamera.compute_camera_rays_pinhole(
                width, height, focal_length=0.0, horizontal_aperture=2.0, vertical_aperture=2.0, device="cpu"
            )

    def test_update_validates_rays_and_transforms(self) -> None:
        """Verify update rejects mistyped or misshaped rays and camera transforms."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        camera_transforms = self._identity_transforms(model.world_count)

        with self.assertRaisesRegex(ValueError, "camera_transforms must have dtype"):
            camera.update(state, rays, rays)
        with self.assertRaisesRegex(ValueError, "camera_transforms must have shape"):
            camera.update(state, camera_transforms.reshape((model.world_count, 1)), rays)
        with self.assertRaisesRegex(ValueError, "camera_rays must have dtype"):
            camera.update(state, camera_transforms, camera_transforms)
        with self.assertRaisesRegex(ValueError, "camera_rays must have shape"):
            camera.update(state, camera_transforms, rays.reshape((1, height, width, 2)))
        with self.assertRaises(TypeError):
            camera.update(state, camera_transforms, np.zeros((height, width, 2), dtype=np.float32))

    def test_sync_transforms_is_explicit_and_not_called_by_update(self) -> None:
        """Verify update() no longer synchronizes render state; sync_transforms does it explicitly."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        transforms = self._identity_transforms(model.world_count)
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

        # A model-less camera cannot sync.
        with self.assertRaisesRegex(RuntimeError, "no model"):
            SensorCamera().sync_transforms(state)

        # Spy on the internal render-context sync to observe who triggers it.
        calls = []
        real_update = camera._render_context.update
        camera._render_context.update = calls.append
        try:
            camera.update(state, transforms, rays, depth_image=depth)
            self.assertEqual(calls, [], "update() must not synchronize render state")

            camera.sync_transforms(state)
            self.assertEqual(len(calls), 1, "sync_transforms() must synchronize render state")
        finally:
            camera._render_context.update = real_update

        # The render still produced a valid frame (rigid scene needs no sync).
        self.assertGreater(float(depth.numpy()[0, height // 2, width // 2]), 0.0)

    def test_model_required_for_outputs_and_utils(self) -> None:
        """Verify output and utility helpers require a model, and report the model device."""
        # A camera without a model cannot produce buffers or utils.
        camera = SensorCamera()
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.create_image_output(1, 4, 3, wp.float32)
        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.utils(1)

        model, camera = self._build_sphere_scene(world_count=2)
        self.assertEqual(camera.device, model.device)
        self.assertFalse(hasattr(model, "render_context"))
        self.assertFalse(hasattr(camera, "render_context"))

    def test_utils_and_scene_config_from_model(self) -> None:
        """Verify a model-backed SensorCamera exposes utils, output buffers, and scene config."""
        width, height = 4, 3
        model, camera = self._build_sphere_scene()
        view_count = model.world_count

        utils = camera.utils(view_count)

        self.assertIsInstance(utils, Utils)
        self.assertIsNot(camera.utils(view_count), utils)
        self.assertFalse(hasattr(utils, "_Utils__sensor_camera"))
        self.assertFalse(hasattr(model, "render_context"))
        self.assertFalse(hasattr(camera, "render_context"))
        self.assertEqual(camera.device, model.device)
        self.assertFalse(hasattr(camera, "_model_ref"))

        output_specs = (
            (camera.create_image_output(view_count, width, height, wp.float32), wp.float32),
            (camera.create_color_image_output(view_count, width, height), wp.uint32),
            (camera.create_depth_image_output(view_count, width, height), wp.float32),
            (camera.create_forward_depth_image_output(view_count, width, height), wp.float32),
            (camera.create_shape_index_image_output(view_count, width, height), wp.uint32),
            (camera.create_normal_image_output(view_count, width, height), wp.vec3f),
            (camera.create_albedo_image_output(view_count, width, height), wp.uint32),
            (camera.create_hdr_color_image_output(view_count, width, height), wp.vec3f),
        )
        for output, dtype in output_specs:
            with self.subTest(dtype=dtype):
                self.assertEqual(output.shape, (view_count, height, width))
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device, model.device)
        color_rgba = utils.to_rgba_from_color(camera.create_color_image_output(view_count, width, height))
        self.assertEqual(color_rgba.shape, (view_count, height, width, 4))
        # Scene configuration is surfaced on the camera; the render context is private.
        camera.create_default_light(enable_shadows=True)
        camera.assign_checkerboard_material(shape_indices=[0])

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_update_requires_arrays_on_model_device(self) -> None:
        """Verify update rejects rays or transforms that are not on the model device."""
        width, height = 2, 2
        model = self._sphere_world_builder().finalize(device="cuda:0")
        camera = SensorCamera(model)
        self.assertEqual(camera.device, model.device)
        state = model.state()

        cpu_rays = self._rays(width, height, device="cpu")
        cpu_transforms = self._identity_transforms(model.world_count, device="cpu")
        cuda_rays = self._rays(width, height, device="cuda:0")
        cuda_transforms = self._identity_transforms(model.world_count, device="cuda:0")

        with self.assertRaisesRegex(RuntimeError, "camera_transforms must be on the model device"):
            camera.update(state, cpu_transforms, cuda_rays)
        with self.assertRaisesRegex(RuntimeError, "camera_rays must be on the model device"):
            camera.update(state, cuda_transforms, cpu_rays)

    def test_update_renders_from_camera_transforms(self) -> None:
        """Verify SensorCamera renders from the camera transforms passed to update."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((view_count, height, width), dtype=wp.uint32, device="cpu")

        # Identity transforms see the sphere placed in front of the camera.
        camera_transforms = self._identity_transforms(view_count)
        camera.update(state, camera_transforms, rays, depth_image=depth, shape_index_image=shape_index)

        center = (0, height // 2, width // 2)
        identity_center_depth = float(depth.numpy()[center])
        self.assertGreater(identity_center_depth, 0.0)
        self.assertTrue(np.any(shape_index.numpy() != 0xFFFFFFFF))
        self.assertFalse(hasattr(model, "render_context"))

        # Move the camera behind the sphere; the center ray no longer hits it.
        behind = np.tile(np.array([0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (view_count, 1))
        camera_transforms.assign(behind)
        depth.zero_()
        camera.update(state, camera_transforms, rays, depth_image=depth)
        self.assertEqual(float(depth.numpy()[center]), 0.0)

    def test_update_respects_disable_clear_flag(self) -> None:
        """Verify SensorCamera clears output images for DISABLE_CLEAR worlds."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(world_count=2)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        camera.default_clear_data = SensorCamera.ClearData(clear_depth=-2.0, clear_shape_index=123)
        world_indices = wp.array(
            [0, int(SensorCamera.WorldRenderFlag.DISABLE_CLEAR)],
            dtype=wp.int32,
            device="cpu",
        )
        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((view_count, height, width), dtype=wp.uint32, device="cpu")

        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            depth_image=depth,
            shape_index_image=shape_index,
            world_indices=world_indices,
        )

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        self.assertEqual(float(depth_np[1, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index_np[1, height // 2, width // 2]), 123)

    def test_update_requires_model(self) -> None:
        """Verify SensorCamera rendering requires a model given at construction."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene(assign_render_context=False)
        state = model.state()
        depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

        with self.assertRaisesRegex(RuntimeError, "no model"):
            camera.update(
                state,
                self._identity_transforms(model.world_count),
                self._rays(width, height),
                depth_image=depth,
            )

    def test_update_respects_disable_preserve_flag(self) -> None:
        """Verify SensorCamera preserves output images for DISABLE_PRESERVE worlds."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(world_count=2)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        world_indices = wp.array(
            [0, int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE)],
            dtype=wp.int32,
            device="cpu",
        )
        depth = wp.full((view_count, height, width), value=42.0, dtype=wp.float32, device="cpu")
        shape_index = wp.full((view_count, height, width), value=456, dtype=wp.uint32, device="cpu")

        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            depth_image=depth,
            shape_index_image=shape_index,
            world_indices=world_indices,
        )

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        self.assertGreater(float(depth_np[0, height // 2, width // 2]), 0.0)
        np.testing.assert_allclose(depth_np[1], 42.0)
        np.testing.assert_array_equal(shape_index_np[1], np.full((height, width), 456, dtype=np.uint32))

    def test_update_defaults_world_indices_to_identity(self) -> None:
        """Verify update maps view i to world i when world_indices is omitted."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene(world_count=2)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")

        # No world_indices passed: each view renders its own world (identity mapping).
        camera.update(state, self._identity_transforms(view_count), rays, depth_image=depth)

        center = (height // 2, width // 2)
        self.assertTrue(all(float(depth.numpy()[v][center]) > 0.0 for v in range(view_count)))

    def test_update_without_world_indices_maps_view_to_world(self) -> None:
        """Verify omitting world_indices renders view i into world i, with no cached mapping array."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene(world_count=5)
        state = model.state()
        rays = self._rays(width, height)
        center = (height // 2, width // 2)

        # The sensor holds no default-mapping array; the renderer uses the view index.
        self.assertFalse(hasattr(camera, "_default_world_indices"))

        # Rendering different view counts (each <= world_count) works with no mapping;
        # each view renders its own world, so every view sees its sphere.
        for view_count in (3, 5, 2, 4):
            depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
            camera.update(state, self._identity_transforms(view_count), rays, depth_image=depth)
            self.assertTrue(all(float(depth.numpy()[v][center]) > 0.0 for v in range(view_count)))

    def test_world_indices_decouple_views_from_worlds(self) -> None:
        """Verify multiple views can render one shared world from different poses."""
        width, height = 8, 6
        # One world (sphere at z=-2) but three views, all rendering world 0.
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        self.assertEqual(model.world_count, 1)

        # Three views of world 0 from progressively closer poses.
        transforms = np.tile(_IDENTITY_XFORM, (3, 1))
        transforms[1, 2] = -0.5
        transforms[2, 2] = -1.0
        camera_transforms = wp.array(transforms, dtype=wp.transformf, device="cpu")
        world_indices = wp.array(np.zeros(3, dtype=np.int32), dtype=wp.int32, device="cpu")

        depth = camera.create_depth_image_output(3, width, height)
        self.assertEqual(depth.shape, (3, height, width))
        camera.update(state, camera_transforms, rays, depth_image=depth, world_indices=world_indices)

        d = depth.numpy()
        center = (height // 2, width // 2)
        self.assertTrue(all(float(d[v][center]) > 0.0 for v in range(3)))
        # The closer camera measures a smaller hit distance.
        self.assertGreater(float(d[0][center]), float(d[2][center]))

    def test_update_rejects_default_world_indices_exceeding_world_count(self) -> None:
        """Verify the default identity mapping is rejected when there are more views than worlds."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene()  # 1 world
        state = model.state()
        rays = self._rays(width, height)
        # Two views on a one-world model with no explicit mapping would index world 1.
        camera_transforms = self._identity_transforms(2)
        depth = wp.zeros((2, height, width), dtype=wp.float32, device="cpu")
        with self.assertRaisesRegex(ValueError, "exceeds model.world_count"):
            camera.update(state, camera_transforms, rays, depth_image=depth)

        # An explicit mapping to the valid world renders both views.
        world_indices = wp.array(np.zeros(2, dtype=np.int32), dtype=wp.int32, device="cpu")
        camera.update(state, camera_transforms, rays, depth_image=depth, world_indices=world_indices)
        center = (height // 2, width // 2)
        self.assertTrue(all(float(depth.numpy()[v][center]) > 0.0 for v in range(2)))

    def test_texture_projection_modes_texture_uvless_shapes(self) -> None:
        """Verify cubic and triplanar projection texture UV-less shapes and differ.

        A checkerboard is projected onto a UV-less sphere; both projection modes
        must texture it, and they must produce distinct results on the curved
        surface.
        """
        width, height = 32, 32

        def render(mode: int) -> np.ndarray:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            sphere_body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.5), q=wp.quat_identity()))
            sphere = builder.add_shape_sphere(sphere_body, radius=1.2, color=(1.0, 1.0, 1.0))
            model = builder.finalize(device="cpu")
            camera = self._camera_with_model(model)
            camera.default_render_config = SensorCamera.RenderConfig(enable_textures=True, texture_projection_mode=mode)
            camera.assign_checkerboard_material(shape_indices=[sphere])
            state = model.state()
            rays = self._rays(width, height, math.radians(60.0))
            albedo = camera.create_albedo_image_output(model.world_count, width, height)
            camera.update(state, self._identity_transforms(model.world_count), rays, albedo_image=albedo)
            return albedo.numpy()

        cubic = render(SensorCamera.TextureProjectionMode.CUBIC)
        triplanar = render(SensorCamera.TextureProjectionMode.TRIPLANAR)

        # Both modes project the checkerboard onto the UV-less sphere (not flat white).
        self.assertGreater(len(np.unique(cubic)), 1)
        self.assertGreater(len(np.unique(triplanar)), 1)
        # The two projection modes produce distinct results on a curved surface.
        self.assertFalse(np.array_equal(cubic, triplanar))

    def test_update_uses_default_render_settings(self) -> None:
        """Verify update falls back to the default clear data and render config."""
        parameters = inspect.signature(SensorCamera.update).parameters
        self.assertIn("camera_transforms", parameters)
        self.assertIn("camera_rays", parameters)
        self.assertIn("world_indices", parameters)
        self.assertIn("clear_data", parameters)
        self.assertIn("render_config", parameters)
        self.assertNotIn("load_textures", parameters)
        self.assertNotIn("world_enabled", parameters)
        self.assertNotIn("model", parameters)

        width, height = 16, 12
        model = self._sphere_world_builder().finalize(device="cpu")
        # Defaults may also be provided at construction.
        camera = SensorCamera(
            model,
            default_clear_data=SensorCamera.ClearData(clear_depth=-2.0, clear_shape_index=123),
            default_render_config=SensorCamera.RenderConfig(max_distance=0.1),
            load_textures=False,
        )
        self.assertFalse(hasattr(camera, "load_textures"))

        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        shape_index = wp.zeros((view_count, height, width), dtype=wp.uint32, device="cpu")

        # No per-call overrides: max_distance=0.1 misses the sphere, so the depth
        # and shape-index outputs take the default clear values.
        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            depth_image=depth,
            shape_index_image=shape_index,
        )

        self.assertEqual(float(depth.numpy()[0, height // 2, width // 2]), -2.0)
        self.assertEqual(int(shape_index.numpy()[0, height // 2, width // 2]), 123)

    def test_update_overrides_default_render_settings(self) -> None:
        """Verify per-call clear_data and render_config override the defaults."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        center = (0, height // 2, width // 2)

        # Defaults would clear to -7.0 and cull the sphere (max_distance=0.1)...
        camera.default_clear_data = SensorCamera.ClearData(clear_depth=-7.0)
        camera.default_render_config = SensorCamera.RenderConfig(max_distance=0.1)

        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        # ...but the per-call overrides raise max_distance so the sphere is hit.
        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            depth_image=depth,
            clear_data=SensorCamera.ClearData(clear_depth=-3.0),
            render_config=SensorCamera.RenderConfig(max_distance=1000.0),
        )
        self.assertGreater(float(depth.numpy()[center]), 0.0)

        # A miss with the override clear_data writes the override's clear value.
        depth.zero_()
        behind = np.tile(np.array([0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (view_count, 1))
        camera.update(
            state,
            wp.array(behind, dtype=wp.transformf, device="cpu"),
            rays,
            depth_image=depth,
            clear_data=SensorCamera.ClearData(clear_depth=-3.0),
            render_config=SensorCamera.RenderConfig(max_distance=1000.0),
        )
        self.assertEqual(float(depth.numpy()[center]), -3.0)

    def test_update_supports_all_render_orders_with_3d_outputs(self) -> None:
        """Verify SensorCamera renders every render order into 3-D outputs."""
        width, height = 16, 12

        for render_order in SensorCamera.RenderOrder:
            with self.subTest(render_order=render_order):
                model, camera = self._build_sphere_scene()
                state = model.state()
                rays = self._rays(width, height)
                camera.default_render_config = SensorCamera.RenderConfig(render_order=render_order)

                depth = wp.zeros((model.world_count, height, width), dtype=wp.float32, device="cpu")

                camera.update(state, self._identity_transforms(model.world_count), rays, depth_image=depth)

                self.assertGreater(float(depth.numpy()[0, height // 2, width // 2]), 0.0)

    def test_multiple_sensor_cameras_render_same_model(self) -> None:
        """Verify multiple independent SensorCamera instances can render the same model."""
        width, height = 8, 6
        model = self._sphere_world_builder().finalize(device="cpu")
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count

        depth_a = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        depth_b = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")

        # Each camera owns its own private render context for the same model.
        camera_a = self._camera_with_model(model)
        camera_b = self._camera_with_model(model)

        camera_a.update(state, self._identity_transforms(view_count), rays, depth_image=depth_a)
        camera_b.update(state, self._identity_transforms(view_count), rays, depth_image=depth_b)

        self.assertGreater(float(depth_a.numpy()[0, height // 2, width // 2]), 0.0)
        self.assertGreater(float(depth_b.numpy()[0, height // 2, width // 2]), 0.0)

    # --- Rendering output channels (ported from SensorTiledCamera coverage) ---

    @staticmethod
    def _shaded_sphere_model(color: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> newton.Model:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(body, radius=1.0, color=color)
        return builder.finalize(device="cpu")

    def _render_color_and_hdr(self, output_color_space) -> tuple[np.ndarray, np.ndarray]:
        width, height = 4, 4
        model = self._shaded_sphere_model()
        camera = self._camera_with_model(model)
        camera.default_render_config = SensorCamera.RenderConfig(output_color_space=output_color_space)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        color = camera.create_color_image_output(view_count, width, height)
        hdr = camera.create_hdr_color_image_output(view_count, width, height)
        camera.update(state, self._identity_transforms(view_count), rays, color_image=color, hdr_color_image=hdr)
        return np.asarray(color.numpy(), dtype=np.uint32), np.asarray(hdr.numpy(), dtype=np.float32)

    def test_render_hdr_color_output(self) -> None:
        """Verify SensorCamera produces a finite, non-zero HDR color channel."""
        color, hdr = self._render_color_and_hdr(newton.utils.ColorSpace.SRGB)
        self.assertEqual(color.shape, (1, 4, 4))
        self.assertEqual(hdr.shape, (1, 4, 4, 3))
        self.assertEqual(color.dtype, np.uint32)
        self.assertEqual(hdr.dtype, np.float32)
        self.assertTrue(np.isfinite(hdr).all())
        self.assertGreater(hdr.max(), 0.0)

    def test_hdr_color_matches_srgb_packed_color(self) -> None:
        """Verify packed color is the sRGB encoding of the HDR color for SRGB output."""
        color, hdr = self._render_color_and_hdr(newton.utils.ColorSpace.SRGB)
        clipped = np.clip(hdr, 0.0, 1.0)
        expected = np.where(clipped <= 0.0031308, clipped * 12.92, 1.055 * np.power(clipped, 1.0 / 2.4) - 0.055)
        packed = color.view(np.uint8).reshape(*color.shape, 4)[..., :3].astype(np.float32) / 255.0
        np.testing.assert_allclose(expected, packed, atol=1.0 / 255.0)

    def test_hdr_color_matches_linear_packed_color(self) -> None:
        """Verify packed color equals the clipped HDR color for LINEAR output."""
        color, hdr = self._render_color_and_hdr(newton.utils.ColorSpace.LINEAR)
        packed = color.view(np.uint8).reshape(*color.shape, 4)[..., :3].astype(np.float32) / 255.0
        np.testing.assert_allclose(np.clip(hdr, 0.0, 1.0), packed, atol=1.0 / 255.0)

    def test_albedo_output_follows_output_color_space(self) -> None:
        """Verify albedo packing honors the render-config output color space."""
        width, height = 8, 8
        model = self._shaded_sphere_model(color=(0.25, 0.5, 0.75))

        def render_albedo(space) -> np.ndarray:
            camera = self._camera_with_model(model)
            camera.default_render_config = SensorCamera.RenderConfig(output_color_space=space)
            state = model.state()
            rays = self._rays(width, height)
            albedo = camera.create_albedo_image_output(model.world_count, width, height)
            camera.update(state, self._identity_transforms(model.world_count), rays, albedo_image=albedo)
            return albedo.numpy()

        srgb = render_albedo(newton.utils.ColorSpace.SRGB)
        linear = render_albedo(newton.utils.ColorSpace.LINEAR)
        self.assertFalse(np.array_equal(srgb, linear))

    def test_render_forward_depth_output(self) -> None:
        """Verify forward-depth is positive and never exceeds ray-hit distance."""
        width, height = 16, 12
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        depth = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        forward = wp.zeros((view_count, height, width), dtype=wp.float32, device="cpu")
        camera.update(
            state, self._identity_transforms(view_count), rays, depth_image=depth, forward_depth_image=forward
        )
        center = (0, height // 2, width // 2)
        fwd = float(forward.numpy()[center])
        ray = float(depth.numpy()[center])
        self.assertGreater(fwd, 0.0)
        self.assertLessEqual(fwd, ray + 1.0e-4)

    # --- Dome (HDRI) lighting ---

    @staticmethod
    def _encode_flat_hdr(value: float, height: int, width: int) -> bytes:
        """Encode a constant-gray Radiance ``.hdr`` (flat RGBE scanlines)."""
        mant, exp2 = math.frexp(value)
        byte = min(255, int(round(mant * 256.0)))
        pixel = bytes((byte, byte, byte, exp2 + 128))
        body = pixel * (width * height)
        header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n" % (height, width)
        return header + body

    def test_dome_lighting_constant_environment_matches_albedo(self) -> None:
        """Verify a constant dome lights a diffuse surface uniformly to albedo*env.

        A constant environment produces constant diffuse irradiance for every
        normal (the SH ``E(n)=piL`` identity), so every lit pixel equals the
        linear albedo scaled by the environment value.
        """
        width, height = 16, 16
        albedo_srgb = (0.25, 0.5, 0.75)
        env_value = 0.8
        model = self._shaded_sphere_model(color=albedo_srgb)
        camera = self._camera_with_model(model)
        camera.set_dome_light(np.full((64, 128, 3), env_value, np.float32), intensity=1.0)

        hdr = camera.create_hdr_color_image_output(1, width, height)
        shape_index = camera.create_shape_index_image_output(1, width, height)
        camera.update(
            model.state(),
            self._identity_transforms(1),
            self._rays(width, height),
            hdr_color_image=hdr,
            shape_index_image=shape_index,
        )

        hit = shape_index.numpy()[0] != 0xFFFFFFFF
        lit = hdr.numpy()[0][hit]
        self.assertGreater(lit.shape[0], 0)
        srgb = np.array(albedo_srgb, dtype=np.float32)
        albedo_linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
        expected = np.broadcast_to(albedo_linear * env_value, lit.shape)
        np.testing.assert_allclose(lit, expected, atol=2.0e-2)

    def test_set_dome_light_color_matches_uniform_array(self) -> None:
        """Verify set_dome_light_color renders identically to an equivalent uniform image."""
        width, height = 16, 16
        color = (0.2, 0.4, 0.6)
        model = self._shaded_sphere_model()

        camera_color = self._camera_with_model(model)
        camera_color.set_dome_light_color(color, intensity=1.5)
        hdr_color = camera_color.create_hdr_color_image_output(1, width, height)
        camera_color.update(
            model.state(), self._identity_transforms(1), self._rays(width, height), hdr_color_image=hdr_color
        )

        camera_array = self._camera_with_model(model)
        camera_array.set_dome_light(np.full((1, 1, 3), color, np.float32), intensity=1.5)
        hdr_array = camera_array.create_hdr_color_image_output(1, width, height)
        camera_array.update(
            model.state(), self._identity_transforms(1), self._rays(width, height), hdr_color_image=hdr_array
        )

        np.testing.assert_allclose(hdr_color.numpy(), hdr_array.numpy())

    def test_dome_lighting_directional_irradiance_favors_facing_normals(self) -> None:
        """Verify SH dome irradiance is greater for normals facing the bright hemisphere."""
        env = np.zeros((32, 64, 3), dtype=np.float32)
        env[:16, :, :] = 3.0  # bright upper (+up) hemisphere
        sh = dome.compute_dome_sh9(env, newton.Axis.Z)
        facing = dome.eval_dome_irradiance_np(sh, (0.0, 0.0, 1.0))
        away = dome.eval_dome_irradiance_np(sh, (0.0, 0.0, -1.0))
        self.assertTrue(np.all(facing > away))
        self.assertTrue(np.all(away >= 0.0))

    def test_dome_lighting_differs_from_fixed_ambient(self) -> None:
        """Verify enabling the dome changes shading versus the fixed hemispheric ambient."""
        width, height = 16, 16
        model = self._shaded_sphere_model(color=(0.4, 0.6, 0.8))
        rays = self._rays(width, height)

        def render(dome_env) -> np.ndarray:
            camera = self._camera_with_model(model)
            if dome_env is not None:
                camera.set_dome_light(dome_env, intensity=1.0)
            hdr = camera.create_hdr_color_image_output(1, width, height)
            camera.update(model.state(), self._identity_transforms(1), rays, hdr_color_image=hdr)
            return hdr.numpy()

        env = np.zeros((32, 64, 3), dtype=np.float32)
        env[:16, :, :] = 2.0
        self.assertFalse(np.allclose(render(None), render(env)))

    def test_dome_shadow_sampling_requires_environment(self) -> None:
        """Verify shadowed dome sampling without a dome environment raises a clear error."""
        width, height = 4, 4
        model, camera = self._build_sphere_scene()
        camera.default_render_config = SensorCamera.RenderConfig(enable_dome_lighting=True, dome_shadow_samples=4)
        hdr = camera.create_hdr_color_image_output(1, width, height)
        with self.assertRaises(RuntimeError):
            camera.update(model.state(), self._identity_transforms(1), self._rays(width, height), hdr_color_image=hdr)

    def test_dome_background_disabled_by_default(self) -> None:
        """Verify dome background display is off by default in RenderConfig."""
        self.assertFalse(SensorCamera.RenderConfig().enable_dome_background)

    def test_dome_background_requires_environment(self) -> None:
        """Verify enabling the dome background without a dome environment raises a clear error."""
        width, height = 4, 4
        model, camera = self._build_sphere_scene()
        camera.default_render_config = SensorCamera.RenderConfig(enable_dome_lighting=True, enable_dome_background=True)
        hdr = camera.create_hdr_color_image_output(1, width, height)
        with self.assertRaises(RuntimeError):
            camera.update(model.state(), self._identity_transforms(1), self._rays(width, height), hdr_color_image=hdr)

    def test_dome_background_shows_environment_on_miss(self) -> None:
        """Verify camera rays that miss all geometry sample the HDRI dome as background."""
        width, height = 16, 16
        model = self._shaded_sphere_model(color=(0.6, 0.6, 0.6))
        background_color = (0.2, 0.5, 0.9)
        env = np.full((32, 64, 3), background_color, dtype=np.float32)

        camera = self._camera_with_model(model)
        camera.set_dome_light(env)
        camera.default_render_config.enable_dome_background = True
        hdr = camera.create_hdr_color_image_output(1, width, height)
        shape_index = camera.create_shape_index_image_output(1, width, height)
        # Wide FOV so the sphere (half-angle ~27 deg) does not fill the frame.
        camera.update(
            model.state(),
            self._identity_transforms(1),
            self._rays(width, height, fov=math.radians(90.0)),
            hdr_color_image=hdr,
            shape_index_image=shape_index,
        )

        hdr_np = hdr.numpy()[0]
        missed = shape_index.numpy()[0] == 0xFFFFFFFF
        self.assertTrue(np.any(missed))
        self.assertTrue(np.any(~missed))
        expected = np.broadcast_to(background_color, hdr_np[missed].shape)
        np.testing.assert_allclose(hdr_np[missed], expected, atol=1.0e-3)

    def test_dome_background_does_not_affect_geometry_shading(self) -> None:
        """Verify enabling the dome background leaves shaded (hit) pixels unchanged."""
        width, height = 16, 16
        model = self._shaded_sphere_model(color=(0.6, 0.6, 0.6))
        env = np.full((32, 64, 3), (0.2, 0.5, 0.9), dtype=np.float32)
        rays = self._rays(width, height, fov=math.radians(90.0))

        def render(enable_background: bool) -> tuple[np.ndarray, np.ndarray]:
            camera = self._camera_with_model(model)
            camera.set_dome_light(env)
            camera.default_render_config.enable_dome_background = enable_background
            hdr = camera.create_hdr_color_image_output(1, width, height)
            shape_index = camera.create_shape_index_image_output(1, width, height)
            camera.update(
                model.state(), self._identity_transforms(1), rays, hdr_color_image=hdr, shape_index_image=shape_index
            )
            return hdr.numpy()[0], shape_index.numpy()[0]

        hdr_off, shape_index_off = render(False)
        hdr_on, shape_index_on = render(True)
        np.testing.assert_array_equal(shape_index_off, shape_index_on)
        hit = shape_index_off != 0xFFFFFFFF
        np.testing.assert_allclose(hdr_off[hit], hdr_on[hit])
        self.assertFalse(np.allclose(hdr_off[~hit], hdr_on[~hit]))

    def test_set_dome_light_from_hdr_file(self) -> None:
        """Verify a Radiance ``.hdr`` file loads and illuminates the scene."""
        width, height = 8, 8
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "env.hdr")
            with open(path, "wb") as handle:
                handle.write(self._encode_flat_hdr(0.5, 8, 16))

            decoded = load_hdr_image(path)
            self.assertEqual(decoded.shape[2], 3)
            self.assertTrue(np.isfinite(decoded).all())
            self.assertTrue((decoded >= 0.0).all())
            np.testing.assert_allclose(decoded, 0.5, atol=1.0e-2)

            model = self._shaded_sphere_model()
            camera = self._camera_with_model(model)
            camera.set_dome_light(path, intensity=1.0)
            hdr = camera.create_hdr_color_image_output(1, width, height)
            shape_index = camera.create_shape_index_image_output(1, width, height)
            camera.update(
                model.state(),
                self._identity_transforms(1),
                self._rays(width, height),
                hdr_color_image=hdr,
                shape_index_image=shape_index,
            )
            hit = shape_index.numpy()[0] != 0xFFFFFFFF
            self.assertGreater(float(hdr.numpy()[0][hit].max()), 0.0)

    @staticmethod
    def _sphere_with_occluder_model() -> newton.Model:
        """A target sphere (shape 0) with a large occluder sphere above it (+up)."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        target = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(target, radius=1.0, color=(0.6, 0.6, 0.6))
        occluder = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 2.4, -2.0), q=wp.quat_identity()))
        builder.add_shape_sphere(occluder, radius=1.6, color=(0.6, 0.6, 0.6))
        return builder.finalize(device="cpu")

    def _render_dome_hdr(self, model, env, samples: int, width: int = 24, height: int = 24):
        """Render ``model`` under a dome ``env`` with ``samples`` shadow rays; return (hdr, shape_index)."""
        camera = self._camera_with_model(model)
        camera.set_dome_light(env)
        camera.default_render_config.dome_shadow_samples = samples
        hdr = camera.create_hdr_color_image_output(1, width, height)
        shape_index = camera.create_shape_index_image_output(1, width, height)
        camera.update(
            model.state(),
            self._identity_transforms(1),
            self._rays(width, height),
            hdr_color_image=hdr,
            shape_index_image=shape_index,
        )
        return hdr.numpy()[0], shape_index.numpy()[0]

    def test_dome_shadow_samples_defaults_to_zero(self) -> None:
        """Verify dome shadow sampling is off by default in RenderConfig."""
        self.assertEqual(SensorCamera.RenderConfig().dome_shadow_samples, 0)

    def test_dome_shadow_sampling_varies_between_frames(self) -> None:
        """Verify shadowed dome sampling is stochastic and reseeds each frame.

        Directions are drawn from a per-frame RNG, so successive renders of the
        same scene under a non-uniform environment differ (Monte-Carlo noise that
        converges under temporal accumulation).
        """
        model = self._shaded_sphere_model(color=(0.6, 0.6, 0.6))
        env = np.zeros((64, 128, 3), dtype=np.float32)
        env[:32] = (1.0, 0.9, 0.8)  # non-uniform, so the estimate carries variance
        env[32:] = 0.05
        camera = self._camera_with_model(model)
        camera.set_dome_light(env)
        camera.default_render_config.dome_shadow_samples = 8

        def render_once() -> np.ndarray:
            hdr = camera.create_hdr_color_image_output(1, 24, 24)
            camera.update(model.state(), self._identity_transforms(1), self._rays(24, 24), hdr_color_image=hdr)
            return hdr.numpy()

        # Same camera, so the per-frame seed advances between the two calls.
        self.assertFalse(np.array_equal(render_once(), render_once()))

    def test_dome_shadow_occluder_darkens_surface(self) -> None:
        """Verify an occluder reduces dome illumination on the shadowed surface."""
        env = np.full((64, 128, 3), 1.0, dtype=np.float32)
        open_hdr, open_idx = self._render_dome_hdr(self._shaded_sphere_model(color=(0.6, 0.6, 0.6)), env, samples=32)
        occ_hdr, occ_idx = self._render_dome_hdr(self._sphere_with_occluder_model(), env, samples=32)
        open_mean = open_hdr[open_idx == 0].mean()
        occ_mean = occ_hdr[occ_idx == 0].mean()
        self.assertLess(float(occ_mean), float(open_mean))

    def test_dome_shadow_unoccluded_matches_unshadowed(self) -> None:
        """Verify an unoccluded convex surface matches the SH result on average.

        A convex surface never self-occludes, so the importance-sampled estimate
        recovers the same irradiance as the analytic SH path. The per-pixel
        estimate is stochastic, so this compares the mean over the surface.
        """
        model = self._shaded_sphere_model(color=(0.6, 0.6, 0.6))
        env = np.full((64, 128, 3), 0.8, dtype=np.float32)
        shadowed, idx = self._render_dome_hdr(model, env, samples=128)
        unshadowed, _ = self._render_dome_hdr(model, env, samples=0)
        np.testing.assert_allclose(shadowed[idx == 0].mean(), unshadowed[idx == 0].mean(), rtol=0.05)

    def test_dome_shadow_importance_sampling_bounds_concentrated_source(self) -> None:
        """Verify importance sampling keeps a tiny bright source firefly-free.

        A concentrated 'sun' would produce extreme per-pixel spikes (and, after
        clipping, an overall-dark surface) under uniform sampling. Importance
        sampling aims rays at it, so single-frame values track the analytic mean
        with bounded maxima.
        """
        model = self._shaded_sphere_model(color=(0.6, 0.6, 0.6))
        env = np.full((64, 128, 3), 0.1, dtype=np.float32)
        env[:3, 60:66] = 2000.0  # tiny, very bright source
        shadowed, idx = self._render_dome_hdr(model, env, samples=64)
        unshadowed, _ = self._render_dome_hdr(model, env, samples=0)
        lit = shadowed[idx == 0]
        # No fireflies: the brightest sampled pixel stays close to the SH maximum.
        self.assertLess(float(lit.max()), 10.0 * float(unshadowed[idx == 0].max()))
        # Energy preserved: the surface is not left dark by undersampling the source.
        self.assertGreater(float(lit.mean()), 0.5 * float(unshadowed[idx == 0].mean()))

    # --- Utils to_rgba / flatten helpers (ported; new 3-D Utils) ---

    def test_utils_to_rgba_helpers_produce_canonical_outputs(self) -> None:
        """Verify the Utils to_rgba helpers return ``(view, H, W, 4)`` uint8 arrays."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        color = camera.create_color_image_output(view_count, width, height)
        depth = camera.create_depth_image_output(view_count, width, height)
        normal = camera.create_normal_image_output(view_count, width, height)
        shape_index = camera.create_shape_index_image_output(view_count, width, height)
        camera.update(
            state,
            self._identity_transforms(view_count),
            rays,
            color_image=color,
            depth_image=depth,
            normal_image=normal,
            shape_index_image=shape_index,
        )

        utils = camera.utils(view_count)
        for rgba in (
            utils.to_rgba_from_color(color),
            utils.to_rgba_from_depth(depth, depth_range=(0.0, 10.0)),
            utils.to_rgba_from_normal(normal),
            utils.to_rgba_from_shape_index(shape_index),
        ):
            self.assertEqual(rgba.shape, (view_count, height, width, 4))
            self.assertEqual(rgba.dtype, wp.uint8)

    def test_utils_postprocessing_helpers(self) -> None:
        """Verify forward-depth conversion, normal/depth flatten, palette colorize, and depth-range branches."""
        width, height, worlds_per_row = 6, 4, 2
        model, camera = self._build_sphere_scene(world_count=4)
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        camera_transforms = self._identity_transforms(view_count)
        depth = camera.create_depth_image_output(view_count, width, height)
        normal = camera.create_normal_image_output(view_count, width, height)
        shape_index = camera.create_shape_index_image_output(view_count, width, height)
        camera.update(
            state, camera_transforms, rays, depth_image=depth, normal_image=normal, shape_index_image=shape_index
        )

        utils = camera.utils(view_count)
        center = (0, height // 2, width // 2)
        self.assertGreater(float(depth.numpy()[center]), 0.0)

        # Ray-distance depth -> forward (planar) depth; must not exceed ray depth.
        forward = utils.convert_ray_depth_to_forward_depth(depth, camera_transforms, rays)
        self.assertEqual(forward.shape, depth.shape)
        self.assertEqual(forward.dtype, wp.float32)
        self.assertLessEqual(float(forward.numpy()[center]), float(depth.numpy()[center]) + 1.0e-4)

        # Flatten normal/depth into one tiled (rows*H, cols*W, 4) grid buffer.
        worlds_per_col = -(-view_count // worlds_per_row)
        for flat in (
            utils.flatten_normal_image_to_rgba(normal, worlds_per_row=worlds_per_row),
            utils.flatten_depth_image_to_rgba(depth, worlds_per_row=worlds_per_row),
        ):
            self.assertEqual(flat.shape, (worlds_per_col * height, worlds_per_row * width, 4))
            self.assertEqual(flat.dtype, wp.uint8)

        # Shape-index colorized via a caller palette (out-of-range indices -> black).
        palette = wp.array(np.array([[10, 20, 30]], dtype=np.uint8), dtype=wp.uint8, device="cpu")
        colored = utils.to_rgba_from_shape_index(shape_index, colors=palette)
        self.assertEqual(colored.shape, (view_count, height, width, 4))

        # to_rgba_from_depth: on-device auto range (depth_range=None) and the near<far guard.
        auto = utils.to_rgba_from_depth(depth)
        self.assertEqual(auto.shape, (view_count, height, width, 4))
        with self.assertRaisesRegex(ValueError, "near < far"):
            utils.to_rgba_from_depth(depth, depth_range=(5.0, 1.0))

    def test_utils_shape_index_hash_colors_differ_by_index(self) -> None:
        """Verify the shape-index hash palette assigns distinct colors (uint32 hash)."""
        width, height = 8, 6
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        shape_index = camera.create_shape_index_image_output(view_count, width, height)
        camera.update(state, self._identity_transforms(view_count), rays, shape_index_image=shape_index)
        rgba = camera.utils(view_count).to_rgba_from_shape_index(shape_index).numpy()
        colors = {tuple(c) for c in rgba.reshape(-1, 4)[:, :3]}
        self.assertGreater(len(colors), 1)

    def test_utils_flatten_rejects_worlds_per_row_below_one(self) -> None:
        """Verify the flatten helpers reject a non-positive ``worlds_per_row``."""
        width, height = 4, 3
        model, camera = self._build_sphere_scene()
        state = model.state()
        rays = self._rays(width, height)
        view_count = model.world_count
        color = camera.create_color_image_output(view_count, width, height)
        camera.update(state, self._identity_transforms(view_count), rays, color_image=color)
        with self.assertRaisesRegex(ValueError, "worlds_per_row"):
            camera.utils(view_count).flatten_color_image_to_rgba(color, worlds_per_row=0)


if __name__ == "__main__":
    unittest.main()
