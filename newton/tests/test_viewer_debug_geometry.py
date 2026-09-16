# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.viewer.kernels import compute_arrow_lines
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices
from newton.viewer import ViewerRTX


class TestViewerDebugGeometry(unittest.TestCase):
    pass


def test_arrow_lines_handle_zero_and_nonzero_lengths(test: TestViewerDebugGeometry, device):
    """Expand arrows into stable shaft and arrowhead line segments."""
    starts = wp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=wp.vec3, device=device)
    ends = wp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 5.0]], dtype=wp.vec3, device=device)
    colors = wp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device=device)
    line_starts = wp.empty(10, dtype=wp.vec3, device=device)
    line_ends = wp.empty(10, dtype=wp.vec3, device=device)
    line_colors = wp.empty(10, dtype=wp.vec3, device=device)

    wp.launch(
        compute_arrow_lines,
        10,
        inputs=[starts, ends, colors, 0.3, 0.35],
        outputs=[line_starts, line_ends, line_colors],
        device=device,
    )

    result_starts = line_starts.numpy()
    result_ends = line_ends.numpy()
    test.assertTrue(np.isnan(result_starts[1:5]).all())
    test.assertTrue(np.isnan(result_ends[1:5]).all())
    assert_np_equal(result_starts[5], np.array([1.0, 2.0, 3.0]), tol=1.0e-6)
    assert_np_equal(result_ends[5], np.array([1.0, 2.0, 5.0]), tol=1.0e-6)
    assert_np_equal(result_starts[6:], np.tile([1.0, 2.0, 5.0], (4, 1)), tol=1.0e-6)
    test.assertTrue(np.isfinite(result_ends[6:]).all())


def test_viewer_rtx_log_arrows_expands_segments(test: TestViewerDebugGeometry, device):
    """Expand each RTX arrow into one shaft and four arrowhead fins."""
    viewer = ViewerRTX.__new__(ViewerRTX)
    viewer.device = device
    viewer._qualify = lambda name: name
    calls = []
    viewer.log_lines = lambda *args, **kwargs: calls.append((args, kwargs))
    starts = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
    ends = wp.array([[0.0, 0.0, 2.0]], dtype=wp.vec3, device=device)

    ViewerRTX.log_arrows(viewer, "/arrows", starts, ends, (0.0, 1.0, 0.0), width=0.02)

    test.assertEqual(len(calls), 1)
    args, kwargs = calls[0]
    test.assertEqual(args[0], "/arrows")
    test.assertEqual(len(args[1]), 5)
    test.assertEqual(len(args[2]), 5)
    test.assertEqual(len(args[3]), 5)
    test.assertEqual(kwargs["width"], 0.02)
    assert_np_equal(args[3].numpy(), np.tile([0.0, 1.0, 0.0], (5, 1)), tol=1.0e-6)


devices = get_test_devices()
add_function_test(
    TestViewerDebugGeometry,
    "test_arrow_lines_handle_zero_and_nonzero_lengths",
    test_arrow_lines_handle_zero_and_nonzero_lengths,
    devices=devices,
)
add_function_test(
    TestViewerDebugGeometry,
    "test_viewer_rtx_log_arrows_expands_segments",
    test_viewer_rtx_log_arrows_expands_segments,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
