from __future__ import annotations

import ctypes
import sys
import types
import unittest
from unittest import mock

from experiment_runner.experiments.gpu_recording import _egl_device_for_cuda_zero
from experiment_runner.gpu_safety import GpuSmokeSafetyError, issue_gpu_execution_permit


class EglDeviceMappingTests(unittest.TestCase):
    """Exercise EGL selection without importing a driver or creating a GL context."""

    def select(self, cuda_devices):
        permit = issue_gpu_execution_permit({
            "DET_EXPERIMENT_ID": "1", "DET_TRIAL_ID": "2", "DET_TASK_ID": "t",
            "DET_ALLOCATION_ID": "a", "DET_TASK_TYPE": "TRIAL", "DET_SLOT_IDS": "[0]",
            "NVIDIA_VISIBLE_DEVICES": "GPU-12345678-1234-1234-1234-123456789abc",
        })

        def enumerate_devices(size, devices, count):
            count._obj.value = len(cuda_devices)
            if devices is not None:
                for index in range(len(cuda_devices)):
                    devices[index] = index + 100
            return 1

        def query_string(device, attribute):
            self.assertEqual(attribute, 0x3055)
            return b"EGL_NV_device_cuda" if cuda_devices[device - 100] is not None else b""

        def query_attribute(device, attribute, output):
            self.assertEqual(attribute, 0x323A)
            output._obj.value = cuda_devices[device - 100]
            return 1

        egl_module = types.ModuleType("pyglet.libs.egl")
        egl_module.egl = types.SimpleNamespace(EGLint=ctypes.c_int, EGLBoolean=ctypes.c_uint)
        egl_module.eglext = types.SimpleNamespace(
            EGLDeviceEXT=ctypes.c_void_p, eglQueryDevicesEXT=enumerate_devices,
        )
        library = types.ModuleType("pyglet.libs.egl.lib")
        library.link_EGL = lambda name, *_: {
            "eglQueryDeviceStringEXT": query_string,
            "eglQueryDeviceAttribEXT": query_attribute,
        }[name]
        with mock.patch.dict(sys.modules, {"pyglet.libs.egl": egl_module, "pyglet.libs.egl.lib": library}):
            return _egl_device_for_cuda_zero(permit)

    def test_cuda_zero_is_mapped_instead_of_assuming_egl_zero(self):
        self.assertEqual(self.select([None, 1, 0]), (2, 102))

    def test_missing_ambiguous_and_software_only_devices_fail_closed(self):
        for devices in ([], [None], [1], [0, 0]):
            with self.subTest(devices=devices), self.assertRaises(RuntimeError):
                self.select(devices)

    def test_enumeration_requires_a_real_gate_permit(self):
        with self.assertRaises(GpuSmokeSafetyError):
            _egl_device_for_cuda_zero(None)

    def test_mapping_failure_reports_which_condition_prevented_selection(self):
        cases = (
            ([None], "no_cuda_device_extension"),
            ([3], "no_logical_cuda_zero"),
            ([0, 0], "ambiguous_logical_cuda_zero"),
        )
        for devices, reason in cases:
            with self.subTest(devices=devices), self.assertRaises(RuntimeError) as caught:
                self.select(devices)
            evidence = caught.exception.diagnostics
            self.assertEqual(evidence["reason_code"], reason)
            self.assertEqual(evidence["egl_device_count"], len(devices))
            self.assertEqual(len(evidence["devices"]), len(devices))
            self.assertIn(reason, str(caught.exception))

    def test_nonzero_cuda_handle_is_reported_without_selecting_it(self):
        with self.assertRaises(RuntimeError) as caught:
            self.select([None, 3])
        evidence = caught.exception.diagnostics
        self.assertFalse(evidence["devices"][0]["supports_cuda_mapping"])
        self.assertEqual(evidence["devices"][1]["cuda_device"], 3)


if __name__ == "__main__":
    unittest.main()
