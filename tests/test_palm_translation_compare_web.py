"""Backend tests for the palm-translation comparison Web UI."""

from __future__ import annotations

import http.client
import json
import os
from io import BytesIO
import threading
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image
from http.server import HTTPServer

from scripts.palm_translation_compare_web import PalmCompareBoard, make_handler


class PalmCompareBoardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = PalmCompareBoard(width=160, height=120)

    @classmethod
    def tearDownClass(cls):
        cls.board.close()

    def test_native_delta_is_about_52mm_and_orientation_matches(self):
        state = self.board.state()
        self.assertAlmostEqual(state["native_delta_norm_mm"], 52.02, places=1)
        my = state["scenes"]["my_grasp"]["native_mm"]
        pd = state["scenes"]["palm_down"]["native_mm"]
        self.assertAlmostEqual(my[0] - pd[0], 45.0, delta=0.05)
        self.assertAlmostEqual(my[1] - pd[1], -8.0, delta=0.1)
        self.assertAlmostEqual(my[2] - pd[2], 24.87, delta=0.05)

    def test_copy_my_grasp_onto_palm_down_zeroes_live_delta(self):
        self.board.reset_native()
        copied = self.board.copy_translation("my_grasp", "palm_down")
        self.assertLess(copied["live_delta_norm_mm"], 1e-6)
        native = self.board.reset_native()
        self.assertGreater(native["live_delta_norm_mm"], 50.0)

    def test_blend_endpoints(self):
        start = self.board.set_blend(0.0)
        self.assertAlmostEqual(start["blend"], 0.0)
        self.assertEqual(
            start["scenes"]["palm_down"]["translation_mm"],
            start["scenes"]["palm_down"]["native_mm"],
        )
        end = self.board.set_blend(1.0)
        for a, b in zip(
            end["scenes"]["palm_down"]["translation_mm"],
            end["scenes"]["my_grasp"]["native_mm"],
        ):
            self.assertAlmostEqual(a, b, places=6)
        self.board.reset_native()

    def test_render_png_and_jpeg_and_camera_move(self):
        png = self.board.render_png()
        jpeg = self.board.render_jpeg()
        self.assertGreater(len(png), 1000)
        self.assertGreater(len(jpeg), 500)
        image = Image.open(BytesIO(png))
        image.load()
        self.assertEqual(image.size, (160 * 2 + 4, 120))
        self.assertGreater(float(np.mean(np.asarray(image))), 5.0)
        before = self.board.state()["camera"]["azimuth"]
        moved = self.board.move_camera("rotate", 0.2, 0.0)
        self.assertNotAlmostEqual(moved["camera"]["azimuth"], before, places=3)
        self.board.reset_camera()

    def test_media_files_exist(self):
        for url in (
            "/media/my_grasp_revolute_s1.mp4",
            "/media/palm_down_60s.mp4",
            "/media/palm_down_20s.mp4",
            "/media/my_grasp_t00.mp4",
        ):
            data, content_type = self.board.media_bytes(url)
            self.assertEqual(content_type, "video/mp4")
            self.assertGreater(len(data), 10_000)


class PalmCompareApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = PalmCompareBoard(width=160, height=120)
        cls.server = HTTPServer(("127.0.0.1", 0), make_handler(cls.board))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.board.close()

    def request(self, method, path, payload=None, headers=None, timeout=20):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        hdrs = {"Content-Type": "application/json"} if body is not None else {}
        if headers:
            hdrs.update(headers)
        connection.request(method, path, body=body, headers=hdrs)
        response = connection.getresponse()
        content = response.read()
        result = (response.status, response.getheader("Content-Type"), content)
        connection.close()
        return result

    def test_page_state_render_video_and_camera(self):
        status, content_type, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"Palm translation", body)
        self.assertIn(b"Drag orbit", body)

        status, _, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        state = json.loads(body)
        self.assertIn("my_grasp", state["scenes"])
        self.assertIn("palm_down", state["scenes"])

        status, content_type, body = self.request("GET", "/api/render.jpg")
        self.assertEqual(status, 200)
        self.assertIn("image/jpeg", content_type)
        image = Image.open(BytesIO(body))
        image.load()
        self.assertEqual(image.format, "JPEG")

        before = json.loads(self.request("GET", "/api/state")[2])["camera"]["distance"]
        status, _, body = self.request(
            "POST", "/api/camera/move", {"action": "zoom", "dx": 0.0, "dy": 0.15}
        )
        self.assertEqual(status, 200)
        self.assertNotAlmostEqual(json.loads(body)["camera"]["distance"], before, places=4)

        status, content_type, body = self.request(
            "GET",
            "/media/palm_down_60s.mp4",
            headers={"Range": "bytes=0-15"},
        )
        self.assertEqual(status, 206)
        self.assertEqual(content_type, "video/mp4")
        self.assertEqual(len(body), 16)


if __name__ == "__main__":
    unittest.main()
