import http.client
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

from PIL import Image

from scripts.edit_hand_pose_web import POSE_ROOT, PoseEditor, make_handler
from http.server import HTTPServer


class HandPoseWebBackendTest(unittest.TestCase):
    def setUp(self):
        POSE_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=POSE_ROOT)
        self.directory = Path(self.temp.name)
        self.output = self.directory / "candidate.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_initial_update_reset_render_save_load_and_overwrite(self):
        editor = PoseEditor("revolute", output_path=self.output, width=320, height=240)
        try:
            initial = editor.state()
            self.assertEqual(initial["physics"], "revolute")
            self.assertEqual(len(initial["translation_mm"]), 3)
            self.assertEqual(len(initial["quaternion_wxyz"]), 4)
            self.assertIn("palm_clearance_mm", initial["metrics"])

            moved = editor.update_pose([12.0, -34.0, 56.0], [10.0, 20.0, 30.0])
            self.assertEqual(moved["translation_mm"], [12.0, -34.0, 56.0])
            for actual, expected in zip(moved["euler_deg"], [10.0, 20.0, 30.0]):
                self.assertAlmostEqual(actual, expected, places=8)

            png = editor.render_png()
            self.assertGreater(len(png), 1000)
            image = Image.open(BytesIO(png))
            image.load()
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (320, 240))

            saved = editor.save(str(self.output), overwrite=False)
            self.assertEqual(len(saved["sha256"]), 64)
            self.assertEqual(saved["content"]["schema_version"], 1)
            with self.assertRaises(FileExistsError):
                editor.save(str(self.output), overwrite=False)
            editor.save(str(self.output), overwrite=True)

            editor.reset()
            self.assertNotEqual(editor.state()["translation_mm"], [12.0, -34.0, 56.0])
            loaded = editor.load(str(self.output))
            self.assertEqual(loaded["translation_mm"], [12.0, -34.0, 56.0])
            editor.update_pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
            reset = editor.reset()
            self.assertEqual(reset["translation_mm"], [12.0, -34.0, 56.0])
        finally:
            editor.close()

    def test_tip_connect_variant_settle_and_render(self):
        editor = PoseEditor(
            "tip_connect", tip_anchor="bottom", output_path=self.output, width=200, height=150
        )
        try:
            state = editor.settle(5)
            self.assertEqual(state["physics"], "tip_connect")
            self.assertEqual(state["metrics"]["settle_steps"], 5)
            self.assertEqual(len(state["metrics"]["contact_forces_n"]), 3)
            image = Image.open(BytesIO(editor.render_png()))
            image.load()
            self.assertEqual(image.size, (200, 150))
        finally:
            editor.close()

    def test_pose_validation_and_path_confinement(self):
        editor = PoseEditor("revolute", output_path=self.output, width=160, height=120)
        try:
            with self.assertRaisesRegex(ValueError, "3 finite"):
                editor.update_pose([1, 2], [0, 0, 0])
            with self.assertRaisesRegex(ValueError, "inside"):
                editor.save("/tmp/escaped.json")
            with self.assertRaisesRegex(ValueError, "end in .json"):
                editor.save(str(self.directory / "pose.txt"))
            malformed = self.directory / "malformed.json"
            malformed.write_text("{bad", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid hand pose JSON"):
                editor.load(str(malformed))
        finally:
            editor.close()


class HandPoseWebApiTest(unittest.TestCase):
    def setUp(self):
        POSE_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=POSE_ROOT)
        self.directory = Path(self.temp.name)
        self.editor = PoseEditor(
            "revolute",
            output_path=self.directory / "api_pose.json",
            width=160,
            height=120,
        )
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(self.editor))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.editor.close()
        self.temp.cleanup()

    def request(self, method, path, payload=None, raw=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )
        body = raw if raw is not None else (
            json.dumps(payload).encode("utf-8") if payload is not None else None
        )
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        content = response.read()
        result = (response.status, response.getheader("Content-Type"), content)
        connection.close()
        return result

    def test_api_state_update_reset_render_and_errors(self):
        status, content_type, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(b"Allegro Hand Pose Studio", body)
        self.assertIn(b"Settle + recompute contacts", body)

        status, _, body = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        state = json.loads(body)
        self.assertEqual(state["physics"], "revolute")

        status, _, body = self.request(
            "POST",
            "/api/pose",
            {"translation_mm": [4, 5, 6], "euler_deg": [1, 2, 3]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["translation_mm"], [4.0, 5.0, 6.0])

        status, content_type, body = self.request("GET", "/api/render.png")
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "image/png")
        Image.open(BytesIO(body)).verify()

        status, _, body = self.request("POST", "/api/pose", raw=b"{bad")
        self.assertEqual(status, 400)
        self.assertIn("valid JSON", json.loads(body)["error"])

        status, _, body = self.request(
            "POST", "/api/save", {"path": "../../escape.json", "overwrite": False}
        )
        self.assertEqual(status, 400)
        self.assertIn("inside", json.loads(body)["error"])

        api_pose = self.directory / "api_saved.json"
        status, _, body = self.request(
            "POST", "/api/save", {"path": str(api_pose), "overwrite": False}
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["sha256"]), 64)
        status, _, _ = self.request(
            "POST", "/api/save", {"path": str(api_pose), "overwrite": False}
        )
        self.assertEqual(status, 400)
        status, _, body = self.request(
            "POST", "/api/load", {"path": str(api_pose)}
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["translation_mm"], [4.0, 5.0, 6.0])

        status, _, _ = self.request("POST", "/api/reset", {})
        self.assertEqual(status, 200)
        status, _, _ = self.request("GET", "/missing")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
