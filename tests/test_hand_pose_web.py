import http.client
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

from PIL import Image

from scripts.edit_hand_pose_web import HTML, POSE_ROOT, PoseEditor, make_handler
from http.server import HTTPServer

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


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

    def test_pose_control_dom_contract_has_unique_valid_rows(self):
        block = re.search(r"const poseDefs=\[(.*?)\];", HTML, re.DOTALL)
        self.assertIsNotNone(block)
        definitions = re.findall(
            r"\{label:'([^']+)',key:'([^']+)',index:(\d+),"
            r"min:([-\d]+),max:([-\d]+),unit:",
            block.group(1),
        )
        self.assertEqual(
            [item[0] for item in definitions],
            ["X", "Y", "Z", "Roll", "Pitch", "Yaw"],
        )
        ids = [f"{key}-{index}" for _, key, index, _, _ in definitions]
        self.assertEqual(len(ids), len(set(ids)))
        for _, _, _, minimum, maximum in definitions:
            self.assertLess(float(minimum), float(maximum))
        self.assertIn(
            "def.index===undefined?def.key:`${def.key}-${def.index}`", HTML
        )
        self.assertIn('type="text" inputmode="decimal"', HTML)
        for event in ("'change'", "'blur'", "'keydown'"):
            self.assertIn(f"n.addEventListener({event}", HTML)


@unittest.skipUnless(
    os.environ.get("HAND_POSE_BROWSER_TESTS") == "1" and sync_playwright is not None,
    "set HAND_POSE_BROWSER_TESTS=1 with Playwright Chromium installed",
)
class HandPoseWebBrowserTest(unittest.TestCase):
    def setUp(self):
        POSE_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=POSE_ROOT)
        self.directory = Path(self.temp.name)
        self.editor = PoseEditor(
            "revolute",
            output_path=self.directory / "browser_output.json",
            width=240,
            height=180,
        )
        handler = make_handler(self.editor)
        handler.log_message = lambda *args: None
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page()
        self.console_errors = []
        self.page.on(
            "console",
            lambda message: self.console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.page.goto(self.url, wait_until="networkidle")

    def tearDown(self):
        self.browser.close()
        self.playwright.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.editor.close()
        self.temp.cleanup()

    def state(self):
        return self.page.evaluate("fetch('/api/state').then(r => r.json())")

    def render(self):
        return self.page.request.get(
            f"{self.url}/api/render.png?test={time.monotonic_ns()}"
        ).body()

    def wait_value(self, group, index, expected):
        self.page.wait_for_function(
            """({group,index,expected}) => fetch('/api/state')
              .then(r => r.json())
              .then(s => Math.abs(s[group][index] - expected) < 1e-6)""",
            arg={"group": group, "index": index, "expected": expected},
        )

    def test_all_pose_and_camera_controls_save_load_reset_and_errors(self):
        self.assertEqual(
            self.page.locator("#poseRows input[type=range]").evaluate_all(
                "els => new Set(els.map(el => el.id)).size"
            ),
            6,
        )
        self.assertEqual(
            self.page.locator("#poseRows input[type=text]").evaluate_all(
                "els => new Set(els.map(el => el.id)).size"
            ),
            6,
        )

        typed = [
            ("translation_mm", 0, "12.5", "Enter"),
            ("translation_mm", 1, "-23.75", "blur"),
            ("translation_mm", 2, "0", "change"),
            ("euler_deg", 0, "45.25", "Enter"),
            ("euler_deg", 1, "-30.5", "blur"),
            ("euler_deg", 2, "0", "change"),
        ]
        previous_render = self.render()
        for group, index, text, commit in typed:
            field = self.page.locator(f"#p-{group}-{index}-n")
            field.fill(text)
            if commit == "Enter":
                field.press("Enter")
            elif commit == "blur":
                field.blur()
            else:
                field.dispatch_event("change")
            expected = float(text)
            self.wait_value(group, index, expected)
            self.assertAlmostEqual(
                float(self.page.locator(f"#p-{group}-{index}-r").input_value()),
                expected,
            )
            current_render = self.render()
            if expected != 0 or self.state()[group][index] != 0:
                self.assertNotEqual(
                    hashlib.sha256(current_render).digest(),
                    hashlib.sha256(previous_render).digest(),
                )
            previous_render = current_render

        # Blank, sign, and decimal-prefix states remain editable until commit.
        field = self.page.locator("#p-translation_mm-0-n")
        for intermediate in ("", "-", ".", "-."):
            field.fill(intermediate)
            self.assertEqual(field.input_value(), intermediate)
        field.blur()
        self.assertIn("must be a complete number", self.page.locator("#status").inner_text())

        slider_values = [30, -35, 40, 55, -60, 65]
        for control, expected in zip(
            self.page.locator("#poseRows input[type=range]").all(), slider_values
        ):
            group = "translation_mm" if "translation" in control.get_attribute("id") else "euler_deg"
            index = int(control.get_attribute("id").split("-")[-2])
            before = self.render()
            control.evaluate(
                "(el,value) => {el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}))}",
                str(expected),
            )
            self.wait_value(group, index, expected)
            self.page.wait_for_timeout(150)
            self.wait_value(group, index, expected)
            self.assertEqual(
                self.page.locator(f"#{control.get_attribute('id')[:-1]}n").input_value(),
                str(expected),
            )
            self.assertNotEqual(
                hashlib.sha256(self.render()).digest(),
                hashlib.sha256(before).digest(),
                control.get_attribute("id"),
            )

        pose_ranges = self.page.locator("#poseRows input[type=range]")
        self.assertEqual(
            pose_ranges.evaluate_all("els => els.map(el => el.dataset.increment)"),
            ["1"] * 6,
        )
        x_range = self.page.locator("#p-translation_mm-0-r")
        x_range.press("ArrowUp")
        self.wait_value("translation_mm", 0, 31)
        self.page.locator("#stepMode").select_option("coarse")
        self.assertEqual(
            pose_ranges.evaluate_all("els => els.map(el => el.dataset.increment)"),
            ["10"] * 6,
        )
        x_range.press("ArrowUp")
        self.wait_value("translation_mm", 0, 41)
        self.page.locator("#stepMode").select_option("fine")

        # Delay the first response so it arrives after a newer update. The
        # stale response must not revert the newer slider/text value.
        self.page.evaluate(
            """() => {
              const realFetch=window.fetch.bind(window);let poseCalls=0;
              window.fetch=(url,options) => {
                const response=realFetch(url,options);
                if(String(url)==='/api/pose' && ++poseCalls===1)
                  return response.then(async r => {
                    await new Promise(resolve => setTimeout(resolve,300));return r;
                  });
                return response;
              };
            }"""
        )
        x_range.evaluate(
            "(el) => {el.value='70';el.dispatchEvent(new Event('input',{bubbles:true}))}"
        )
        self.page.wait_for_timeout(120)
        x_range.evaluate(
            "(el) => {el.value='80';el.dispatchEvent(new Event('input',{bubbles:true}))}"
        )
        self.wait_value("translation_mm", 0, 80)
        self.page.wait_for_timeout(350)
        self.assertEqual(x_range.input_value(), "80")
        self.assertEqual(self.page.locator("#p-translation_mm-0-n").input_value(), "80")
        self.wait_value("translation_mm", 0, 80)

        camera_values = {
            "c-azimuth-r": 160,
            "c-elevation-r": -30,
            "c-distance-r": 0.55,
            "c-lookat-0-r": 0.04,
            "c-lookat-1-r": -0.03,
            "c-lookat-2-r": 0.02,
        }
        for control_id, expected in camera_values.items():
            before = self.render()
            control = self.page.locator(f"#{control_id}")
            control.evaluate(
                "(el,value) => {el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}))}",
                str(expected),
            )
            key = control_id.removeprefix("c-").removesuffix("-r")
            if key.startswith("lookat-"):
                index = int(key[-1])
                self.page.wait_for_function(
                    """({index,expected}) => fetch('/api/state').then(r => r.json())
                      .then(s => Math.abs(s.camera.lookat[index] - expected) < 1e-6)""",
                    arg={"index": index, "expected": expected},
                )
            else:
                self.page.wait_for_function(
                    """({key,expected}) => fetch('/api/state').then(r => r.json())
                      .then(s => Math.abs(s.camera[key] - expected) < 1e-6)""",
                    arg={"key": key, "expected": expected},
                )
            self.page.wait_for_timeout(150)
            self.assertNotEqual(
                hashlib.sha256(self.render()).digest(),
                hashlib.sha256(before).digest(),
                control_id,
            )

        saved_path = self.directory / "browser_saved.json"
        self.page.locator("#path").fill(str(saved_path))
        x_range.evaluate(
            "(el) => {el.value='82';el.dispatchEvent(new Event('input',{bubbles:true}))}"
        )
        self.page.locator("#save").click()
        self.page.wait_for_function(
            "() => document.querySelector('#status').textContent.startsWith('Saved ')"
        )
        self.assertTrue(saved_path.is_file())
        self.assertAlmostEqual(
            json.loads(saved_path.read_text(encoding="utf-8"))["translation"][0],
            0.082,
        )

        self.page.locator("#p-translation_mm-0-n").fill("77")
        self.page.locator("#p-translation_mm-0-n").press("Enter")
        self.wait_value("translation_mm", 0, 77)
        self.page.locator("#path").fill(str(saved_path))
        self.page.locator("#load").click()
        self.wait_value("translation_mm", 0, 82)
        self.page.locator("#reset").click()
        self.wait_value("translation_mm", 0, 82)

        self.assertEqual(self.console_errors, [])
        self.page.locator("#path").fill(str(self.directory / "missing.json"))
        self.page.locator("#load").click()
        self.page.wait_for_function(
            "() => document.querySelector('#status').classList.contains('error')"
        )
        self.assertIn("does not exist", self.page.locator("#status").inner_text())


if __name__ == "__main__":
    unittest.main()
