"""Tests for the SatNav API health endpoint."""

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from fastapi.testclient import TestClient

from app.adapters.backend_health_checker import BackendHealthChecker
from app.adapters.flight_control_auth_client import (
    FlightControlAuthClient,
    FlightControlAuthError,
)
from app.adapters.flight_control_device_registry import FlightControlDeviceRegistry
from app.adapters.flight_control_flight_client import FlightControlFlightClient
from app.adapters.rtmp_stream_checker import RtmpStreamChecker
from app.adapters.swiftvln_deploy_client import SwiftVLNDeployClient
import app.main as main_module
from app.main import SERVICE_NAME, SERVICE_VERSION, app, model_client
from app.services.model_inference_service import ModelInferenceService

# 仅用于单元测试注入，不代表任何部署默认值。
TEST_RTMP_URL = "rtmp://unit-test.invalid/live/satnav"


class HealthEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rtmp_env_patch = patch.dict(
            os.environ,
            {"SATNAV_RTMP_URL": TEST_RTMP_URL},
            clear=False,
        )
        self.rtmp_env_patch.start()
        self.rtmp_start_mock = patch.object(RtmpStreamChecker, "start").start()
        self.rtmp_close_mock = patch.object(RtmpStreamChecker, "close").start()
        self.start_mock = patch.object(
            model_client, "start_in_background"
        ).start()
        self.close_mock = patch.object(model_client, "close").start()
        self.status_mock = patch.object(
            model_client,
            "status",
            return_value={
                "state": "ready",
                "loaded": True,
                "process_running": True,
                "model_name": "test-model",
                "checkpoint_path": "/tmp/test-checkpoint",
                "gpu": {"cuda_visible_devices": "0"},
                "error": None,
                "started_at": "2026-07-31T00:00:00+00:00",
                "ready_at": "2026-07-31T00:00:01+00:00",
            },
        ).start()
        self.logs_mock = patch.object(
            model_client,
            "logs",
            return_value={
                "logs": [
                    {
                        "sequence": 1,
                        "timestamp": "2026-07-31T00:00:00+00:00",
                        "source": "swiftvln",
                        "stream": "stderr",
                        "level": "info",
                        "message": "Loading checkpoint shards",
                    }
                ],
                "latest_sequence": 1,
                "has_more": False,
            },
        ).start()
        self.addCleanup(self.rtmp_env_patch.stop)
        self.addCleanup(patch.stopall)

    def test_health_returns_service_metadata(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/satnav/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], SERVICE_NAME)
        self.assertEqual(payload["version"], SERVICE_VERSION)
        self.assertIn("timestamp", payload)

    def test_model_status_reports_loaded_model(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/satnav/system/model/status")
            self.start_mock.assert_called_once_with()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["model"]["loaded"])
        self.assertEqual(payload["model"]["state"], "ready")
        self.close_mock.assert_called_once_with()

    def test_model_logs_returns_incremental_model_logs(self) -> None:
        with TestClient(app) as client:
            response = client.get(
                "/api/satnav/system/model/logs",
                params={"after_sequence": 0, "limit": 100},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["logs"][0]["source"], "swiftvln")
        self.assertEqual(payload["latest_sequence"], 1)
        self.logs_mock.assert_called_once_with(after_sequence=0, limit=100)

    def test_rtmp_status_reports_active_stream(self) -> None:
        with TestClient(app) as client:
            checker = main_module.rtmp_checker
            self.assertIsNotNone(checker)
            with patch.object(
                checker,
                "check",
                return_value={
                    "rtmp_url": TEST_RTMP_URL,
                    "connected": True,
                    "stream_active": True,
                    "probe_method": "opencv",
                    "video": {"codec": "unknown", "width": 1920, "height": 1080},
                    "checked_at": "2026-07-31T00:00:00+00:00",
                },
            ) as check_mock:
                response = client.get("/api/satnav/system/rtmp/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["stream_active"])
        self.assertEqual(payload["video"]["width"], 1920)
        check_mock.assert_called_once_with()

    def test_backend_status_requires_configuration(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/satnav/system/flight-rc-backend/status")
        self.assertEqual(response.status_code, 503)

    def test_backend_status_reports_probe_result(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SATNAV_RTMP_URL": TEST_RTMP_URL,
                "SATNAV_BACKEND_HOST": "127.0.0.1",
                "SATNAV_BACKEND_PORT": "6789",
            },
            clear=False,
        ):
            with TestClient(app) as client:
                checker = main_module.backend_health_checker
                self.assertIsNotNone(checker)
                with patch.object(
                    checker,
                    "check",
                    return_value={
                        "backend_host": "127.0.0.1",
                        "backend_port": 6789,
                        "api_docs_path": "/v3/api-docs",
                        "healthy": True,
                        "probe_method": "api_docs",
                        "status_code": 200,
                        "service_title": "CloudSDK API",
                        "checked_at": "2026-07-31T00:00:00+00:00",
                    },
                ) as check_mock:
                    response = client.get("/api/satnav/system/flight-rc-backend/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["healthy"])
        check_mock.assert_called_once_with()

    def test_flight_rc_backend_login_requires_credentials_without_cached_token(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SATNAV_RTMP_URL": TEST_RTMP_URL,
                "SATNAV_BACKEND_HOST": "127.0.0.1",
                "SATNAV_BACKEND_PORT": "6789",
            },
            clear=False,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/api/satnav/system/flight-rc-backend/login",
                    json={},
                )
        self.assertEqual(response.status_code, 400)
        self.assertIn("username and password", response.json()["detail"])

    def test_flight_rc_backend_login_returns_backend_payload(self) -> None:
        backend_response = {
            "code": 0,
            "message": "success",
            "data": {
                "user_id": "user-1",
                "username": "adminPC",
                "workspace_id": "ws-1",
                "user_type": 1,
                "mqtt_username": "admin",
                "mqtt_addr": "mqtt://host:1883",
                "access_token": "token-abc",
            },
        }
        with patch.dict(
            os.environ,
            {
                "SATNAV_RTMP_URL": TEST_RTMP_URL,
                "SATNAV_BACKEND_HOST": "127.0.0.1",
                "SATNAV_BACKEND_PORT": "6789",
            },
            clear=False,
        ):
            with TestClient(app) as client:
                auth_client = main_module.flight_control_auth
                self.assertIsNotNone(auth_client)
                with patch.object(
                    auth_client,
                    "login_or_refresh",
                    return_value={
                        "auth_method": "login",
                        "logged_in": True,
                        "backend": backend_response,
                        "timestamp": "2026-07-31T00:00:00+00:00",
                    },
                ) as login_mock:
                    login_response = client.post(
                        "/api/satnav/system/flight-rc-backend/login",
                        json={
                            "username": "adminPC",
                            "password": "adminPC",
                            "flag": 1,
                        },
                    )

        self.assertEqual(login_response.status_code, 200)
        self.assertEqual(login_response.json()["auth_method"], "login")
        login_mock.assert_called_once_with(
            username="adminPC",
            password="adminPC",
            flag=1,
        )

    def test_model_inference_returns_service_payload(self) -> None:
        expected = {
            "instruction": "fly forward",
            "session_id": "satnav-infer-abc",
            "started_new_session": True,
            "performed_inference": True,
            "raw_action_text": "1 2 3 4",
            "actions": [1, 2, 3, 4],
            "next_action": 1,
            "remaining_actions": [1, 2, 3, 4],
            "completed_action": None,
            "deploy_state": "waiting_feedback",
            "frame_sequence": 7,
            "image_path": "/tmp/infer.jpg",
            "timing": {"capture_ms": 12.0},
            "timestamp": "2026-07-31T00:00:00+00:00",
        }
        with patch.object(
            ModelInferenceService,
            "from_environment",
        ) as factory_mock:
            service_mock = factory_mock.return_value
            service_mock.run.return_value = expected
            with TestClient(app) as client:
                response = client.post(
                    "/api/satnav/model/inference",
                    json={"instruction": "fly forward"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        service_mock.run.assert_called_once_with("fly forward")

    def test_model_inference_rejects_empty_instruction(self) -> None:
        with TestClient(app) as client:
            response = client.post(
                "/api/satnav/model/inference",
                json={"instruction": ""},
            )
        self.assertEqual(response.status_code, 422)

    def test_model_inference_maps_service_errors(self) -> None:
        from app.services.model_inference_service import ModelInferenceError

        with patch.object(
            ModelInferenceService,
            "from_environment",
        ) as factory_mock:
            service_mock = factory_mock.return_value
            service_mock.run.side_effect = ModelInferenceError("model is not ready")
            with TestClient(app) as client:
                response = client.post(
                    "/api/satnav/model/inference",
                    json={"instruction": "fly forward"},
                )
        self.assertEqual(response.status_code, 400)
        self.assertIn("model is not ready", response.json()["detail"])

    def test_media_raw_img_returns_jpeg_when_frame_available(self) -> None:
        import numpy as np

        with TestClient(app) as client:
            checker = main_module.rtmp_checker
            self.assertIsNotNone(checker)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            with checker._lock:
                checker._frame = frame
                checker._frame_shape = (480, 640)
                checker._frame_sequence = 5
            response = client.get("/api/satnav/media/raw_img")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["x-frame-sequence"], "5")
        self.assertTrue(response.content.startswith(b"\xff\xd8"))

    def test_media_raw_img_returns_404_without_frame(self) -> None:
        with TestClient(app) as client:
            checker = main_module.rtmp_checker
            self.assertIsNotNone(checker)
            with checker._lock:
                checker._frame = None
                checker._frame_sequence = 0
            response = client.get("/api/satnav/media/raw_img")

        self.assertEqual(response.status_code, 404)

    def test_media_model_input_img_returns_cached_jpeg(self) -> None:
        from app.media_cache import model_input_cache

        model_input_cache.update(
            jpeg_bytes=b"\xff\xd8\xff\xd9",
            frame_sequence=9,
            image_path="/tmp/infer_test.jpg",
        )
        try:
            with TestClient(app) as client:
                response = client.get("/api/satnav/media/model_input_img")
        finally:
            model_input_cache.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["x-frame-sequence"], "9")
        self.assertEqual(response.content, b"\xff\xd8\xff\xd9")

    def test_media_model_input_img_returns_404_before_inference(self) -> None:
        from app.media_cache import model_input_cache

        model_input_cache.clear()
        with TestClient(app) as client:
            response = client.get("/api/satnav/media/model_input_img")

        self.assertEqual(response.status_code, 404)


class FlightControlAuthClientTest(unittest.TestCase):
    def _backend_payload(self, token: str = "token-abc") -> bytes:
        return json.dumps(
            {
                "code": 0,
                "message": "success",
                "data": {
                    "user_id": "user-1",
                    "username": "adminPC",
                    "workspace_id": "ws-1",
                    "user_type": 1,
                    "mqtt_username": "admin",
                    "mqtt_addr": "mqtt://host:1883",
                    "access_token": token,
                },
            }
        ).encode("utf-8")

    @patch("app.adapters.flight_control_auth_client.urlopen")
    def test_login_caches_access_token(self, urlopen_mock) -> None:
        response = urlopen_mock.return_value.__enter__.return_value
        response.getcode.return_value = 200
        response.read.return_value = self._backend_payload("token-login")
        client = FlightControlAuthClient(host="127.0.0.1", port=6789)

        payload = client.login_or_refresh(
            username="adminPC",
            password="adminPC",
            flag=1,
        )

        self.assertEqual(payload["auth_method"], "login")
        self.assertEqual(client.get_auth_headers()["x-auth-token"], "token-login")
        self.assertTrue(client.status()["logged_in"])
        request = urlopen_mock.call_args[0][0]
        self.assertIn("/manage/api/v1/login", request.full_url)

    @patch("app.adapters.flight_control_auth_client.urlopen")
    def test_refresh_uses_cached_token(self, urlopen_mock) -> None:
        login_response = urlopen_mock.return_value.__enter__.return_value
        login_response.getcode.return_value = 200
        login_response.read.return_value = self._backend_payload("token-old")
        client = FlightControlAuthClient(host="127.0.0.1", port=6789)
        client.login_or_refresh(username="adminPC", password="adminPC", flag=1)

        refresh_response = urlopen_mock.return_value.__enter__.return_value
        refresh_response.getcode.return_value = 200
        refresh_response.read.return_value = self._backend_payload("token-new")
        payload = client.login_or_refresh()

        self.assertEqual(payload["auth_method"], "refresh")
        self.assertEqual(client.get_auth_headers()["x-auth-token"], "token-new")
        refresh_request = urlopen_mock.call_args[0][0]
        self.assertIn("/manage/api/v1/token/refresh", refresh_request.full_url)
        self.assertEqual(refresh_request.headers["X-auth-token"], "token-old")

    @patch("app.adapters.flight_control_auth_client.urlopen")
    def test_refresh_401_clears_cached_token(self, urlopen_mock) -> None:
        login_response = urlopen_mock.return_value.__enter__.return_value
        login_response.getcode.return_value = 200
        login_response.read.return_value = self._backend_payload("token-old")
        client = FlightControlAuthClient(host="127.0.0.1", port=6789)
        client.login_or_refresh(username="adminPC", password="adminPC", flag=1)

        error = HTTPError(
            url="http://127.0.0.1:6789/manage/api/v1/token/refresh",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )
        error.read = lambda: b'{"code":401,"message":"unauthorized"}'
        urlopen_mock.side_effect = error

        with self.assertRaises(FlightControlAuthError):
            client.login_or_refresh()

        self.assertFalse(client.status()["logged_in"])

    def test_login_requires_credentials_when_cache_empty(self) -> None:
        client = FlightControlAuthClient(host="127.0.0.1", port=6789)
        with self.assertRaises(FlightControlAuthError):
            client.login_or_refresh()

    def test_clear_removes_cached_token(self) -> None:
        client = FlightControlAuthClient(host="127.0.0.1", port=6789)
        client._store_session(
            access_token="token-abc",
            data={"username": "adminPC", "workspace_id": "ws-1"},
            auth_method="login",
        )
        client.clear()
        self.assertFalse(client.status()["logged_in"])


class BackendHealthCheckerTest(unittest.TestCase):
    @patch("app.adapters.backend_health_checker.urlopen")
    def test_api_docs_probe_accepts_valid_openapi(self, urlopen_mock) -> None:
        body = json.dumps(
            {
                "openapi": "3.0.1",
                "info": {"title": "CloudSDK API", "version": "1.0.0"},
            }
        ).encode("utf-8")
        response = urlopen_mock.return_value.__enter__.return_value
        response.getcode.return_value = 200
        response.read.return_value = body
        checker = BackendHealthChecker(host="127.0.0.1", port=6789)
        payload = checker.check()
        self.assertTrue(payload["healthy"])
        self.assertEqual(payload["service_title"], "CloudSDK API")

    @patch("app.adapters.backend_health_checker.urlopen")
    def test_api_docs_probe_rejects_unexpected_title(self, urlopen_mock) -> None:
        body = json.dumps(
            {
                "openapi": "3.0.1",
                "info": {"title": "Other API", "version": "1.0.0"},
            }
        ).encode("utf-8")
        response = urlopen_mock.return_value.__enter__.return_value
        response.getcode.return_value = 200
        response.read.return_value = body
        checker = BackendHealthChecker(host="127.0.0.1", port=6789)
        payload = checker.check()
        self.assertFalse(payload["healthy"])
        self.assertIn("unexpected service title", payload["error"])

    def test_missing_host_returns_none_from_environment(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if key != "SATNAV_BACKEND_HOST"
        }
        with patch.dict(os.environ, env, clear=True):
            checker = BackendHealthChecker.from_environment()
        self.assertIsNone(checker)


class RtmpStreamCheckerTest(unittest.TestCase):
    def test_missing_env_raises_with_guidance(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if key != "SATNAV_RTMP_URL"
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                RtmpStreamChecker.from_environment()

        message = str(ctx.exception)
        self.assertIn("SATNAV_RTMP_URL", message)
        self.assertIn("start_api.sh", message)

    def test_invalid_url_scheme_is_rejected(self) -> None:
        checker = RtmpStreamChecker(rtmp_url="http://127.0.0.1/live/satnav")
        payload = checker.check()
        self.assertFalse(payload["connected"])
        self.assertIn("rtmp://", payload["error"])

    def test_check_reports_connected_reader(self) -> None:
        checker = RtmpStreamChecker(rtmp_url="rtmp://127.0.0.1/live/satnav")
        with checker._lock:
            checker._state = "connected"
            checker._frame_shape = (1080, 1920)
            checker._frame_sequence = 3
        payload = checker.check()
        self.assertTrue(payload["stream_active"])
        self.assertEqual(payload["video"]["width"], 1920)

    def test_check_reports_connecting_state(self) -> None:
        checker = RtmpStreamChecker(rtmp_url="rtmp://127.0.0.1/live/satnav")
        with checker._lock:
            checker._state = "connecting"
        payload = checker.check()
        self.assertFalse(payload["stream_active"])
        self.assertIn("connecting", payload["error"])

    def test_start_launches_background_reader(self) -> None:
        checker = RtmpStreamChecker(rtmp_url="rtmp://127.0.0.1/live/satnav")
        with patch.object(checker, "_reader_loop") as reader_mock:
            checker.start()
            checker._thread.join(timeout=1.0)
        reader_mock.assert_called_once()
        checker.close()

    def test_request_fresh_frame_waits_for_new_sequence(self) -> None:
        import numpy as np

        checker = RtmpStreamChecker(rtmp_url="rtmp://127.0.0.1/live/satnav")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def publish_frame() -> None:
            with checker._frame_condition:
                checker._frame = frame
                checker._frame_shape = (480, 640)
                checker._frame_sequence += 1
                checker._frame_condition.notify_all()

        publish_frame()
        with checker._frame_condition:
            checker._frame_sequence = 0

        with patch.object(checker, "_stop_event") as stop_mock:
            stop_mock.is_set.return_value = False
            result_holder: dict[str, object] = {}

            def waiter() -> None:
                sequence, captured = checker.request_fresh_frame(timeout_s=2.0)
                result_holder["sequence"] = sequence
                result_holder["shape"] = captured.shape

            thread = threading.Thread(target=waiter)
            thread.start()
            time.sleep(0.05)
            publish_frame()
            thread.join(timeout=2.0)

        self.assertEqual(result_holder["sequence"], 1)
        self.assertEqual(result_holder["shape"], (480, 640, 3))


class ModelInferenceServiceTest(unittest.TestCase):
    def test_run_starts_session_and_sends_image(self) -> None:
        import numpy as np
        from PIL import Image

        from app.services.model_inference_service import ModelInferenceService

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        model_client = SwiftVLNDeployClient(repo_root=Path("/tmp"))
        rtmp_checker = RtmpStreamChecker(rtmp_url="rtmp://127.0.0.1/live/satnav")

        with tempfile.TemporaryDirectory() as temp_dir:
            service = ModelInferenceService(
                model_client=model_client,
                rtmp_checker=rtmp_checker,
                frame_timeout_s=1.0,
                resize_mode="center-crop",
                input_root=Path(temp_dir),
            )
            with patch.object(
                model_client,
                "status",
                return_value={"state": "ready"},
            ), patch.object(
                rtmp_checker,
                "check",
                return_value={"stream_active": True},
            ), patch.object(
                model_client,
                "session_info",
                return_value={
                    "session_started": False,
                    "session_id": None,
                    "instruction": None,
                    "state": None,
                },
            ), patch.object(
                model_client,
                "start_session",
                return_value={
                    "type": "start",
                    "session_id": "satnav-infer-test",
                    "state": "waiting_image",
                },
            ) as start_mock, patch.object(
                rtmp_checker,
                "request_fresh_frame",
                return_value=(3, frame),
            ), patch(
                "app.services.model_inference_service.prepare_model_image",
                return_value=Image.new("RGB", (448, 448)),
            ), patch.object(
                model_client,
                "send_image",
                return_value={
                    "type": "image",
                    "session_id": "satnav-infer-test",
                    "state": "waiting_feedback",
                    "performed_inference": True,
                    "raw_action_text": "1 2 3 4",
                    "actions": [1, 2, 3, 4],
                    "next_action": 1,
                    "remaining_actions": [1, 2, 3, 4],
                    "completed_action": None,
                },
            ) as image_mock:
                payload = service.run("fly to the lake")

        self.assertTrue(payload["performed_inference"])
        self.assertTrue(payload["started_new_session"])
        start_mock.assert_called_once()
        image_mock.assert_called_once()
        image_path = Path(image_mock.call_args[0][0])
        self.assertTrue(image_path.is_file())


class ModelPathMappingTest(unittest.TestCase):
    def test_explicit_model_path_is_mapped_to_deploy_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "hf-model"
            model_path.mkdir()
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            (model_path / "model-00001.safetensors").touch()
            client = SwiftVLNDeployClient(
                repo_root=root,
                model_name=(
                    "swiftvln-satnav-3b-1ep-f32s4-overlap0-"
                    "pf-h8-b1.0-pool-s2-noembed"
                ),
                model_path=model_path,
            )

            output_root = client._prepare_model_registry()

            checkpoint = (
                output_root
                / "swiftvln"
                / client.model_name
                / "checkpoint-1"
            )
            self.assertTrue(checkpoint.is_symlink())
            self.assertEqual(checkpoint.resolve(), model_path.resolve())


class OperatorSessionLogWriterTest(unittest.TestCase):
    def test_start_append_close_writes_log_file(self) -> None:
        from app.services.operator_session_log import OperatorSessionLogWriter

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            writer = OperatorSessionLogWriter(
                log_root=root / "satnav" / "runtime" / "logs",
                repo_root=root,
            )
            started = writer.start()
            self.assertTrue(started["active"])
            self.assertTrue(started["log_path_relative"].startswith("satnav/runtime/logs/operator-"))

            writer.append(["[12:00:00.000] [EVT] {\"event\":\"inference_clicked\"}"])
            again = writer.start()
            self.assertEqual(again["log_path"], started["log_path"])
            writer.close()
            self.assertFalse(writer.status()["active"])

            log_path = Path(started["log_path"])
            self.assertTrue(log_path.is_file())
            self.assertIn("inference_clicked", log_path.read_text(encoding="utf-8"))


class CorsConfigTest(unittest.TestCase):
    def test_default_origin_when_env_missing(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if key != "SATNAV_CORS_ORIGINS"
        }
        with patch.dict(os.environ, env, clear=True):
            from app.cors import cors_origins_from_environment

            self.assertEqual(
                cors_origins_from_environment(),
                ["http://127.0.0.1:5173"],
            )

    def test_parse_comma_separated_origins(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SATNAV_CORS_ORIGINS": (
                    "http://127.0.0.1:5173, https://satnav.example.com"
                )
            },
            clear=False,
        ):
            from app.cors import cors_origins_from_environment

            self.assertEqual(
                cors_origins_from_environment(),
                [
                    "http://127.0.0.1:5173",
                    "https://satnav.example.com",
                ],
            )


class FlightControlFlightClientOsdTest(unittest.TestCase):
    def setUp(self) -> None:
        self.auth_client = FlightControlAuthClient(host="127.0.0.1", port=6789)
        self.auth_client._store_session(
            access_token="token-abc",
            data={"username": "adminPC", "workspace_id": "ws-1"},
            auth_method="login",
        )
        self.registry = FlightControlDeviceRegistry()
        self.registry.set_workspace_id("ws-1")
        self.registry.register(
            rc_sn="RCPLUS2_SN_EXAMPLE",
            device_sn="AIRCRAFT_SN_EXAMPLE",
        )
        self.registry.set_client_id("postman-rc2-001")
        self.client = FlightControlFlightClient(self.auth_client, self.registry)

    @patch.object(FlightControlFlightClient, "_post_json")
    def test_get_osd_latest_returns_snapshot_fields(self, post_json_mock) -> None:
        post_json_mock.return_value = {
            "code": 0,
            "message": "success",
            "data": {
                "device_sn": "AIRCRAFT_SN_EXAMPLE",
                "attitude_head": 42.5,
                "latitude": 22.6070293,
                "longitude": 114.0561159,
                "height": 25.3,
                "received_at_ms": 1743518000456,
                "age_ms": 120,
            },
        }

        payload = self.client.get_osd_latest()

        post_json_mock.assert_called_once_with(
            path_suffix="osd/latest",
            payload={
                "client_id": "postman-rc2-001",
                "rc_sn": "RCPLUS2_SN_EXAMPLE",
                "device_sn": "AIRCRAFT_SN_EXAMPLE",
            },
        )
        self.assertEqual(payload["latitude"], 22.6070293)
        self.assertEqual(payload["height"], 25.3)
        self.assertEqual(payload["received_at_ms"], 1743518000456)


if __name__ == "__main__":
    unittest.main()
