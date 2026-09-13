# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for the settings panel's Custom Models feature.

Covers the ``~/.kiss/MY_MODELS.json`` CRUD helpers in
``kiss.core.models.model_info`` and the daemon command handlers
(``getMyModels`` / ``saveMyModel`` / ``deleteMyModel``) in
``kiss.server.commands``, exercising the real file on disk.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import kiss.core.models.model_info as mi
from kiss.server.commands import _CommandsMixin
from kiss.server.sorcar import API


class _MyModelsFileCase(unittest.TestCase):
    """Base: redirect ``USER_MY_MODELS_PATH`` into a temp dir."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._orig_path = mi.USER_MY_MODELS_PATH
        mi.USER_MY_MODELS_PATH = Path(self._tmp.name) / "MY_MODELS.json"

    def tearDown(self) -> None:
        mi.USER_MY_MODELS_PATH = self._orig_path
        self._tmp.cleanup()

    def read_raw(self) -> dict[str, Any]:
        raw = json.loads(mi.USER_MY_MODELS_PATH.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)
        return raw


class TestCustomModelCrud(_MyModelsFileCase):
    """save/list/delete_custom_model against the real file."""

    def test_add_lists_and_persists_all_fields(self) -> None:
        self.assertIsNone(
            mi.save_custom_model(
                " my-model ",
                endpoint="http://localhost:8080/v1",
                api_key="sk-1",
                headers="X-A: 1\nX-B: 2",
            )
        )
        rows = mi.list_custom_models()
        self.assertEqual(
            rows,
            [{
                "name": "my-model",
                "endpoint": "http://localhost:8080/v1",
                "api_key": "sk-1",
                "headers": "X-A: 1\nX-B: 2",
            }],
        )
        entry = self.read_raw()["my-model"]
        # The catalog loader requires these keys on every entry.
        self.assertEqual(entry["context_length"], 128000)
        self.assertEqual(entry["input_price_per_1M"], 0.0)
        self.assertEqual(entry["output_price_per_1M"], 0.0)

    def test_add_preserves_documentation_keys(self) -> None:
        mi.save_custom_model("m1", endpoint="http://x/v1")
        raw = self.read_raw()
        self.assertIn("_documentation", raw)
        self.assertIn("_example/my-org/my-custom-model", raw)
        self.assertEqual(mi.list_custom_models()[0]["name"], "m1")

    def test_rejects_empty_and_underscore_names(self) -> None:
        self.assertEqual(
            mi.save_custom_model("   "),
            "Custom model name must not be empty",
        )
        self.assertEqual(
            mi.save_custom_model("_hidden"),
            "Custom model name must not start with '_'",
        )
        self.assertEqual(mi.list_custom_models(), [])

    def test_update_merges_and_clears_empty_fields(self) -> None:
        mi.save_custom_model("m", endpoint="http://a/v1", api_key="k1")
        # A hand-written extra key must survive an endpoint edit.
        raw = self.read_raw()
        raw["m"]["thinking"] = "xhigh"
        raw["m"]["context_length"] = 42
        mi.USER_MY_MODELS_PATH.write_text(json.dumps(raw), encoding="utf-8")

        self.assertIsNone(
            mi.save_custom_model(
                "m", endpoint="http://b/v1", api_key="", original_name="m"
            )
        )
        entry = self.read_raw()["m"]
        self.assertEqual(entry["endpoint"], "http://b/v1")
        self.assertNotIn("api_key", entry)
        self.assertNotIn("headers", entry)
        self.assertEqual(entry["thinking"], "xhigh")
        self.assertEqual(entry["context_length"], 42)

    def test_rename_carries_entry_over_and_drops_old_key(self) -> None:
        mi.save_custom_model("old", endpoint="http://a/v1", api_key="k")
        mi.save_custom_model(
            "new", endpoint="http://a/v1", api_key="k", original_name="old"
        )
        raw = self.read_raw()
        self.assertNotIn("old", raw)
        self.assertEqual(raw["new"]["endpoint"], "http://a/v1")
        self.assertEqual(raw["new"]["api_key"], "k")

    def test_rename_from_missing_original_still_adds(self) -> None:
        self.assertIsNone(
            mi.save_custom_model(
                "n", endpoint="http://a/v1", original_name="ghost"
            )
        )
        self.assertEqual(mi.list_custom_models()[0]["name"], "n")

    def test_rename_onto_existing_name_is_refused(self) -> None:
        mi.save_custom_model("a", endpoint="http://a/v1")
        mi.save_custom_model("b", endpoint="http://b/v1", api_key="kb")
        # Renaming "a" onto the existing "b" would destroy two records
        # in one Save; it must be refused and the file left untouched.
        before = mi.USER_MY_MODELS_PATH.read_text(encoding="utf-8")
        error = mi.save_custom_model(
            "b", endpoint="http://c/v1", original_name="a"
        )
        self.assertEqual(error, "A custom model named 'b' already exists")
        self.assertEqual(
            mi.USER_MY_MODELS_PATH.read_text(encoding="utf-8"), before
        )

    def test_add_with_existing_name_is_refused(self) -> None:
        mi.save_custom_model("a", endpoint="http://a/v1")
        error = mi.save_custom_model("a", endpoint="http://b/v1")
        assert error is not None
        self.assertIn("already exists; use its Edit button", error)
        self.assertEqual(self.read_raw()["a"]["endpoint"], "http://a/v1")

    def test_underscore_original_name_is_refused(self) -> None:
        error = mi.save_custom_model(
            "x", endpoint="http://a/v1", original_name="_documentation"
        )
        self.assertEqual(
            error, "Original model name must not start with '_'"
        )

    def test_new_entry_shadowing_bundled_model_keeps_its_metadata(
        self,
    ) -> None:
        # Overriding a bundled catalog model must seed THAT model's
        # context/prices, not the generic 128K/$0 defaults, or the
        # override would silently break budget accounting.
        bundled_name = sorted(mi.MODEL_INFO)[0]
        info = mi.MODEL_INFO[bundled_name]
        self.assertIsNone(
            mi.save_custom_model(bundled_name, endpoint="http://a/v1")
        )
        entry = self.read_raw()[bundled_name]
        self.assertEqual(entry["context_length"], info.context_length)
        self.assertEqual(entry["input_price_per_1M"], info.input_price_per_1M)
        self.assertEqual(
            entry["output_price_per_1M"], info.output_price_per_1M
        )

    def test_delete_refuses_reserved_keys(self) -> None:
        mi.list_custom_models()  # seed the file
        error = mi.delete_custom_model("_documentation")
        self.assertEqual(
            error, "Custom model name must not start with '_'"
        )
        self.assertIn("_documentation", self.read_raw())

    def test_custom_model_config_builds_from_entry(self) -> None:
        mi.save_custom_model(
            "m",
            endpoint="http://localhost:1/v1",
            api_key="sk",
            headers="X-A: 1\njunk-line\nX-B: 2",
        )
        self.assertEqual(
            mi.custom_model_config("m"),
            {
                "base_url": "http://localhost:1/v1",
                "api_key": "sk",
                "extra_headers": {"X-A": "1", "X-B": "2"},
            },
        )
        mi.save_custom_model("bare", endpoint="http://x/v1", original_name="")
        self.assertEqual(
            mi.custom_model_config("bare"), {"base_url": "http://x/v1"}
        )
        self.assertIsNone(mi.custom_model_config("ghost"))
        # An entry without an endpoint (a plain catalog override) has
        # no per-model transport config.
        mi.save_custom_model("no-endpoint")
        self.assertIsNone(mi.custom_model_config("no-endpoint"))

    def test_delete_removes_only_named_entry(self) -> None:
        mi.save_custom_model("m1", endpoint="http://a/v1")
        mi.save_custom_model("m2", endpoint="http://b/v1")
        mi.delete_custom_model("m1")
        self.assertEqual(
            [m["name"] for m in mi.list_custom_models()], ["m2"]
        )
        self.assertIn("_documentation", self.read_raw())

    def test_delete_missing_name_is_a_noop(self) -> None:
        mi.save_custom_model("m1", endpoint="http://a/v1")
        before = mi.USER_MY_MODELS_PATH.read_text(encoding="utf-8")
        mi.delete_custom_model("ghost")
        self.assertEqual(
            mi.USER_MY_MODELS_PATH.read_text(encoding="utf-8"), before
        )

    def test_list_tolerates_non_string_and_missing_fields(self) -> None:
        mi.USER_MY_MODELS_PATH.write_text(
            json.dumps({
                "weird": {
                    "context_length": 1,
                    "input_price_per_1M": 0,
                    "output_price_per_1M": 0,
                    "endpoint": 123,
                },
            }),
            encoding="utf-8",
        )
        self.assertEqual(
            mi.list_custom_models(),
            [{"name": "weird", "endpoint": "", "api_key": "", "headers": ""}],
        )

    def test_corrupt_file_refuses_writes_and_is_preserved(self) -> None:
        # A corrupt registry must never be "recovered" by clobbering:
        # whatever hand-edited models it still holds would be lost.
        mi.USER_MY_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        mi.USER_MY_MODELS_PATH.write_text("{not json", encoding="utf-8")
        self.assertEqual(mi.list_custom_models(), [])
        error = mi.save_custom_model("m", endpoint="http://a/v1")
        assert error is not None
        self.assertIn("unreadable", error)
        self.assertEqual(
            mi.USER_MY_MODELS_PATH.read_text(encoding="utf-8"), "{not json"
        )
        error = mi.delete_custom_model("m")
        assert error is not None
        self.assertIn("unreadable", error)

    def test_non_object_json_reads_as_none(self) -> None:
        mi.USER_MY_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        mi.USER_MY_MODELS_PATH.write_text("[1, 2]", encoding="utf-8")
        self.assertIsNone(mi._read_my_models_file())


class _FakePrinter:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def broadcast(self, msg: dict[str, Any]) -> None:
        self.messages.append(msg)


class _FakeServer(_CommandsMixin):
    def __init__(self) -> None:
        self.printer: Any = _FakePrinter()
        self.work_dir = "/tmp"
        self._state_lock = threading.RLock()

    def last(self, event_type: str) -> dict[str, Any] | None:
        for msg in reversed(self.printer.messages):
            if msg.get("type") == event_type:
                return dict(msg)
        return None


class TestMyModelsCommandHandlers(_MyModelsFileCase):
    """The daemon commands drive the file and broadcast myModelsData."""

    def setUp(self) -> None:
        super().setUp()
        self.server = _FakeServer()

    def test_commands_are_in_the_server_api_catalog(self) -> None:
        self.assertIn("getMyModels", API)
        self.assertEqual(API["saveMyModel"].required, ("name",))
        self.assertEqual(API["deleteMyModel"].required, ("name",))
        for name in ("getMyModels", "saveMyModel", "deleteMyModel"):
            self.assertEqual(API[name].handler, "forward")
            self.assertIn(name, _CommandsMixin._HANDLERS)

    def test_get_my_models_is_conn_scoped(self) -> None:
        self.server._cmd_get_my_models({"connId": "c1"})
        msg = self.server.last("myModelsData")
        assert msg is not None
        self.assertEqual(msg["connId"], "c1")
        self.assertEqual(msg["models"], [])

        self.server._cmd_get_my_models({})
        msg = self.server.last("myModelsData")
        assert msg is not None
        self.assertNotIn("connId", msg)

    def test_save_my_model_writes_file_and_broadcasts_to_all(self) -> None:
        self.server._cmd_save_my_model({
            "name": "m1",
            "endpoint": "http://localhost:1/v1",
            "apiKey": "sk",
            "headers": "X: 1",
            "connId": "c1",
        })
        msg = self.server.last("myModelsData")
        assert msg is not None
        self.assertNotIn(
            "connId", msg,
            "mutations repaint every window, not only the sender",
        )
        self.assertEqual(
            msg["models"],
            [{
                "name": "m1",
                "endpoint": "http://localhost:1/v1",
                "api_key": "sk",
                "headers": "X: 1",
            }],
        )
        self.assertIn("m1", self.read_raw())

    def test_save_my_model_rename_via_original_name(self) -> None:
        self.server._cmd_save_my_model({"name": "m1", "endpoint": "http://a"})
        self.server._cmd_save_my_model({
            "name": "m2", "endpoint": "http://a", "originalName": "m1",
        })
        msg = self.server.last("myModelsData")
        assert msg is not None
        self.assertEqual([m["name"] for m in msg["models"]], ["m2"])

    def test_save_my_model_bad_name_answers_error_to_sender(self) -> None:
        self.server._cmd_save_my_model({"name": "  ", "connId": "c9"})
        err = self.server.last("error")
        assert err is not None
        self.assertEqual(err["connId"], "c9")
        self.assertIn("must not be empty", err["text"])
        self.assertIsNone(self.server.last("myModelsData"))

        # Without a connId the error still goes out (to all).
        self.server._cmd_save_my_model({"name": "_x"})
        err = self.server.last("error")
        assert err is not None
        self.assertNotIn("connId", err)

    def test_save_my_model_tolerates_non_string_fields(self) -> None:
        self.server._cmd_save_my_model({
            "name": "m1", "endpoint": 42, "apiKey": None, "headers": ["x"],
        })
        msg = self.server.last("myModelsData")
        assert msg is not None
        self.assertEqual(
            msg["models"],
            [{"name": "m1", "endpoint": "", "api_key": "", "headers": ""}],
        )

    def test_delete_my_model_removes_and_broadcasts(self) -> None:
        self.server._cmd_save_my_model({"name": "m1", "endpoint": "http://a"})
        self.server._cmd_delete_my_model({"name": "m1"})
        msg = self.server.last("myModelsData")
        assert msg is not None
        self.assertEqual(msg["models"], [])
        self.assertNotIn("m1", self.read_raw())

    def test_delete_my_model_non_string_name_still_broadcasts(self) -> None:
        self.server._cmd_save_my_model({"name": "m1", "endpoint": "http://a"})
        self.server._cmd_delete_my_model({"name": 42})
        msg = self.server.last("myModelsData")
        assert msg is not None
        self.assertEqual([m["name"] for m in msg["models"]], ["m1"])

    def test_delete_my_model_reserved_key_answers_error(self) -> None:
        mi.list_custom_models()  # seed the file
        self.server._cmd_delete_my_model({
            "name": "_documentation", "connId": "c2",
        })
        err = self.server.last("error")
        assert err is not None
        self.assertEqual(err["connId"], "c2")
        self.assertIn("_documentation", self.read_raw())

    def test_save_my_model_write_failure_answers_error(self) -> None:
        # A read-only registry directory makes the atomic write's temp
        # staging raise OSError; the handler must answer with an error
        # event instead of killing the connection's dispatch loop.
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        mi.save_custom_model("m1", endpoint="http://a/v1")
        mi.USER_MY_MODELS_PATH.parent.chmod(0o555)
        try:
            self.server._cmd_save_my_model({
                "name": "m2", "endpoint": "http://b/v1", "connId": "c3",
            })
            err = self.server.last("error")
            assert err is not None
            self.assertIn("Could not write", err["text"])
            self.assertEqual(err["connId"], "c3")
            self.assertIsNone(self.server.last("myModelsData"))

            self.server._cmd_delete_my_model({"name": "m1"})
            err = self.server.last("error")
            assert err is not None
            self.assertIn("Could not write", err["text"])
        finally:
            mi.USER_MY_MODELS_PATH.parent.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
