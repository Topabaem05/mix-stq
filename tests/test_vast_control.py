import io
import json
import os
import ssl
import stat
import subprocess
import sys
import urllib.error

import pytest
import vast_control

SENTINEL_TOKEN = "SENTINEL_MIXSTQ_TASK2_TOKEN_7f53c91a"


class _Response:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def _offer(offer_id=1, **overrides):
    offer = {
        "id": offer_id,
        "rentable": True,
        "num_gpus": 1,
        "gpu_name": "RTX 4090",
        "gpu_ram": 24_000,
        "dph_total": 0.5,
        "disk_space": 100,
        "cpu_ram": 64_000,
        "cpu_cores": 16,
        "inet_down": 500.0,
        "reliability2": 0.98,
        "geolocation": "KR",
    }
    offer.update(overrides)
    return offer


def test_request_uses_verified_in_process_https_without_observable_token_leakage(
    monkeypatch, capsys
):
    captured = {}
    expected_context = object()

    def forbidden_subprocess(*_args, **_kwargs):
        pytest.fail("API requests must not start a child process")

    def fake_urlopen(request, **kwargs):
        captured["request"] = request
        captured["metadata_for_logging"] = {
            "method": request.get_method(),
            "url": request.full_url,
            "timeout": kwargs["timeout"],
        }
        captured["context"] = kwargs["context"]
        return _Response({"success": True, "echo": SENTINEL_TOKEN})

    monkeypatch.setenv("MIXSTQ_VAST_KEY", SENTINEL_TOKEN)
    monkeypatch.setattr(subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(vast_control, "_https_context", lambda: expected_context)
    monkeypatch.setattr(vast_control.urllib.request, "urlopen", fake_urlopen)

    result = vast_control._request("POST", "/bundles", {"limit": 1})

    request = captured["request"]
    assert request.get_header("Authorization") == "Bearer " + SENTINEL_TOKEN
    assert captured["context"] is expected_context
    assert result == {"success": True, "echo": "[REDACTED]"}
    assert SENTINEL_TOKEN not in repr(request)
    assert SENTINEL_TOKEN not in json.dumps(captured["metadata_for_logging"])
    output = capsys.readouterr()
    assert SENTINEL_TOKEN not in output.out
    assert SENTINEL_TOKEN not in output.err


def test_https_context_requires_certificate_verification():
    context = vast_control._https_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


@pytest.mark.parametrize("failure", ["http", "invalid-json"])
def test_request_failures_hide_token_headers_and_response_body(
    monkeypatch, capsys, failure
):
    monkeypatch.setenv("MIXSTQ_VAST_KEY", SENTINEL_TOKEN)
    monkeypatch.setattr(vast_control, "_https_context", ssl.create_default_context)

    if failure == "http":
        error = urllib.error.HTTPError(
            vast_control.API + "/instances/",
            401,
            "body=" + SENTINEL_TOKEN,
            {"X-Debug-Token": SENTINEL_TOKEN},
            io.BytesIO(("server body " + SENTINEL_TOKEN).encode()),
        )

        def fake_urlopen(*_args, **_kwargs):
            raise error

    else:

        def fake_urlopen(*_args, **_kwargs):
            response = _Response({"unused": True})
            response.body = ("not json " + SENTINEL_TOKEN).encode()
            return response

    monkeypatch.setattr(vast_control.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(vast_control.VastError) as caught:
        vast_control._request("GET", "/instances/")

    output = capsys.readouterr()
    observable = str(caught.value) + output.out + output.err
    assert SENTINEL_TOKEN not in observable
    assert "server body" not in observable
    assert "not json" not in observable


def test_empty_candidate_key_files_never_build_authorization_or_open_url(
    monkeypatch, tmp_path, capsys
):
    primary = tmp_path / ".vast_api_key"
    secondary = tmp_path / ".config/vastai/vast_api_key"
    secondary.parent.mkdir(parents=True)
    primary.write_text("  \n", encoding="utf-8")
    secondary.write_text("\t\n", encoding="utf-8")
    calls = {"request": 0, "urlopen": 0}

    def forbidden_request(*_args, **_kwargs):
        calls["request"] += 1
        pytest.fail("an empty key must not build an Authorization header")

    def forbidden_urlopen(*_args, **_kwargs):
        calls["urlopen"] += 1
        pytest.fail("an empty key must not reach urlopen")

    monkeypatch.delenv("MIXSTQ_VAST_KEY", raising=False)
    monkeypatch.setattr(vast_control.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(vast_control.urllib.request, "Request", forbidden_request)
    monkeypatch.setattr(vast_control.urllib.request, "urlopen", forbidden_urlopen)

    with pytest.raises(vast_control.VastError, match=r"no vast\.ai key found") as caught:
        vast_control._request("GET", "/instances/")

    assert calls == {"request": 0, "urlopen": 0}
    assert "Authorization" not in str(caught.value)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_unexpected_instances_payload_never_exposes_ephemeral_secret(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: {
            "ephemeral_key": SENTINEL_TOKEN,
            "detail": "server payload must stay private",
        },
    )

    with pytest.raises(vast_control.VastError) as caught:
        vast_control.instances()

    output = capsys.readouterr()
    observable = str(caught.value) + repr(caught.value) + output.out + output.err
    assert str(caught.value) == "unexpected /instances/ payload"
    assert SENTINEL_TOKEN not in observable
    assert "ephemeral_key" not in observable
    assert "server payload" not in observable


def test_xdg_default_and_explicit_state_override(monkeypatch, tmp_path):
    xdg_home = tmp_path / "xdg-state"
    default_path = xdg_home / "mixstq/vast_state.json"
    override_path = tmp_path / "override" / "events.json"
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg_home))
    monkeypatch.setattr(vast_control, "STATE", default_path)

    assert vast_control._default_state_path() == default_path

    vast_control._save(
        {"event": "destroy", "at": 100.0, "id": 9},
        state_path=override_path,
    )

    assert override_path.is_file()
    assert not default_path.exists()


def test_state_write_is_atomic_private_and_sanitized(monkeypatch, tmp_path, capsys):
    state_path = tmp_path / "xdg" / "mixstq" / "vast_state.json"
    replacements = []
    original_replace = os.replace
    monkeypatch.setattr(vast_control, "STATE", state_path)

    def recording_replace(source, destination):
        replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(vast_control.os, "replace", recording_replace)

    vast_control._save(
        {
            "event": "create",
            "at": 123.0,
            "offer": 7,
            "response": {
                "success": True,
                "new_contract": 42,
                "ephemeral_private_key": SENTINEL_TOKEN,
            },
            "debug_token": SENTINEL_TOKEN,
        }
    )

    serialized = state_path.read_text(encoding="utf-8")
    assert json.loads(serialized) == [
        {"event": "create", "at": 123.0, "offer": 7, "instance_id": 42}
    ]
    assert SENTINEL_TOKEN not in serialized
    assert "response" not in serialized
    assert "ephemeral_private_key" not in serialized
    assert stat.S_IMODE(state_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
    assert len(replacements) == 1
    temporary, destination = map(os.fspath, replacements[0])
    assert os.path.dirname(temporary) == os.fspath(state_path.parent)
    assert destination == os.fspath(state_path)
    assert not list(state_path.parent.glob(".vast_state.json.*"))
    output = capsys.readouterr()
    assert SENTINEL_TOKEN not in output.out + output.err


def test_confirmed_create_never_persists_or_prints_raw_response(
    monkeypatch, tmp_path, capsys
):
    state_path = tmp_path / "mixstq" / "vast_state.json"
    monkeypatch.setattr(vast_control, "STATE", state_path)
    monkeypatch.setattr(
        vast_control,
        "_search_offers",
        lambda *_args, **_kwargs: [_offer(offer_id=123, machine_id=71654)],
    )
    monkeypatch.setattr(
        vast_control,
        "create",
        lambda *_args, **_kwargs: {
            "success": True,
            "new_contract": 456,
            "ephemeral_key": SENTINEL_TOKEN,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vast-control",
            "create",
            "--offer",
            "123",
            "--max-hourly",
            "0.75",
            "--confirm",
        ],
    )

    assert vast_control.main() == 0

    output = capsys.readouterr()
    serialized = state_path.read_text(encoding="utf-8")
    assert SENTINEL_TOKEN not in output.out + output.err + serialized
    assert "ephemeral_key" not in output.out + serialized
    event = json.loads(serialized)[0]
    assert event.pop("at") > 0
    assert event == {
        "event": "create",
        "offer": 123,
        "instance_id": 456,
    }


INVALID_NUMBERS = [True, -1, float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize(
    "field",
    [
        "min_system_ram_gb",
        "min_cpu_cores",
        "min_download_mbps",
        "min_reliability",
    ],
)
@pytest.mark.parametrize("value", INVALID_NUMBERS)
def test_search_rejects_invalid_optional_minima_before_network(
    monkeypatch, field, value
):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not reach network"),
    )
    minima = {field: value}

    with pytest.raises(vast_control.VastError):
        vast_control.search("RTX_4090", 0.6, 24, 80, 10, **minima)


@pytest.mark.parametrize("field", ["max_price", "min_vram", "disk", "limit"])
@pytest.mark.parametrize("value", [*INVALID_NUMBERS, 0])
def test_search_rejects_invalid_required_numbers_before_network(
    monkeypatch, field, value
):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not reach network"),
    )
    arguments = {
        "gpu": "RTX_4090",
        "max_price": 0.6,
        "min_vram": 24,
        "disk": 80,
        "limit": 10,
    }
    arguments[field] = value

    with pytest.raises(vast_control.VastError):
        vast_control.search(**arguments)


def test_reliability_above_one_is_rejected_before_network(monkeypatch):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not reach network"),
    )

    with pytest.raises(vast_control.VastError):
        vast_control.search(
            "RTX_4090", 0.6, 24, 80, 10, min_reliability=1.0001
        )


def test_zero_optional_minima_are_allowed(monkeypatch):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: {"offers": [_offer()]},
    )

    rows = vast_control.search(
        "RTX_4090",
        0.6,
        24,
        80,
        10,
        min_system_ram_gb=0,
        min_cpu_cores=0,
        min_download_mbps=0,
        min_reliability=0,
    )

    assert [row["id"] for row in rows] == [1]


@pytest.mark.parametrize("field", ["offer_id", "disk"])
@pytest.mark.parametrize("value", [*INVALID_NUMBERS, 0])
def test_create_rejects_invalid_required_numbers_before_network(
    monkeypatch, field, value
):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not reach network"),
    )
    arguments = {"offer_id": 1, "image": "fake/image", "disk": 80, "onstart": None}
    arguments[field] = value

    with pytest.raises(vast_control.VastError):
        vast_control.create(**arguments)


@pytest.mark.parametrize("value", [*INVALID_NUMBERS, 0])
def test_destroy_rejects_invalid_instance_id_before_network(monkeypatch, value):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not reach network"),
    )

    with pytest.raises(vast_control.VastError):
        vast_control.destroy(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_ram", -1),
        ("dph_total", 0),
        ("disk_space", float("inf")),
        ("cpu_ram", True),
        ("cpu_cores", float("nan")),
        ("inet_down", -1),
        ("reliability2", 1.01),
    ],
)
def test_search_rejects_offers_with_invalid_prices_or_resources(
    monkeypatch, field, value
):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: {"offers": [_offer(**{field: value})]},
    )

    assert vast_control.search("RTX_4090", 0.6, 24, 80, 10) == []


def test_invalid_confirmed_create_has_no_network_or_state_mutation(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "mixstq" / "vast_state.json"
    monkeypatch.setattr(vast_control, "STATE", state_path)
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not reach network"),
    )
    monkeypatch.setattr(
        vast_control,
        "_save",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not mutate state"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vast-control",
            "create",
            "--offer",
            "123",
            "--max-hourly",
            "nan",
            "--confirm",
        ],
    )

    with pytest.raises(vast_control.VastError):
        vast_control.main()

    assert not state_path.exists()


def test_search_adds_exact_raw_constraints_and_returns_resource_fields(monkeypatch):
    captured = {}

    def fake_request(method, path, payload):
        captured.update(method=method, path=path, payload=payload)
        return {"offers": [_offer()]}

    monkeypatch.setattr(vast_control, "_request", fake_request)

    rows = vast_control.search(
        "RTX_4090",
        0.6,
        24,
        80,
        10,
        min_system_ram_gb=64,
        min_cpu_cores=16,
        min_download_mbps=500,
        min_reliability=0.98,
    )

    assert captured == {
        "method": "POST",
        "path": "/bundles",
        "payload": {
            "rentable": {"eq": True},
            "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": 24_000},
            "dph_total": {"lte": 0.6},
            "disk_space": {"gte": 80},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 40,
            "cpu_ram": {"gte": 64_000},
            "cpu_cores": {"gte": 16},
            "inet_down": {"gte": 500},
            "reliability": {"gte": 0.98},
        },
    }
    assert rows == [
        {
            "id": 1,
            "gpu": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram_gb": 23.4,
            "dph": 0.5,
            "disk_gb": 100,
            "system_ram_gb": 64.0,
            "cpu_cores": 16,
            "down_mbps": 500.0,
            "reliability": 0.98,
            "geo": "KR",
            "machine_id": None,
            "dph_all_in": 0.5,
            "storage_cost_per_gb_month_usd": None,
        }
    ]


@pytest.mark.parametrize(
    ("field", "minimum", "below"),
    [
        ("cpu_ram", {"min_system_ram_gb": 64}, 63_999.99),
        ("cpu_cores", {"min_cpu_cores": 16}, 15.99),
        ("inet_down", {"min_download_mbps": 500}, 499.99),
        ("reliability2", {"min_reliability": 0.98}, 0.9799),
    ],
)
def test_search_accepts_exact_boundaries_and_rejects_below_or_missing(
    monkeypatch, field, minimum, below
):
    boundary = _offer(offer_id=1)
    below_offer = _offer(offer_id=2, **{field: below})
    missing_offer = _offer(offer_id=3, **{field: None})
    malformed_offer = _offer(offer_id=4, **{field: "unknown"})
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: {
            "offers": [boundary, below_offer, missing_offer, malformed_offer]
        },
    )

    rows = vast_control.search("RTX_4090", 0.6, 24, 80, 10, **minimum)

    assert [row["id"] for row in rows] == [1]


def test_search_defaults_keep_offers_with_missing_optional_resource_fields(monkeypatch):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: {
            "offers": [
                _offer(cpu_ram=None, cpu_cores=None, inet_down=None, reliability2=None)
            ]
        },
    )

    rows = vast_control.search("RTX_4090", 0.6, 24, 80, 10)

    assert rows[0]["system_ram_gb"] == 0.0
    assert rows[0]["cpu_cores"] == 0
    assert rows[0]["down_mbps"] == 0.0
    assert rows[0]["reliability"] == 0.0


def test_search_accepts_current_offer_reliability_field_as_fallback(monkeypatch):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: {
            "offers": [_offer(reliability2=None, reliability=0.98)]
        },
    )

    rows = vast_control.search(
        "RTX_4090", 0.6, 24, 80, 10, min_reliability=0.98
    )

    assert rows[0]["reliability"] == 0.98


def test_search_cli_passes_optional_resource_constraints(monkeypatch):
    captured = {}

    def fake_search(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return []

    monkeypatch.setattr(vast_control, "search", fake_search)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vast-control",
            "search",
            "--min-system-ram-gb",
            "64",
            "--min-cpu-cores",
            "16",
            "--min-download-mbps",
            "500",
            "--min-reliability",
            "0.98",
            "--exclude-machine",
            "142444",
        ],
    )

    assert vast_control.main() == 0
    assert captured == {
        "args": ("RTX_4090", 0.6, 24, 80, 10),
        "kwargs": {
            "min_system_ram_gb": 64.0,
            "min_cpu_cores": 16.0,
            "min_download_mbps": 500.0,
            "min_reliability": 0.98,
            "exclude_machines": (142444,),
        },
    }


def test_confirmed_create_refuses_an_operator_unseen_machine(monkeypatch, tmp_path):
    state_path = tmp_path / "mixstq" / "vast_state.json"
    monkeypatch.setattr(vast_control, "STATE", state_path)
    monkeypatch.setattr(
        vast_control,
        "_search_offers",
        lambda *_args, **_kwargs: pytest.fail("an unpinned confirm must not search"),
    )
    monkeypatch.setattr(
        vast_control,
        "create",
        lambda *_args, **_kwargs: pytest.fail("an unpinned confirm must not create"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["vast-control", "create", "--max-hourly", "0.75", "--confirm"],
    )

    with pytest.raises(vast_control.VastError, match="--machine"):
        vast_control.main()

    assert not state_path.exists()


def test_any_machine_flag_allows_a_confirm_without_a_pinned_machine(monkeypatch, tmp_path):
    state_path = tmp_path / "mixstq" / "vast_state.json"
    monkeypatch.setattr(vast_control, "STATE", state_path)
    monkeypatch.setattr(
        vast_control,
        "_search_offers",
        lambda *_args, **_kwargs: [_offer(offer_id=11, machine_id=71654)],
    )
    created = {}

    def fake_create(offer_id, *_args, **_kwargs):
        created["offer_id"] = offer_id
        return {"success": True, "new_contract": 12}

    monkeypatch.setattr(vast_control, "create", fake_create)
    monkeypatch.setattr(
        sys,
        "argv",
        ["vast-control", "create", "--max-hourly", "0.75", "--any-machine", "--confirm"],
    )

    assert vast_control.main() == 0
    assert created["offer_id"] == 11


def test_price_cap_covers_storage_when_the_offer_exposes_it(monkeypatch):
    constraints = {
        "gpu": "",
        "max_price": 1.20,
        "min_vram": 24,
        "disk": 380,
        "min_system_ram_gb": 0.0,
        "min_cpu_cores": 0.0,
        "min_download_mbps": 0.0,
        "min_reliability": 0.0,
        "exclude_machines": (),
    }
    # $0.20/GB/month over 380 GB is $0.104/hour on Vast's 730-hour month.
    over = _offer(dph_total=1.15, disk_space=400, storage_cost=0.2)
    assert vast_control._offer_violation(over, **constraints) is not None
    under = _offer(dph_total=1.05, disk_space=400, storage_cost=0.2)
    assert vast_control._offer_violation(under, **constraints) is None

    compute_only = _offer(dph_total=1.15, disk_space=400)
    assert vast_control._offer_violation(compute_only, **constraints) is None

    monkeypatch.setattr(
        vast_control, "_request", lambda *_args, **_kwargs: {"offers": [under, compute_only]}
    )
    rows = vast_control.search("", 1.20, 24, 380, 10)
    assert rows[0]["dph_all_in"] == round(1.05 + 0.2 * 380 / 730, 4)
    assert rows[0]["storage_cost_per_gb_month_usd"] == 0.2
    assert rows[1]["dph_all_in"] == 1.15
    assert rows[1]["storage_cost_per_gb_month_usd"] is None


def test_search_reports_machine_id_and_honours_exclusions(monkeypatch):
    monkeypatch.setattr(
        vast_control,
        "_request",
        lambda *_args, **_kwargs: {
            "offers": [
                _offer(offer_id=1, machine_id=142444, dph_total=0.4),
                _offer(offer_id=2, machine_id=71654),
            ]
        },
    )

    rows = vast_control.search("RTX_4090", 0.6, 24, 80, 10)
    assert [(row["id"], row["machine_id"]) for row in rows] == [(1, 142444), (2, 71654)]

    kept = vast_control.search("RTX_4090", 0.6, 24, 80, 10, exclude_machines=(142444,))
    assert [row["id"] for row in kept] == [2]


def test_confirmed_create_rents_the_fresh_offer_id_for_the_same_machine(
    monkeypatch, tmp_path, capsys
):
    state_path = tmp_path / "mixstq" / "vast_state.json"
    monkeypatch.setattr(vast_control, "STATE", state_path)
    created = {}
    # Vast hands out a different chunk id for the same machine on every /bundles call, so the
    # stale id from the operator's earlier search is never in the fresh rentable set.
    monkeypatch.setattr(
        vast_control,
        "_search_offers",
        lambda *_args, **_kwargs: [_offer(offer_id=44937483, machine_id=142444)],
    )

    def fake_create(offer_id, image, disk, onstart):
        created.update(offer_id=offer_id, image=image, disk=disk, onstart=onstart)
        return {"success": True, "new_contract": 99}

    monkeypatch.setattr(vast_control, "create", fake_create)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vast-control",
            "create",
            "--offer",
            "44937497",
            "--max-hourly",
            "0.75",
            "--min-vram",
            "24",
            "--disk",
            "80",
            "--min-reliability",
            "0.98",
            "--confirm",
        ],
    )

    assert vast_control.main() == 0

    assert created["offer_id"] == 44937483
    output = capsys.readouterr().out
    assert "44937483" in output
    assert "142444" in output
    event = json.loads(state_path.read_text(encoding="utf-8"))[0]
    assert event["offer"] == 44937483
    assert event["instance_id"] == 99


def test_confirmed_create_excludes_machines_and_fails_closed_when_none_remain(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "mixstq" / "vast_state.json"
    monkeypatch.setattr(vast_control, "STATE", state_path)
    captured = {}

    def fake_search_offers(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return []

    monkeypatch.setattr(vast_control, "_search_offers", fake_search_offers)
    monkeypatch.setattr(
        vast_control,
        "create",
        lambda *_args, **_kwargs: pytest.fail("create must not run without a fresh offer"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vast-control",
            "create",
            "--max-hourly",
            "0.75",
            "--exclude-machine",
            "142444",
            "--exclude-machine",
            "71654",
            "--any-machine",
            "--confirm",
        ],
    )

    with pytest.raises(vast_control.VastError, match="no rentable offer"):
        vast_control.main()

    assert captured["kwargs"]["exclude_machines"] == (142444, 71654)
    assert not state_path.exists()


def test_confirmed_create_revalidates_the_fresh_offer_and_refuses_a_violation(
    monkeypatch, tmp_path
):
    state_path = tmp_path / "mixstq" / "vast_state.json"
    monkeypatch.setattr(vast_control, "STATE", state_path)
    monkeypatch.setattr(
        vast_control,
        "_search_offers",
        lambda *_args, **_kwargs: [_offer(offer_id=7, machine_id=1, dph_total=0.9)],
    )
    monkeypatch.setattr(
        vast_control,
        "create",
        lambda *_args, **_kwargs: pytest.fail("create must not run after failed revalidation"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["vast-control", "create", "--max-hourly", "0.75", "--any-machine", "--confirm"],
    )

    with pytest.raises(vast_control.VastError, match="revalidation"):
        vast_control.main()

    assert not state_path.exists()


@pytest.mark.parametrize(
    ("field", "value", "constraints"),
    [
        ("dph_total", 0.9, {}),
        ("gpu_ram", 23_000, {}),
        ("disk_space", 40, {}),
        ("num_gpus", 2, {}),
        ("cpu_ram", 32_000, {"min_system_ram_gb": 64}),
        ("cpu_cores", 8, {"min_cpu_cores": 16}),
        ("inet_down", 100.0, {"min_download_mbps": 500}),
        ("reliability2", 0.90, {"min_reliability": 0.98}),
        ("machine_id", 142444, {"exclude_machines": (142444,)}),
    ],
)
def test_offer_violation_names_every_constraint_it_rejects(field, value, constraints):
    offer = _offer(**{"machine_id": 71654, field: value})
    arguments = {
        "gpu": "",
        "max_price": 0.75,
        "min_vram": 24,
        "disk": 80,
        "min_system_ram_gb": 0.0,
        "min_cpu_cores": 0.0,
        "min_download_mbps": 0.0,
        "min_reliability": 0.0,
        "exclude_machines": (),
        **constraints,
    }

    assert vast_control._offer_violation(offer, **arguments) is not None
    assert vast_control._offer_violation(_offer(machine_id=71654), **arguments) is None


def test_create_confirmation_revalidates_all_requested_constraints(monkeypatch):
    captured = {}

    def fake_search(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return []

    monkeypatch.setattr(vast_control, "_search_offers", fake_search)
    monkeypatch.setattr(
        vast_control,
        "create",
        lambda *_args, **_kwargs: pytest.fail("create must not run after failed revalidation"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vast-control",
            "create",
            "--offer",
            "123",
            "--max-hourly",
            "0.75",
            "--min-vram",
            "24",
            "--disk",
            "120",
            "--min-system-ram-gb",
            "64",
            "--min-cpu-cores",
            "16",
            "--min-download-mbps",
            "500",
            "--min-reliability",
            "0.98",
            "--confirm",
        ],
    )

    with pytest.raises(vast_control.VastError, match="no rentable offer"):
        vast_control.main()

    assert captured == {
        "args": ("", 0.75, 24, 120, 200),
        "kwargs": {
            "min_system_ram_gb": 64.0,
            "min_cpu_cores": 16.0,
            "min_download_mbps": 500.0,
            "min_reliability": 0.98,
            "exclude_machines": (),
        },
    }


def test_create_dry_run_keeps_hourly_cap_and_has_no_side_effect(
    monkeypatch, tmp_path, capsys
):
    state_path = tmp_path / "mixstq" / "vast_state.json"
    monkeypatch.setattr(vast_control, "STATE", state_path)
    monkeypatch.setattr(
        vast_control,
        "_search_offers",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not search or revalidate"),
    )
    monkeypatch.setattr(
        vast_control,
        "create",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not create"),
    )
    monkeypatch.setattr(
        vast_control,
        "_save",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not write state"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vast-control",
            "create",
            "--offer",
            "123",
            "--max-hourly",
            "0.75",
            "--min-system-ram-gb",
            "64",
        ],
    )

    assert vast_control.main() == 0

    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "hourly cap 0.7500" in output
    assert not state_path.exists()
