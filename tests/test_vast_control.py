import sys

import pytest
import vast_control


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
        },
    }


def test_create_confirmation_revalidates_all_requested_constraints(monkeypatch):
    captured = {}

    def fake_search(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return []

    monkeypatch.setattr(vast_control, "search", fake_search)
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

    with pytest.raises(vast_control.VastError, match="offer 123 not in the rentable set"):
        vast_control.main()

    assert captured == {
        "args": ("", 0.75, 24, 120, 200),
        "kwargs": {
            "min_system_ram_gb": 64.0,
            "min_cpu_cores": 16.0,
            "min_download_mbps": 500.0,
            "min_reliability": 0.98,
        },
    }


def test_create_dry_run_keeps_hourly_cap_and_has_no_api_side_effect(monkeypatch, capsys):
    monkeypatch.setattr(
        vast_control,
        "search",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not search or revalidate"),
    )
    monkeypatch.setattr(
        vast_control,
        "create",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not create"),
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
