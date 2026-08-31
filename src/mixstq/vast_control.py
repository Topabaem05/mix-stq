from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://console.vast.ai/api/v0"


def _default_state_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home).expanduser() if state_home else Path.home() / ".local/state"
    return base / "mixstq/vast_state.json"


STATE = _default_state_path()


class VastError(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("MIXSTQ_VAST_KEY", "").strip()
    if key:
        return key
    for candidate in (Path.home() / ".vast_api_key", Path.home() / ".config/vastai/vast_api_key"):
        if candidate.is_file():
            key = candidate.read_text(encoding="utf-8").strip()
            if key:
                return key
    raise VastError(
        "no vast.ai key found: set MIXSTQ_VAST_KEY or create ~/.vast_api_key"
    )


def _https_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _redact_secret(value: object, secret: str) -> object:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, list):
        return [_redact_secret(item, secret) for item in value]
    if isinstance(value, dict):
        return {
            _redact_secret(key, secret): _redact_secret(item, secret)
            for key, item in value.items()
        }
    return value


def _request(method: str, path: str, payload: dict | None = None, timeout: int = 90) -> dict:
    _require_positive("timeout", timeout)
    url = API + path
    key = _key()
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=_https_context()
        ) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise VastError(
            "Vast API request failed with HTTP %d for %s" % (error.code, path)
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise VastError("Vast API request failed for " + path) from None
    if not body:
        raise VastError("Vast API returned an empty response for " + path)
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise VastError("Vast API returned invalid JSON for " + path) from None
    if not isinstance(parsed, dict):
        raise VastError("Vast API returned an unexpected payload for " + path)
    return _redact_secret(parsed, key)


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _require_finite(name: str, value: object) -> int | float:
    number = _number(value)
    if number is None:
        raise VastError(name + " must be a finite number")
    return number


def _require_positive(name: str, value: object) -> int | float:
    number = _require_finite(name, value)
    if number <= 0:
        raise VastError(name + " must be greater than zero")
    return number


def _require_nonnegative(name: str, value: object) -> int | float:
    number = _require_finite(name, value)
    if number < 0:
        raise VastError(name + " must be zero or greater")
    return number


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VastError(name + " must be a positive integer")
    return value


def _validate_search_inputs(
    max_price: object,
    min_vram: object,
    disk: object,
    limit: object,
    min_system_ram_gb: object,
    min_cpu_cores: object,
    min_download_mbps: object,
    min_reliability: object,
) -> None:
    _require_positive("max_price", max_price)
    _require_positive_integer("min_vram", min_vram)
    _require_positive_integer("disk", disk)
    _require_positive_integer("limit", limit)
    _require_nonnegative("min_system_ram_gb", min_system_ram_gb)
    _require_nonnegative("min_cpu_cores", min_cpu_cores)
    _require_nonnegative("min_download_mbps", min_download_mbps)
    reliability = _require_finite("min_reliability", min_reliability)
    if not 0 <= reliability <= 1:
        raise VastError("min_reliability must be between zero and one")


def _offer_reliability(offer: dict) -> int | float | None:
    reliability = offer.get("reliability2")
    if reliability is not None:
        return _number(reliability)
    return _number(offer.get("reliability"))


def search(
    gpu: str,
    max_price: float,
    min_vram: int,
    disk: int,
    limit: int,
    min_system_ram_gb: float = 0.0,
    min_cpu_cores: float = 0.0,
    min_download_mbps: float = 0.0,
    min_reliability: float = 0.0,
) -> list[dict]:
    _validate_search_inputs(
        max_price,
        min_vram,
        disk,
        limit,
        min_system_ram_gb,
        min_cpu_cores,
        min_download_mbps,
        min_reliability,
    )
    query = {
        "rentable": {"eq": True},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": min_vram * 1000},
        "dph_total": {"lte": max_price},
        "disk_space": {"gte": disk},
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": max(limit * 4, 32),
    }
    if min_system_ram_gb > 0:
        query["cpu_ram"] = {"gte": min_system_ram_gb * 1000}
    if min_cpu_cores > 0:
        query["cpu_cores"] = {"gte": min_cpu_cores}
    if min_download_mbps > 0:
        query["inet_down"] = {"gte": min_download_mbps}
    if min_reliability > 0:
        query["reliability"] = {"gte": min_reliability}
    response = _request("POST", "/bundles", query)
    offers = response.get("offers", [])
    if not isinstance(offers, list):
        raise VastError("unexpected /bundles/ payload; key may be invalid")
    selected = []
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        offer_id = offer.get("id")
        if (
            isinstance(offer_id, bool)
            or not isinstance(offer_id, int)
            or offer_id <= 0
        ):
            continue
        if offer.get("rentable") is not True:
            continue
        if _number(offer.get("num_gpus")) != 1:
            continue
        gpu_ram = _number(offer.get("gpu_ram"))
        if gpu_ram is None or gpu_ram <= 0 or gpu_ram < min_vram * 1000:
            continue
        dph_total = _number(offer.get("dph_total"))
        if dph_total is None or dph_total <= 0 or dph_total > max_price:
            continue
        disk_space = _number(offer.get("disk_space"))
        if disk_space is None or disk_space <= 0 or disk_space < disk:
            continue
        if gpu and offer.get("gpu_name") != gpu.replace("_", " "):
            continue
        cpu_ram = _number(offer.get("cpu_ram"))
        if offer.get("cpu_ram") is not None and (cpu_ram is None or cpu_ram < 0):
            continue
        if min_system_ram_gb > 0 and (
            cpu_ram is None or cpu_ram < min_system_ram_gb * 1000
        ):
            continue
        cpu_cores = _number(offer.get("cpu_cores"))
        if offer.get("cpu_cores") is not None and (
            cpu_cores is None or cpu_cores < 0
        ):
            continue
        if min_cpu_cores > 0 and (cpu_cores is None or cpu_cores < min_cpu_cores):
            continue
        inet_down = _number(offer.get("inet_down"))
        if offer.get("inet_down") is not None and (
            inet_down is None or inet_down < 0
        ):
            continue
        if min_download_mbps > 0 and (
            inet_down is None or inet_down < min_download_mbps
        ):
            continue
        reliability = _offer_reliability(offer)
        has_reliability = (
            offer.get("reliability2") is not None
            or offer.get("reliability") is not None
        )
        if has_reliability and (
            reliability is None or not 0 <= reliability <= 1
        ):
            continue
        if min_reliability > 0 and (
            reliability is None or reliability < min_reliability
        ):
            continue
        selected.append(offer)
    selected.sort(key=lambda o: _number(o.get("dph_total")) or 1e9)
    selected = selected[:limit]
    return [
        {
            "id": offer["id"],
            "gpu": offer.get("gpu_name"),
            "num_gpus": offer.get("num_gpus"),
            "gpu_ram_gb": round((offer.get("gpu_ram") or 0) / 1024, 1),
            "dph": round(offer.get("dph_total") or 0.0, 4),
            "disk_gb": offer.get("disk_space"),
            "system_ram_gb": (_number(offer.get("cpu_ram")) or 0) / 1000,
            "cpu_cores": _number(offer.get("cpu_cores")) or 0,
            "down_mbps": round(_number(offer.get("inet_down")) or 0.0, 1),
            "reliability": round(_offer_reliability(offer) or 0.0, 4),
            "geo": offer.get("geolocation"),
        }
        for offer in selected
    ]


def create(offer_id: int, image: str, disk: int, onstart: str | None) -> dict:
    _require_positive_integer("offer_id", offer_id)
    _require_positive_integer("disk", disk)
    payload = {
        "client_id": "me",
        "image": image,
        "disk": disk,
        "runtype": "ssh",
        "ssh": True,
        "direct": True,
    }
    if onstart:
        payload["onstart_cmd"] = onstart
    response = _request("PUT", "/asks/%d/" % offer_id, payload)
    if response.get("success") is not True:
        raise VastError("Vast instance creation was rejected")
    return response


def instances() -> list[dict]:
    response = _request("GET", "/instances/")
    if "instances" not in response:
        raise VastError("unexpected /instances/ payload")
    rows = []
    for inst in response.get("instances", []):
        started = inst.get("start_date") or 0.0
        rental_seconds = max(time.time() - started, 0.0) if started else 0.0
        hourly = inst.get("dph_total") or 0.0
        rows.append({
            "id": inst.get("id"),
            "status": inst.get("actual_status"),
            "intended": inst.get("intended_status"),
            "gpu": inst.get("gpu_name"),
            "dph": round(hourly, 4),
            "ssh_host": inst.get("ssh_host"),
            "ssh_port": inst.get("ssh_port"),
            "my_rental_min": round(rental_seconds / 60.0, 2),
            "my_cost_so_far": round((rental_seconds / 3600.0) * hourly, 4),
        })
    return rows


def destroy(instance_id: int) -> dict:
    _require_positive_integer("instance_id", instance_id)
    return _request("DELETE", "/instances/%d/" % instance_id, {})


def _instance_id(response: dict) -> int | None:
    for field in ("new_contract", "instance_id", "id"):
        value = response.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _sanitize_event(record: object) -> dict:
    if not isinstance(record, dict):
        raise VastError("Vast state contains an invalid event")
    event = record.get("event")
    at = _require_positive("event timestamp", record.get("at"))
    if event == "create":
        sanitized = {
            "event": "create",
            "at": at,
            "offer": _require_positive_integer("event offer", record.get("offer")),
        }
        instance_id = record.get("instance_id")
        response = record.get("response")
        if instance_id is None and isinstance(response, dict):
            instance_id = _instance_id(response)
        if instance_id is not None:
            sanitized["instance_id"] = _require_positive_integer(
                "event instance_id", instance_id
            )
        return sanitized
    if event == "destroy":
        return {
            "event": "destroy",
            "at": at,
            "id": _require_positive_integer("event id", record.get("id")),
        }
    raise VastError("Vast state contains an unknown event")


def _save(record: dict, state_path: str | Path | None = None) -> None:
    path = Path(state_path).expanduser() if state_path is not None else STATE
    try:
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise VastError("Vast state does not contain an event list")
            history = [_sanitize_event(item) for item in loaded]
        else:
            history = []
        history.append(_sanitize_event(record))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise VastError("unable to read Vast state") from None

    parent_existed = path.parent.exists()
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if state_path is None or not parent_existed:
            path.parent.chmod(0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix="." + path.name + "."
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(history, handle, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
        temporary_path = None
    except OSError:
        raise VastError("unable to update Vast state") from None
    finally:
        if temporary_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary_path.unlink()


def _create_summary(response: dict) -> dict:
    summary = {"success": response.get("success") is True}
    instance_id = _instance_id(response)
    if instance_id is not None:
        summary["instance_id"] = instance_id
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="MIX-STQ vast.ai control")
    sub = parser.add_subparsers(dest="command", required=True)

    find = sub.add_parser("search")
    find.add_argument("--gpu", default="RTX_4090")
    find.add_argument("--max-price", type=float, default=0.60)
    find.add_argument("--min-vram", type=int, default=24)
    find.add_argument("--disk", type=int, default=80)
    find.add_argument("--limit", type=int, default=10)
    find.add_argument("--min-system-ram-gb", type=float, default=0.0,
                      help="minimum system RAM in decimal GB (Vast cpu_ram MB / 1000)")
    find.add_argument("--min-cpu-cores", type=float, default=0.0,
                      help="minimum Vast cpu_cores value")
    find.add_argument("--min-download-mbps", type=float, default=0.0,
                      help="minimum Vast inet_down value in Mbps")
    find.add_argument("--min-reliability", type=float, default=0.0,
                      help="minimum Vast offer reliability fraction (for example, 0.98)")

    up = sub.add_parser("create")
    up.add_argument("--offer", type=int, required=True)
    up.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    up.add_argument("--disk", type=int, default=80)
    up.add_argument("--onstart", default=None)
    up.add_argument("--max-hourly", type=float, required=True)
    up.add_argument("--min-vram", type=int, default=24)
    up.add_argument("--min-system-ram-gb", type=float, default=0.0,
                    help="minimum system RAM in decimal GB (Vast cpu_ram MB / 1000)")
    up.add_argument("--min-cpu-cores", type=float, default=0.0,
                    help="minimum Vast cpu_cores value")
    up.add_argument("--min-download-mbps", type=float, default=0.0,
                    help="minimum Vast inet_down value in Mbps")
    up.add_argument("--min-reliability", type=float, default=0.0,
                    help="minimum Vast offer reliability fraction (for example, 0.98)")
    up.add_argument("--confirm", action="store_true")

    sub.add_parser("list")

    down = sub.add_parser("destroy")
    down.add_argument("--id", type=int, required=True)
    down.add_argument("--confirm", action="store_true")

    args = parser.parse_args()

    if args.command == "search":
        offers = search(
            args.gpu,
            args.max_price,
            args.min_vram,
            args.disk,
            args.limit,
            min_system_ram_gb=args.min_system_ram_gb,
            min_cpu_cores=args.min_cpu_cores,
            min_download_mbps=args.min_download_mbps,
            min_reliability=args.min_reliability,
        )
        print(json.dumps(offers, indent=1))
        if offers:
            cheapest = offers[0]
            print("cheapest: id=%s %s %.4f $/hr -> 2h = $%.2f" % (
                cheapest["id"], cheapest["gpu"], cheapest["dph"], 2 * cheapest["dph"]))
        return 0

    if args.command == "create":
        _require_positive_integer("offer", args.offer)
        _validate_search_inputs(
            args.max_hourly,
            args.min_vram,
            args.disk,
            200,
            args.min_system_ram_gb,
            args.min_cpu_cores,
            args.min_download_mbps,
            args.min_reliability,
        )
        if not args.confirm:
            print("DRY RUN. would create offer %d with hourly cap %.4f" % (args.offer, args.max_hourly))
            print("re-run with --confirm to spend money")
            return 0
        offers = {
            offer["id"]: offer
            for offer in search(
                "",
                args.max_hourly,
                args.min_vram,
                args.disk,
                200,
                min_system_ram_gb=args.min_system_ram_gb,
                min_cpu_cores=args.min_cpu_cores,
                min_download_mbps=args.min_download_mbps,
                min_reliability=args.min_reliability,
            )
        }
        offer = offers.get(args.offer)
        if offer is None:
            raise VastError(
                "offer %d not in the rentable set at cap %.4f $/hr, %d GB VRAM, %d GB disk; "
                "re-run search" % (args.offer, args.max_hourly, args.min_vram, args.disk)
            )
        if offer["dph"] > args.max_hourly:
            raise VastError(
                "offer %.4f $/hr exceeds --max-hourly %.4f" % (offer["dph"], args.max_hourly)
            )
        response = create(args.offer, args.image, args.disk, args.onstart)
        _save({"event": "create", "at": time.time(), "offer": args.offer, "response": response})
        print(json.dumps(_create_summary(response), indent=1))
        return 0

    if args.command == "list":
        rows = instances()
        print(json.dumps(rows, indent=1))
        print("my total burn so far: $%.4f" % sum(r["my_cost_so_far"] for r in rows))
        return 0

    if args.command == "destroy":
        _require_positive_integer("instance id", args.id)
        if not args.confirm:
            print("DRY RUN. would destroy instance %d; re-run with --confirm" % args.id)
            return 0
        response = destroy(args.id)
        _save({"event": "destroy", "at": time.time(), "id": args.id, "response": response})
        print(json.dumps({"id": args.id, "success": response.get("success") is True}, indent=1))
        return 0

    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VastError as error:
        print("VAST ERROR: " + str(error), file=sys.stderr)
        raise SystemExit(2) from error
