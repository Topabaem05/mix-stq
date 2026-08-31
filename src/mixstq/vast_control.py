from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

API = "https://console.vast.ai/api/v0"
STATE = Path(__file__).with_name("vast_state.json")


class VastError(RuntimeError):
    pass


def _key() -> str:
    key = os.environ.get("MIXSTQ_VAST_KEY", "").strip()
    if key:
        return key
    for candidate in (Path.home() / ".vast_api_key", Path.home() / ".config/vastai/vast_api_key"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    raise VastError(
        "no vast.ai key found: set MIXSTQ_VAST_KEY or create ~/.vast_api_key"
    )


def _request(method: str, path: str, payload: dict | None = None, timeout: int = 90) -> dict:
    url = API + path
    command = ["curl", "-sS", "-m", str(timeout), "-X", method, url,
               "-H", "Authorization: Bearer " + _key(),
               "-H", "Content-Type: application/json"]
    if payload is not None:
        command += ["-d", json.dumps(payload)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise VastError("curl failed: " + result.stderr.strip())
    body = result.stdout.strip()
    if not body:
        raise VastError("empty response from " + path)
    if body.lstrip().startswith("<"):
        raise VastError(
            "API returned HTML rather than JSON for %s; the key is probably invalid or revoked"
            % path
        )
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise VastError("non-JSON response from %s: %s" % (path, body[:300])) from error


def _number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _offer_reliability(offer: dict) -> int | float | None:
    reliability = _number(offer.get("reliability2"))
    return reliability if reliability is not None else _number(offer.get("reliability"))


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
        if not offer.get("rentable", False):
            continue
        if _number(offer.get("num_gpus")) != 1:
            continue
        gpu_ram = _number(offer.get("gpu_ram"))
        if gpu_ram is None or gpu_ram < min_vram * 1000:
            continue
        dph_total = _number(offer.get("dph_total"))
        if dph_total is None or dph_total > max_price:
            continue
        disk_space = _number(offer.get("disk_space"))
        if disk_space is None or disk_space < disk:
            continue
        if gpu and offer.get("gpu_name") != gpu.replace("_", " "):
            continue
        cpu_ram = _number(offer.get("cpu_ram"))
        if min_system_ram_gb > 0 and (
            cpu_ram is None or cpu_ram < min_system_ram_gb * 1000
        ):
            continue
        cpu_cores = _number(offer.get("cpu_cores"))
        if min_cpu_cores > 0 and (cpu_cores is None or cpu_cores < min_cpu_cores):
            continue
        inet_down = _number(offer.get("inet_down"))
        if min_download_mbps > 0 and (
            inet_down is None or inet_down < min_download_mbps
        ):
            continue
        reliability = _offer_reliability(offer)
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
    if not response.get("success"):
        raise VastError("create failed: " + json.dumps(response)[:300])
    return response


def instances() -> list[dict]:
    response = _request("GET", "/instances/")
    if "instances" not in response:
        raise VastError(
            "unexpected /instances/ payload; key may be invalid: " + json.dumps(response)[:200]
        )
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
    return _request("DELETE", "/instances/%d/" % instance_id, {})


def _save(record: dict) -> None:
    history = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else []
    history.append(record)
    STATE.write_text(json.dumps(history, indent=1), encoding="utf-8")


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
        print(json.dumps(response, indent=1))
        return 0

    if args.command == "list":
        rows = instances()
        print(json.dumps(rows, indent=1))
        print("my total burn so far: $%.4f" % sum(r["my_cost_so_far"] for r in rows))
        return 0

    if args.command == "destroy":
        if not args.confirm:
            print("DRY RUN. would destroy instance %d; re-run with --confirm" % args.id)
            return 0
        response = destroy(args.id)
        _save({"event": "destroy", "at": time.time(), "id": args.id, "response": response})
        print(json.dumps(response, indent=1))
        return 0

    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VastError as error:
        print("VAST ERROR: " + str(error), file=sys.stderr)
        raise SystemExit(2) from error
