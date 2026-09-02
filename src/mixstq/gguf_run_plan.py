"""Pinned, credential-free Qwen3.8-27B GGUF experiment planning and preflights."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from . import llama_calibration
else:
    llama_calibration = importlib.import_module("llama_calibration")


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LLAMA_CPP_COMMIT = "580e88d8b7dece7099d9b62323521d0254ff3615"
REPOSITORY_REMOTE = "https://github.com/Topabaem05/mix-stq.git"
LLAMA_CPP_REMOTE = "https://github.com/ggml-org/llama.cpp.git"
ARTIFACT_RELATIVE = ("mix-stq", "artifacts", "qwen38-gguf-v27")
PHASES = (
    "bootstrap",
    "calibration",
    "convert",
    "imatrix",
    "quantize",
    "smoke",
    "audit",
    "split",
    "upload",
)
TIERS = ("IQ3_XXS", "IQ4_XS", "Q4_K_M", "Q5_K_M")
MODEL_NAMES = ("BF16", *TIERS)
REQUIRED_EXECUTABLES = (
    "llama-imatrix",
    "llama-quantize",
    "llama-cli",
    "llama-server",
    "llama-perplexity",
    "llama-bench",
    "llama-tokenize",
    "llama-gguf-split",
)
# Run 1 built and probed exactly these five and recorded their help surface: four exit 0 and
# llama-quantize exits 1 after printing usage. The three Task 6 binaries were never built, so
# nothing observed says they print "usage"; demanding it there would be an unvalidated gate.
LEDGER_VERIFIED_EXECUTABLES = (
    "llama-imatrix",
    "llama-quantize",
    "llama-cli",
    "llama-tokenize",
    "llama-gguf-split",
)
MODEL_SNAPSHOT_FILES = (
    "config.json",
    "README.md",
    "tokenizer.json",
    "tokenizer_config.json",
)
RUN_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
TOKEN_COUNT_PATTERN = re.compile(r"Total number of tokens: ([0-9]+)\Z")
EXPECTED_LAYER_PATTERN = ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 16
HOST_REQUIRED_KEYS = (
    "gpu_count",
    "gpu_vendor",
    "advertised_gpu_vram_mib",
    "observed_gpu_vram_mib",
    "ram_gb",
    "cpu_cores",
    "workspace_free_gb",
    "active_compute_pids",
    "offer_download_mbps",
    "offer_reliability",
    "offer_hourly_usd",
)
GPU_COUNT_REQUIRED = 1
GPU_VRAM_MIN_MIB = 80_000
RAM_MIN_DECIMAL_GB = 96
CPU_CORES_MIN = 16
WORKSPACE_MIN_DECIMAL_GB = 300
OFFER_DOWNLOAD_MIN_MBPS = 500
OFFER_RELIABILITY_MIN = 0.98
OFFER_HOURLY_MAX_USD = 1.20
IMATRIX_CHUNKS = 128
IMATRIX_CONTEXT_TOKENS = 512
IMATRIX_BATCH_TOKENS = 512
IMATRIX_UBATCH_TOKENS = 128
IMATRIX_THREADS = 32
IMATRIX_TOKEN_CAPACITY = IMATRIX_CHUNKS * IMATRIX_CONTEXT_TOKENS
SMOKE_SEED = 22
SMOKE_PREDICT_TOKENS = 16
SPLIT_MAX_DECIMAL_SIZE = "8G"
HF_DATASET_REPO = "topabaem/mix-stq-artifacts"
HF_UPLOAD_PREFIX = "paid-run/qwen38-gguf-frontier-v27"
EVIDENCE_SOURCES = (
    "calibration",
    "imatrix",
    "smoke",
    "preflight",
    "audits",
    "wikitext",
)
# Run 1 wrote per-arm results to eval/ (singular) and never created a bench directory, so the
# operator and the planner must agree on these names. All three are required: measuring them is
# the point of the second paid run.
TASK6_EVIDENCE_SOURCES = ("eval", "ppl", "bench")
TASK6_COMPLETE_MARKER = "eval/.task6-complete"
CONVERTER_RUNNER = (
    "import runpy,sys;"
    "from pathlib import Path;"
    "root=Path(sys.argv[1]);"
    "matches=[p for p in root.glob('convert_*_to_gguf.py') if p.name.split('_')[1]=='hf'];"
    "len(matches)==1 or sys.exit('pinned converter surface unavailable');"
    "sys.path.insert(0,str(root));"
    "sys.argv=[str(matches[0]),*sys.argv[2:]];"
    "runpy.run_path(str(matches[0]),run_name='__main__')"
)
# llama-quantize prints its usage and then exits 1 at the pinned llama.cpp commit, so a bare
# `--help` probe fails bootstrap for a binary that is present and self-documenting. Every probe
# requires one of the two documented usage exit codes; only a probe over a binary whose help
# surface the ledger actually recorded also requires the usage text itself.
HELP_PROBE_RUNNER = (
    "import subprocess,sys;"
    "mode,executable=sys.argv[1],sys.argv[2];"
    "mode in ('strict','lenient') or sys.exit('unknown help probe mode');"
    "completed=subprocess.run([executable,'--help'],stdin=subprocess.DEVNULL,"
    "stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',timeout=120);"
    "text=completed.stdout;"
    "completed.returncode in (0,1) or sys.exit('pinned executable help probe failed');"
    "'usage' in text.lower() or mode == 'lenient' or sys.exit('printed no usage text');"
    "'usage' in text.lower() or print('warning: ' + executable + ' printed no usage text')"
)
# Gather the small artifacts into one evidence directory. A required source must exist and hold at
# least one file, so an upload cannot silently publish an empty or missing result set; only the
# planner's --allow-missing-task6-evidence escape marks the Task 6 directories optional.
EVIDENCE_ASSEMBLY_RUNNER = (
    "import shutil,sys\n"
    "from pathlib import Path\n"
    "evidence = Path(sys.argv[1])\n"
    "entries = sys.argv[2:]\n"
    "entries and len(entries) % 2 == 0 or sys.exit('expected mode/path pairs')\n"
    "evidence.mkdir(parents=True, exist_ok=True)\n"
    "gathered = 0\n"
    "for mode, argument in zip(entries[::2], entries[1::2]):\n"
    "    source = Path(argument)\n"
    "    populated = source.is_dir() and any(p.is_file() for p in source.rglob('*'))\n"
    "    if not populated:\n"
    "        mode == 'optional' or sys.exit('required evidence is missing or empty: ' + source.name)\n"
    "        print('skipped absent optional ' + source.name)\n"
    "        continue\n"
    "    shutil.copytree(source, evidence / source.name, dirs_exist_ok=True)\n"
    "    gathered += 1\n"
    "    print('gathered ' + source.name)\n"
    "gathered or sys.exit('no evidence directory was available to gather')\n"
)
# Amendment 3 moves the mandatory public verification onto the host. The host CLI is logged in,
# so stripping the environment is not enough: huggingface_hub falls back to the token file at
# $HF_HOME/token, which defaults to $HOME/.cache/huggingface/token. The re-download therefore
# runs with the environment stripped, implicit tokens disabled, and HOME/HF_HOME/XDG pointed at
# a fresh empty sandbox, so no stored credential is reachable. No credential is ever named in
# the planner output.
UNAUTHENTICATED_RUNNER = (
    "import os,sys,tempfile\n"
    "for name in list(os.environ):\n"
    "    upper = name.upper()\n"
    "    if 'TOKEN' in upper or upper.startswith('HF_') or upper.startswith('HUGGINGFACE'):\n"
    "        os.environ.pop(name, None)\n"
    "len(sys.argv) > 2 or sys.exit('usage: <sandbox-root> <command> [argument ...]')\n"
    "root = sys.argv[1]\n"
    "os.makedirs(root, exist_ok=True)\n"
    "sandbox = tempfile.mkdtemp(prefix='unauthenticated-', dir=root)\n"
    "os.environ['HOME'] = sandbox\n"
    "os.environ['USERPROFILE'] = sandbox\n"
    "os.environ['XDG_CACHE_HOME'] = os.path.join(sandbox, 'cache')\n"
    "os.environ['XDG_CONFIG_HOME'] = os.path.join(sandbox, 'config')\n"
    "os.environ['XDG_DATA_HOME'] = os.path.join(sandbox, 'data')\n"
    "os.environ['HF_HOME'] = os.path.join(sandbox, 'huggingface')\n"
    "os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'\n"
    # Without this the xet chunk cache keeps a second copy of every file under $HF_HOME/xet.
    "os.environ['HF_HUB_DISABLE_XET'] = '1'\n"
    "os.execv(sys.argv[2], sys.argv[2:])\n"
)
# urlopen raises on any non-2xx HTTPS response, so an unraised status is the success case; the
# check that carries weight is that an unauthenticated reader sees private == False.
PUBLIC_REPO_CHECK_RUNNER = (
    "import json,sys,urllib.error,urllib.request\n"
    "try:\n"
    "    with urllib.request.urlopen(sys.argv[1], timeout=60) as response:\n"
    "        status = getattr(response, 'status', None)\n"
    "        body = response.read()\n"
    "except urllib.error.HTTPError as error:\n"
    "    sys.exit('repository metadata returned HTTP ' + str(error.code))\n"
    "except (urllib.error.URLError, TimeoutError, OSError) as error:\n"
    "    sys.exit('repository metadata is unreachable: ' + type(error).__name__)\n"
    "status in (200, None) or sys.exit('repository metadata returned HTTP ' + str(status))\n"
    "try:\n"
    "    metadata = json.loads(body.decode('utf-8'))\n"
    "except (UnicodeDecodeError, ValueError):\n"
    "    sys.exit('repository metadata is not readable JSON')\n"
    "isinstance(metadata, dict) or sys.exit('repository metadata is not an object')\n"
    "metadata.get('private') is False or sys.exit('dataset repository is not public')\n"
    # A gated repository needs an accepted agreement, so its artifacts are not publicly fetchable.
    "metadata.get('gated') in (False, None) or sys.exit('dataset repository is gated')\n"
    "isinstance(metadata.get('id'), str) or sys.exit('repository metadata is malformed')\n"
    "print('public dataset repository confirmed: ' + metadata['id'])\n"
)
# Verify each uploaded object against its public copy one file at a time: fetch a single
# path-in-repo anonymously, compare sha256, release it, then move to the next. The extra disk the
# verification needs therefore never exceeds one shard, and the CLI's local .cache skeleton is
# removed with the verification tree.
PUBLIC_VERIFY_RUNNER = (
    "import hashlib,shutil,subprocess,sys\n"
    "from pathlib import Path\n"
    "def digest(path):\n"
    "    value = hashlib.sha256()\n"
    "    with path.open('rb') as stream:\n"
    "        for chunk in iter(lambda: stream.read(1048576), b''):\n"
    "            value.update(chunk)\n"
    "    return value.hexdigest()\n"
    "python, runner, sandbox, cli, repo, prefix, source, verify = sys.argv[1:9]\n"
    "source = Path(source)\n"
    "verify = Path(verify)\n"
    "files = sorted(path for path in source.rglob('*') if path.is_file())\n"
    "files or sys.exit('nothing to verify under ' + str(source))\n"
    "for path in files:\n"
    "    remote = prefix + '/' + path.relative_to(source).as_posix()\n"
    "    completed = subprocess.run([python, '-c', runner, sandbox, cli, 'download', repo,\n"
    "                                remote, '--repo-type', 'dataset', '--local-dir', str(verify)],\n"
    "                               check=False)\n"
    "    completed.returncode == 0 or sys.exit('public re-download failed for ' + remote)\n"
    "    public = verify / remote\n"
    "    public.is_file() or sys.exit('public re-download is missing ' + remote)\n"
    "    digest(path) == digest(public) or sys.exit('public sha256 mismatch for ' + remote)\n"
    "    print('verified ' + remote)\n"
    "    public.unlink()\n"
    "    shutil.rmtree(sandbox, ignore_errors=True)\n"
    "shutil.rmtree(verify, ignore_errors=True)\n"
    "print('verified ' + str(len(files)) + ' files against the public copy')\n"
)
SENSITIVE_TEXT_MARKERS = (
    "hf_",
    "HF_TOKEN",
    "MIXSTQ_VAST_KEY",
    "--token",
    "Authorization",
    "/.superpowers/",
    "/.codex/",
    "/.fablize/",
)


class PreflightError(ValueError):
    pass


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, "error: invalid arguments\n")


def _contains_sensitive_text(value: str) -> bool:
    return any(marker in value for marker in SENSITIVE_TEXT_MARKERS)


def _validate_workspace(workspace: Path) -> Path:
    path = Path(workspace)
    normalized = Path(os.path.normpath(os.fspath(path)))
    if not path.is_absolute() or path != normalized or path == Path("/"):
        raise PreflightError("workspace must be an absolute normalized non-root path")
    if _contains_sensitive_text(os.fspath(path)):
        raise PreflightError("workspace is not safe for public planner output")
    return path


def _validate_run_commit(run_commit: str) -> str:
    if not isinstance(run_commit, str) or RUN_COMMIT_PATTERN.fullmatch(run_commit) is None:
        raise PreflightError("run_commit must be exactly 40 lowercase hexadecimal characters")
    return run_commit


def _artifact_root(workspace: Path) -> Path:
    root = workspace.joinpath(*ARTIFACT_RELATIVE)
    if not root.is_relative_to(workspace):
        raise PreflightError("artifact root escaped workspace")
    return root


def _paths(workspace: Path) -> dict[str, Path]:
    root = _artifact_root(workspace)
    repo = workspace / "mix-stq"
    llama_source = root / "source" / "llama.cpp"
    build = llama_source / "build"
    venv = root / "venv"
    paths = {
        "root": root,
        "repo": repo,
        "llama_source": llama_source,
        "build": build,
        "python": venv / "bin" / "python",
        "hf": venv / "bin" / "hf",
        "model_snapshot": root / "model-snapshot",
        "host_snapshot": root / "preflight" / "host-snapshot.json",
        "corpus": root / "calibration" / "corpus.txt",
        "manifest": root / "calibration" / "manifest.json",
        "bf16": root / "models" / "qwen38-27b-bf16.gguf",
        "imatrix": root / "imatrix" / "qwen38-27b.imatrix.gguf",
        "projector": root / "projector" / "qwen38-27b-mmproj-bf16.gguf",
        "projector_dir": root / "projector",
        "evidence": root / "evidence",
        "public_verify": root / "public-verify",
        "unauthenticated_home": root / "public-verify" / "unauthenticated-home",
    }
    for tier in TIERS:
        slug = tier.lower()
        paths[f"model_{tier}"] = root / "models" / f"qwen38-27b-{slug}.gguf"
        paths[f"smoke_{tier}"] = root / "smoke" / f"{slug}.txt"
        paths[f"split_dir_{tier}"] = root / "splits" / slug
        paths[f"split_{tier}"] = root / "splits" / slug / f"qwen38-27b-{slug}"
    paths["smoke_BF16"] = root / "smoke" / "bf16.txt"
    for name, path in paths.items():
        if name != "repo" and not path.is_relative_to(root):
            raise PreflightError("generated path escaped artifact root")
    return paths


def _command_strings(command: Sequence[object]) -> list[str]:
    return [os.fspath(value) if isinstance(value, os.PathLike) else str(value) for value in command]


def build_plan(
    workspace: Path,
    run_commit: str,
    *,
    allow_missing_task6_evidence: bool = False,
) -> dict[str, list[list[str]]]:
    """Return the immutable nine-phase run plan as argv arrays without executing it."""

    workspace = _validate_workspace(workspace)
    run_commit = _validate_run_commit(run_commit)
    paths = _paths(workspace)
    root = paths["root"]
    repo = paths["repo"]
    llama_source = paths["llama_source"]
    build = paths["build"]
    python = paths["python"]
    bin_dir = build / "bin"
    model_config = paths["model_snapshot"] / "config.json"

    bootstrap: list[list[object]] = [
        ["git", "clone", "--no-tags", REPOSITORY_REMOTE, repo],
        ["git", "-C", repo, "checkout", "--detach", run_commit],
        [
            "mkdir",
            "-p",
            root / "source",
            root / "preflight",
            root / "audits",
            root / "calibration",
            root / "evidence",
            root / "models",
            root / "projector",
            root / "imatrix",
            root / "smoke",
            *(root / "splits" / tier.lower() for tier in TIERS),
        ],
        ["python3", "-m", "venv", root / "venv"],
        [python, "-m", "pip", "install", "--no-deps", repo],
        [
            python,
            "-m",
            "mixstq.gguf_run_plan",
            "--action",
            "host-preflight",
            "--host-snapshot",
            paths["host_snapshot"],
        ],
        [python, "-m", "pip", "install", f"{repo}[gpu]"],
        ["git", "clone", "--no-tags", LLAMA_CPP_REMOTE, llama_source],
        ["git", "-C", llama_source, "checkout", "--detach", LLAMA_CPP_COMMIT],
        [
            "cmake",
            "-S",
            llama_source,
            "-B",
            build,
            "-DGGML_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        [
            "cmake",
            "--build",
            build,
            "--config",
            "Release",
            "-j",
            str(CPU_CORES_MIN),
            "--target",
            *REQUIRED_EXECUTABLES,
        ],
        [python, "-c", CONVERTER_RUNNER, llama_source, "--help"],
    ]
    for executable in REQUIRED_EXECUTABLES:
        bootstrap.append(["test", "-x", bin_dir / executable])
        mode = "strict" if executable in LEDGER_VERIFIED_EXECUTABLES else "lenient"
        bootstrap.append([python, "-c", HELP_PROBE_RUNNER, mode, bin_dir / executable])
    bootstrap.extend(
        [
            [paths["hf"], "--help"],
            [
                paths["hf"],
                "download",
                MODEL_ID,
                *MODEL_SNAPSHOT_FILES,
                "--revision",
                MODEL_REVISION,
                "--local-dir",
                paths["model_snapshot"],
            ],
            [
                python,
                "-m",
                "mixstq.gguf_run_plan",
                "--action",
                "model-preflight",
                "--model-config",
                model_config,
            ],
            [
                paths["hf"],
                "download",
                MODEL_ID,
                "--revision",
                MODEL_REVISION,
                "--local-dir",
                paths["model_snapshot"],
            ],
        ]
    )

    calibration = [
        [
            python,
            "-m",
            "mixstq.llama_calibration",
            "--out",
            paths["corpus"],
            "--manifest",
            paths["manifest"],
        ]
    ]
    convert = [
        [
            python,
            "-c",
            CONVERTER_RUNNER,
            llama_source,
            paths["model_snapshot"],
            "--outfile",
            paths["bf16"],
            "--outtype",
            "bf16",
            "--no-mtp",
        ],
        # The preregistration requires the vision projector once, from the same pinned converter,
        # and excludes it from every text bpw denominator and from the benchmarks.
        [
            python,
            "-c",
            CONVERTER_RUNNER,
            llama_source,
            paths["model_snapshot"],
            "--outfile",
            paths["projector"],
            "--outtype",
            "bf16",
            "--mmproj",
        ],
    ]
    imatrix = [
        [
            python,
            "-m",
            "mixstq.gguf_run_plan",
            "--action",
            "token-preflight",
            "--llama-bin",
            bin_dir / "llama-tokenize",
            "--vocab-model",
            paths["bf16"],
            "--corpus",
            paths["corpus"],
            "--manifest",
            paths["manifest"],
        ],
        [
            bin_dir / "llama-imatrix",
            "--model",
            paths["bf16"],
            "--file",
            paths["corpus"],
            "--output-file",
            paths["imatrix"],
            "--ctx-size",
            str(IMATRIX_CONTEXT_TOKENS),
            "--batch-size",
            str(IMATRIX_BATCH_TOKENS),
            "--ubatch-size",
            str(IMATRIX_UBATCH_TOKENS),
            "--chunks",
            str(IMATRIX_CHUNKS),
            "--threads",
            str(IMATRIX_THREADS),
            "--no-ppl",
        ],
    ]
    quantize = [
        [
            bin_dir / "llama-quantize",
            "--imatrix",
            paths["imatrix"],
            paths["bf16"],
            paths[f"model_{tier}"],
            tier,
            str(IMATRIX_THREADS),
        ]
        for tier in TIERS
    ]
    smoke = [
        [
            bin_dir / "llama-cli",
            "--model",
            paths["bf16"] if name == "BF16" else paths[f"model_{name}"],
            "--seed",
            str(SMOKE_SEED),
            "--temperature",
            "0",
            "--n-predict",
            str(SMOKE_PREDICT_TOKENS),
            "--prompt",
            "Reply with one nonempty word.",
            "--no-display-prompt",
            "--single-turn",
            "--output-file",
            paths[f"smoke_{name}"],
            "--log-disable",
        ]
        for name in MODEL_NAMES
    ]
    audit: list[list[object]] = [
        [
            python,
            "-m",
            "mixstq.gguf_run_plan",
            "--action",
            "audit",
            "--workspace",
            workspace,
        ]
    ]
    audit.extend(
        [
            python,
            "-m",
            "mixstq.gguf_audit",
            "--model",
            paths["bf16"] if name == "BF16" else paths[f"model_{name}"],
            "--out",
            root / "audits" / f"{name}.json",
        ]
        for name in MODEL_NAMES
    )
    split = [
        [
            bin_dir / "llama-gguf-split",
            "--split",
            "--split-max-size",
            SPLIT_MAX_DECIMAL_SIZE,
            paths[f"model_{tier}"],
            paths[f"split_{tier}"],
        ]
        for tier in TIERS
    ]
    upload: list[list[object]] = []
    if not allow_missing_task6_evidence:
        # An in-order runner must not be able to publish before Task 6 closed; the operator writes
        # this marker after the paired statistics, PPL and llama-bench outputs are in place.
        upload.append(["test", "-f", root / TASK6_COMPLETE_MARKER])
    upload.append(
        [
            python,
            "-c",
            PUBLIC_REPO_CHECK_RUNNER,
            f"https://huggingface.co/api/datasets/{HF_DATASET_REPO}",
        ]
    )
    task6_mode = "optional" if allow_missing_task6_evidence else "required"
    upload.append(
        [
            python,
            "-c",
            EVIDENCE_ASSEMBLY_RUNNER,
            paths["evidence"],
            *(
                argument
                for name in EVIDENCE_SOURCES
                for argument in ("required", root / name)
            ),
            *(
                argument
                for name in TASK6_EVIDENCE_SOURCES
                for argument in (task6_mode, root / name)
            ),
        ]
    )
    for name, local in (
        *((tier, paths[f"split_dir_{tier}"]) for tier in TIERS),
        ("projector", paths["projector_dir"]),
        ("evidence", paths["evidence"]),
    ):
        remote = f"{HF_UPLOAD_PREFIX}/{name}"
        upload.extend(
            [
                # The host CLI is already logged in by the user; no token reaches argv or output.
                [
                    paths["hf"],
                    "upload",
                    HF_DATASET_REPO,
                    local,
                    remote,
                    "--repo-type",
                    "dataset",
                ],
                [
                    python,
                    "-c",
                    PUBLIC_VERIFY_RUNNER,
                    python,
                    UNAUTHENTICATED_RUNNER,
                    paths["unauthenticated_home"],
                    paths["hf"],
                    HF_DATASET_REPO,
                    remote,
                    local,
                    paths["public_verify"],
                ],
            ]
        )

    plan_objects = {
        "bootstrap": bootstrap,
        "calibration": calibration,
        "convert": convert,
        "imatrix": imatrix,
        "quantize": quantize,
        "smoke": smoke,
        "audit": audit,
        "split": split,
        "upload": upload,
    }
    plan = {
        phase: [_command_strings(command) for command in plan_objects[phase]]
        for phase in PHASES
    }
    serialized = json.dumps(plan, sort_keys=True)
    if _contains_sensitive_text(serialized):
        raise PreflightError("planner output failed credential-safety validation")
    return plan


def _required_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise PreflightError(f"model config field {key} must be an object")
    return value


def _expect_value(mapping: Mapping[str, object], key: str, expected: object) -> None:
    value = mapping.get(key)
    if isinstance(expected, int) and not isinstance(expected, bool):
        matches = type(value) is int and value == expected
    else:
        matches = type(value) is type(expected) and value == expected
    if not matches:
        raise PreflightError(f"model config field {key} does not match the pinned revision")


def validate_model_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the exact nested config of the pinned model revision."""

    if not isinstance(config, Mapping):
        raise PreflightError("model config must be an object")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or architectures != ["Qwen3_5ForConditionalGeneration"]:
        raise PreflightError("model architecture does not match the pinned revision")
    _expect_value(config, "model_type", "qwen3_5")
    _expect_value(config, "language_model_only", False)
    _expect_value(config, "tie_word_embeddings", False)
    text = _required_mapping(config, "text_config")
    expected_text_values = (
        ("model_type", "qwen3_5_text"),
        ("dtype", "bfloat16"),
        ("num_hidden_layers", 64),
        ("hidden_size", 5120),
        ("intermediate_size", 17408),
        ("num_attention_heads", 24),
        ("num_key_value_heads", 4),
        ("head_dim", 256),
        ("vocab_size", 248320),
        ("tie_word_embeddings", False),
        ("full_attention_interval", 4),
        ("mtp_num_hidden_layers", 1),
    )
    for key, expected in expected_text_values:
        _expect_value(text, key, expected)
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != EXPECTED_LAYER_PATTERN:
        raise PreflightError("model layer pattern does not match the pinned revision")
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "architecture": architectures[0],
        "text_model_type": text["model_type"],
        "dtype": text["dtype"],
        "language_layers": text["num_hidden_layers"],
        "hidden_size": text["hidden_size"],
        "intermediate_size": text["intermediate_size"],
        "attention_heads": text["num_attention_heads"],
        "kv_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"],
        "vocab_size": text["vocab_size"],
        "untied_output_head": True,
        "layer_pattern": "3-linear-then-1-full",
        "pattern_repeats": 16,
        "mtp_layers_excluded_by_plan": text["mtp_num_hidden_layers"],
    }


def _finite_number(snapshot: Mapping[str, object], key: str) -> float:
    value = snapshot.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightError(f"host snapshot field {key} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise PreflightError(f"host snapshot field {key} must be finite and nonnegative")
    return normalized


def _nonnegative_integer(snapshot: Mapping[str, object], key: str) -> int:
    value = snapshot.get(key)
    if type(value) is not int or value < 0:
        raise PreflightError(f"host snapshot field {key} must be a nonnegative integer")
    return value


def validate_host_snapshot(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Validate host and offer facts; GB fields use decimal GB and VRAM uses MiB."""

    if not isinstance(snapshot, Mapping):
        raise PreflightError("host snapshot must be an object")
    missing = tuple(key for key in HOST_REQUIRED_KEYS if key not in snapshot)
    if missing:
        raise PreflightError("host snapshot is missing required fields")

    gpu_count = _nonnegative_integer(snapshot, "gpu_count")
    gpu_vendor = snapshot.get("gpu_vendor")
    advertised_vram = _finite_number(snapshot, "advertised_gpu_vram_mib")
    observed_vram = _finite_number(snapshot, "observed_gpu_vram_mib")
    ram_gb = _finite_number(snapshot, "ram_gb")
    cpu_cores = _nonnegative_integer(snapshot, "cpu_cores")
    workspace_free_gb = _finite_number(snapshot, "workspace_free_gb")
    download_mbps = _finite_number(snapshot, "offer_download_mbps")
    reliability = _finite_number(snapshot, "offer_reliability")
    hourly_usd = _finite_number(snapshot, "offer_hourly_usd")

    pids = snapshot.get("active_compute_pids")
    if not isinstance(pids, (list, tuple)):
        raise PreflightError("host snapshot active_compute_pids must be an array")
    if any(type(pid) is not int or pid <= 0 for pid in pids):
        raise PreflightError("host snapshot active_compute_pids contains an invalid PID")
    if pids:
        raise PreflightError("host snapshot reports an existing GPU compute process")
    if gpu_count != GPU_COUNT_REQUIRED:
        raise PreflightError("host must expose exactly one NVIDIA GPU")
    if gpu_vendor != "NVIDIA":
        raise PreflightError("host GPU vendor must be NVIDIA")
    if advertised_vram < GPU_VRAM_MIN_MIB or observed_vram < GPU_VRAM_MIN_MIB:
        raise PreflightError("host GPU VRAM is below the MiB floor")
    if ram_gb < RAM_MIN_DECIMAL_GB:
        raise PreflightError("host RAM is below the decimal-GB floor")
    if cpu_cores < CPU_CORES_MIN:
        raise PreflightError("host CPU count is below the floor")
    if workspace_free_gb < WORKSPACE_MIN_DECIMAL_GB:
        raise PreflightError("host workspace disk is below the decimal-GB floor")
    if download_mbps < OFFER_DOWNLOAD_MIN_MBPS:
        raise PreflightError("offer download speed is below the floor")
    if not OFFER_RELIABILITY_MIN <= reliability <= 1:
        raise PreflightError("offer reliability is outside the accepted range")
    if not 0 < hourly_usd <= OFFER_HOURLY_MAX_USD:
        raise PreflightError("offer hourly price is outside the accepted range")

    return {
        "gpu_count": gpu_count,
        "gpu_vendor": gpu_vendor,
        "advertised_gpu_vram_mib": advertised_vram,
        "observed_gpu_vram_mib": observed_vram,
        "ram_decimal_gb": ram_gb,
        "cpu_cores": cpu_cores,
        "workspace_free_decimal_gb": workspace_free_gb,
        "active_compute_pids": [],
        "offer_download_mbps": download_mbps,
        "offer_reliability": reliability,
        "offer_hourly_usd": hourly_usd,
    }


def _validate_canonical_manifest(manifest: Mapping[str, Any]) -> None:
    selection = manifest.get("selection")
    records = manifest.get("records")
    capacity = manifest.get("imatrix_capacity")
    if not isinstance(selection, Mapping) or not isinstance(records, list):
        raise PreflightError("calibration manifest is not canonical")
    if selection.get("domain_order") != list(llama_calibration.DOMAIN_ORDER):
        raise PreflightError("calibration manifest has a noncanonical domain order")
    if selection.get("per_domain") != llama_calibration.CANONICAL_PER_DOMAIN:
        raise PreflightError("calibration manifest has a noncanonical record count")
    if selection.get("min_chars") != llama_calibration.CANONICAL_MIN_CHARS:
        raise PreflightError("calibration manifest has a noncanonical selection floor")
    if len(records) != llama_calibration.CANONICAL_PER_DOMAIN * len(llama_calibration.DOMAIN_ORDER):
        raise PreflightError("calibration manifest does not contain exactly 96 records")
    if not isinstance(capacity, Mapping):
        raise PreflightError("calibration manifest lacks imatrix capacity metadata")
    expected_capacity = (
        ("chunks", IMATRIX_CHUNKS),
        ("tokens_per_chunk", IMATRIX_CONTEXT_TOKENS),
        ("total_token_capacity", IMATRIX_TOKEN_CAPACITY),
    )
    if any(capacity.get(key) != value for key, value in expected_capacity):
        raise PreflightError("calibration manifest imatrix capacity is not canonical")


def exact_tokenizer_preflight(
    tokenizer: Path,
    model: Path,
    corpus: Path,
    manifest: Path,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    """Run pinned llama-tokenize and reject ambiguous, over-capacity, or mutable input."""

    tokenizer = Path(tokenizer)
    model = Path(model)
    corpus = Path(corpus)
    manifest = Path(manifest)
    if not tokenizer.is_absolute() or not tokenizer.is_file() or not os.access(tokenizer, os.X_OK):
        raise PreflightError("tokenizer must be an absolute executable file")
    if not model.is_absolute() or not model.is_file():
        raise PreflightError("tokenizer model must be an absolute file")
    if not corpus.is_absolute() or not manifest.is_absolute():
        raise PreflightError("calibration paths must be absolute")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise PreflightError("tokenizer timeout must be numeric")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise PreflightError("tokenizer timeout must be finite and positive")

    try:
        committed = llama_calibration.require_committed_corpus(corpus, manifest)
    except (OSError, RuntimeError, ValueError) as error:
        raise PreflightError("calibration publication is not valid and committed") from error
    _validate_canonical_manifest(committed)
    marker = llama_calibration.commit_marker_path(manifest)
    before = (corpus.read_bytes(), manifest.read_bytes(), marker.read_bytes())
    argv = [
        os.fspath(tokenizer),
        "--model",
        os.fspath(model),
        "--file",
        os.fspath(corpus),
        "--ids",
        "--show-count",
        "--log-disable",
    ]
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout,
            check=False,
            shell=False,
            env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired as error:
        raise PreflightError("tokenizer preflight timed out") from error
    except (OSError, UnicodeError) as error:
        raise PreflightError("tokenizer preflight could not execute safely") from error
    after = (corpus.read_bytes(), manifest.read_bytes(), marker.read_bytes())
    if after != before:
        raise PreflightError("tokenizer preflight detected calibration publication mutation")
    if completed.returncode != 0:
        raise PreflightError("tokenizer preflight exited unsuccessfully")
    matches = [
        TOKEN_COUNT_PATTERN.fullmatch(line)
        for line in completed.stdout.splitlines()
        if line.startswith("Total number of tokens:")
    ]
    if len(matches) != 1 or matches[0] is None:
        raise PreflightError("tokenizer output did not contain one exact token count")
    token_count = int(matches[0].group(1))
    if not 0 < token_count <= IMATRIX_TOKEN_CAPACITY:
        raise PreflightError("exact tokenizer count is outside imatrix capacity")
    corpus_info = committed.get("corpus")
    if not isinstance(corpus_info, Mapping) or not isinstance(corpus_info.get("sha256"), str):
        raise PreflightError("calibration corpus identity is invalid")
    return {
        "token_count": token_count,
        "capacity_tokens": IMATRIX_TOKEN_CAPACITY,
        "chunks": IMATRIX_CHUNKS,
        "tokens_per_chunk": IMATRIX_CONTEXT_TOKENS,
        "corpus_sha256": corpus_info["sha256"],
        "committed": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_artifacts(workspace: Path) -> dict[str, object]:
    paths = _paths(_validate_workspace(workspace))
    inventory: dict[str, object] = {}
    for name in MODEL_NAMES:
        model_path = paths["bf16"] if name == "BF16" else paths[f"model_{name}"]
        smoke_path = paths[f"smoke_{name}"]
        if not model_path.is_file() or model_path.stat().st_size <= 0:
            raise PreflightError("audit found a missing or empty model artifact")
        if not smoke_path.is_file() or not smoke_path.read_text(encoding="utf-8").strip():
            raise PreflightError("audit found a missing or empty smoke completion")
        inventory[name] = {
            "model_bytes": model_path.stat().st_size,
            "model_sha256": _sha256_file(model_path),
            "smoke_sha256": _sha256_file(smoke_path),
        }
    if not paths["imatrix"].is_file() or paths["imatrix"].stat().st_size <= 0:
        raise PreflightError("audit found a missing or empty imatrix artifact")
    projector = paths["projector"]
    if not projector.is_file() or projector.stat().st_size <= 0:
        raise PreflightError("audit found a missing or empty vision projector artifact")
    return {
        "models": inventory,
        "imatrix_sha256": _sha256_file(paths["imatrix"]),
        # The preregistration keeps the projector out of both bpw denominators and the
        # benchmarks, so it is recorded beside the text arms and never inside them.
        "projector": {
            "bytes": projector.stat().st_size,
            "sha256": _sha256_file(projector),
            "excluded_from_text_bpw": True,
        },
        "smoke_contract": "five independent nonempty completions",
    }


def _load_json_object(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"{description} is not readable UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise PreflightError(f"{description} must be a JSON object")
    return value


def _format_shell(plan: Mapping[str, Sequence[Sequence[str]]]) -> str:
    lines = ["set -eu"]
    for phase in PHASES:
        lines.extend(("", f"# {phase}"))
        lines.extend(shlex.join(command) for command in plan[phase])
    return "\n".join(lines) + "\n"


def _parser() -> SafeArgumentParser:
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("plan", "model-preflight", "host-preflight", "token-preflight", "audit"),
        default="plan",
    )
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-commit")
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--host-snapshot", type=Path)
    parser.add_argument("--llama-bin", type=Path)
    parser.add_argument("--vocab-model", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--allow-missing-task6-evidence",
        action="store_true",
        help="publish without the Task 6 completion marker and with eval/ppl/bench optional",
    )
    return parser


def _require_args(parser: SafeArgumentParser, args: argparse.Namespace, names: Sequence[str]) -> None:
    if any(getattr(args, name) is None for name in names):
        parser.error("missing action arguments")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.allow_missing_task6_evidence and args.action != "plan":
        # The escape hatch only shapes the emitted upload phase; accepting it elsewhere would
        # imply the preflights and the audit honour it, and they do not.
        parser.exit(2, "error: --allow-missing-task6-evidence applies only to --action plan\n")
    try:
        if args.action == "plan":
            _require_args(parser, args, ("workspace", "run_commit"))
            result = build_plan(
                args.workspace,
                args.run_commit,
                allow_missing_task6_evidence=args.allow_missing_task6_evidence,
            )
            if args.format == "shell":
                sys.stdout.write(_format_shell(result))
            else:
                print(json.dumps(result, indent=2))
            return 0
        if args.action == "model-preflight":
            _require_args(parser, args, ("model_config",))
            result = validate_model_config(_load_json_object(args.model_config, "model config"))
        elif args.action == "host-preflight":
            _require_args(parser, args, ("host_snapshot",))
            result = validate_host_snapshot(_load_json_object(args.host_snapshot, "host snapshot"))
        elif args.action == "token-preflight":
            _require_args(parser, args, ("llama_bin", "vocab_model", "corpus", "manifest"))
            result = exact_tokenizer_preflight(
                args.llama_bin,
                args.vocab_model,
                args.corpus,
                args.manifest,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            _require_args(parser, args, ("workspace",))
            result = _audit_artifacts(args.workspace)
    except PreflightError as error:
        message = str(error)
        if _contains_sensitive_text(message):
            message = "preflight rejected unsafe input"
        parser.exit(2, f"error: {message}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
