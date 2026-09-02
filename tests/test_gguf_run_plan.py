from __future__ import annotations

import copy
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import gguf_run_plan as run_plan
import llama_calibration as calibration
import pytest

RUN_COMMIT = "0fa807ad85ab5565c23c78fe9d2a6a60570d782a"
FORBIDDEN = (
    "hf_",
    "HF_TOKEN",
    "MIXSTQ_VAST_KEY",
    "--token",
    "Authorization",
    ".superpowers/sdd",
)
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~-]{20,}"),
)


def _model_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": False,
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "text_config": {
            "dtype": "bfloat16",
            "full_attention_interval": 4,
            "head_dim": 256,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "layer_types": list(run_plan.EXPECTED_LAYER_PATTERN),
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 24,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "tie_word_embeddings": False,
            "vocab_size": 248320,
        },
    }


def _host_snapshot() -> dict[str, object]:
    return {
        "gpu_count": 1,
        "gpu_vendor": "NVIDIA",
        "advertised_gpu_vram_mib": 80_000,
        "observed_gpu_vram_mib": 80_000,
        "ram_gb": 96,
        "cpu_cores": 16,
        "workspace_free_gb": 300,
        "active_compute_pids": [],
        "offer_download_mbps": 500,
        "offer_reliability": 0.98,
        "offer_hourly_usd": 1.20,
    }


def _loader(dataset_id: str, config: str | None = None, **kwargs: object):
    del config, kwargs
    spec = next(spec for spec in calibration.DATASETS if spec.dataset_id == dataset_id)
    return [
        {spec.field: f"{spec.domain}-{index:02d} " + (chr(97 + index % 26) * 220)}
        for index in range(calibration.CANONICAL_PER_DOMAIN)
    ]


def _committed_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    corpus = tmp_path / "corpus.txt"
    manifest = tmp_path / "manifest.json"
    calibration._build_corpus_with_loader(
        corpus,
        manifest,
        per_domain=calibration.CANONICAL_PER_DOMAIN,
        min_chars=calibration.CANONICAL_MIN_CHARS,
        load_dataset_fn=_loader,
    )
    return corpus, manifest, calibration.commit_marker_path(manifest)


def _fake_executable(tmp_path: Path, body: str) -> Path:
    executable = tmp_path / "fake-llama"
    executable.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def _vocab_model(tmp_path: Path) -> Path:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"vocab-only-test-double")
    return model


def _assert_safe(text: str) -> None:
    assert all(marker not in text for marker in FORBIDDEN)
    assert all(pattern.search(text) is None for pattern in SECRET_PATTERNS)


def test_build_plan_exact_phases_pins_paths_and_no_side_effects(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    plan = run_plan.build_plan(workspace, RUN_COMMIT)

    assert tuple(plan) == run_plan.PHASES
    assert [len(plan[phase]) for phase in plan] == [32, 1, 2, 2, 4, 5, 1, 4]
    assert not workspace.exists()
    serialized = json.dumps(plan)
    _assert_safe(serialized)
    assert RUN_COMMIT in serialized
    assert run_plan.MODEL_REVISION in serialized
    assert run_plan.LLAMA_CPP_COMMIT in serialized
    assert run_plan.REPOSITORY_REMOTE in serialized
    assert run_plan.LLAMA_CPP_REMOTE in serialized
    assert run_plan.MODEL_ID in serialized

    artifact_root = workspace / "mix-stq" / "artifacts" / "qwen38-gguf-v27"
    for commands in plan.values():
        for command in commands:
            assert isinstance(command, list)
            assert all(isinstance(argument, str) for argument in command)
            for argument in command:
                if str(artifact_root.parent) in argument and argument != str(workspace / "mix-stq"):
                    assert argument.startswith(str(artifact_root)) or argument.endswith("[gpu]")


def test_plan_conversion_imatrix_quant_smoke_and_split_contracts(tmp_path: Path) -> None:
    plan = run_plan.build_plan(tmp_path / "workspace", RUN_COMMIT)
    artifact_root = tmp_path / "workspace" / "mix-stq" / "artifacts" / "qwen38-gguf-v27"
    bf16 = str(artifact_root / "models" / "qwen38-27b-bf16.gguf")
    imatrix_path = str(artifact_root / "imatrix" / "qwen38-27b.imatrix.gguf")
    corpus = str(artifact_root / "calibration" / "corpus.txt")

    conversion = plan["convert"][0]
    assert conversion[-6:] == [
        str(artifact_root / "model-snapshot"),
        "--outfile",
        bf16,
        "--outtype",
        "bf16",
        "--no-mtp",
    ]
    assert "f16" not in conversion

    preflight, imatrix = plan["imatrix"]
    assert preflight[preflight.index("--vocab-model") + 1] == bf16
    assert preflight[preflight.index("--corpus") + 1] == corpus
    expected_flags = {
        "--model": bf16,
        "--file": corpus,
        "--output-file": imatrix_path,
        "--ctx-size": "512",
        "--batch-size": "512",
        "--ubatch-size": "128",
        "--chunks": "128",
        "--threads": "32",
    }
    for flag, value in expected_flags.items():
        assert imatrix[imatrix.index(flag) + 1] == value
    assert "--no-ppl" in imatrix

    quantize = plan["quantize"]
    assert [command[-2] for command in quantize] == list(run_plan.TIERS)
    for command, tier in zip(quantize, run_plan.TIERS, strict=True):
        assert command[command.index("--imatrix") + 1] == imatrix_path
        assert command[-5] == imatrix_path
        assert command[-4] == bf16
        assert command[-3].endswith(f"qwen38-27b-{tier.lower()}.gguf")
        assert "--allow-requantize" not in command
    quant_outputs = {command[-3] for command in quantize}
    assert all(not any(output in command[:-3] for output in quant_outputs) for command in quantize)

    assert len(plan["smoke"]) == 5
    smoke_models = [command[command.index("--model") + 1] for command in plan["smoke"]]
    assert smoke_models == [bf16, *(command[-3] for command in quantize)]
    for command in plan["smoke"]:
        assert command[command.index("--seed") + 1] == "22"
        assert command[command.index("--temperature") + 1] == "0"
        assert command[command.index("--n-predict") + 1] == "16"
        assert "--no-display-prompt" in command
        assert "--single-turn" in command
        assert "--no-conversation" not in command
        assert "--output-file" in command

    assert [command[-2] for command in plan["split"]] == [
        str(artifact_root / "models" / f"qwen38-27b-{tier.lower()}.gguf")
        for tier in run_plan.TIERS
    ]
    assert all(command[command.index("--split-max-size") + 1] == "8G" for command in plan["split"])


def test_convert_emits_the_pinned_vision_projector_after_the_text_model(tmp_path: Path) -> None:
    plan = run_plan.build_plan(tmp_path / "workspace", RUN_COMMIT)
    artifact_root = tmp_path / "workspace" / "mix-stq" / "artifacts" / "qwen38-gguf-v27"
    snapshot = str(artifact_root / "model-snapshot")
    projector = str(artifact_root / "projector" / "qwen38-27b-mmproj-bf16.gguf")

    text, vision = plan["convert"]
    assert text[-1] == "--no-mtp"
    assert "--mmproj" not in text
    assert vision[:2] == text[:2]
    assert vision[2] == run_plan.CONVERTER_RUNNER
    assert vision[3] == text[3]
    assert vision[-6:] == [
        snapshot,
        "--outfile",
        projector,
        "--outtype",
        "bf16",
        "--mmproj",
    ]
    assert "--no-mtp" not in vision

    mkdir = next(command for command in plan["bootstrap"] if command[:2] == ["mkdir", "-p"])
    assert str(artifact_root / "projector") in mkdir
    for phase in ("imatrix", "quantize", "smoke", "audit", "split"):
        assert all(projector not in command for command in plan[phase])


def test_bootstrap_downloads_every_pinned_snapshot_file_without_multi_value_include(
    tmp_path: Path,
) -> None:
    plan = run_plan.build_plan(tmp_path / "workspace", RUN_COMMIT)
    snapshot = str(
        tmp_path / "workspace" / "mix-stq" / "artifacts" / "qwen38-gguf-v27" / "model-snapshot"
    )
    downloads = [
        command
        for command in plan["bootstrap"]
        if command[0].endswith("/hf") and "download" in command
    ]
    assert len(downloads) == 2

    partial, full = downloads
    assert partial[1:3] == ["download", run_plan.MODEL_ID]
    files = partial[3 : 3 + len(run_plan.MODEL_SNAPSHOT_FILES)]
    assert files == list(run_plan.MODEL_SNAPSHOT_FILES)
    assert "config.json" in files
    assert partial[partial.index("--revision") + 1] == run_plan.MODEL_REVISION
    assert partial[partial.index("--local-dir") + 1] == snapshot
    assert "--include" not in partial
    assert "--include" not in full

    preflight_index = next(
        index
        for index, command in enumerate(plan["bootstrap"])
        if "model-preflight" in command
    )
    assert plan["bootstrap"].index(partial) < preflight_index < plan["bootstrap"].index(full)


def test_bootstrap_builds_and_probes_every_required_executable(tmp_path: Path) -> None:
    plan = run_plan.build_plan(tmp_path / "workspace", RUN_COMMIT)
    build = str(
        tmp_path
        / "workspace"
        / "mix-stq"
        / "artifacts"
        / "qwen38-gguf-v27"
        / "source"
        / "llama.cpp"
        / "build"
    )
    assert set(run_plan.REQUIRED_EXECUTABLES) >= {
        "llama-imatrix",
        "llama-quantize",
        "llama-cli",
        "llama-server",
        "llama-perplexity",
        "llama-bench",
        "llama-tokenize",
        "llama-gguf-split",
    }

    targets = next(command for command in plan["bootstrap"] if command[:2] == ["cmake", "--build"])
    assert targets[targets.index("--target") + 1 :] == list(run_plan.REQUIRED_EXECUTABLES)

    for executable in run_plan.REQUIRED_EXECUTABLES:
        binary = f"{build}/bin/{executable}"
        assert ["test", "-x", binary] in plan["bootstrap"]
        probe = [
            command
            for command in plan["bootstrap"]
            if command[-1] == binary and command[0] != "test"
        ]
        assert len(probe) == 1
        assert probe[0][1:3] == ["-c", run_plan.HELP_PROBE_RUNNER]
        assert [binary, "--help"] not in plan["bootstrap"]


def test_help_probe_runner_accepts_a_pinned_usage_exit_code_of_one(tmp_path: Path) -> None:
    quantize_like = _fake_executable(
        tmp_path,
        "import sys\nprint('usage: llama-quantize [options] model-f32.gguf')\nraise SystemExit(1)",
    )
    completed = subprocess.run(
        [sys.executable, "-c", run_plan.HELP_PROBE_RUNNER, str(quantize_like)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0

    silent = _fake_executable(tmp_path, "raise SystemExit(1)")
    rejected = subprocess.run(
        [sys.executable, "-c", run_plan.HELP_PROBE_RUNNER, str(silent)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert rejected.returncode != 0


@pytest.mark.parametrize(
    "workspace,run_commit",
    [
        (Path("relative"), RUN_COMMIT),
        (Path("/workspace/../escape"), RUN_COMMIT),
        (Path("/"), RUN_COMMIT),
        (Path("/workspace"), "A" * 40),
        (Path("/workspace"), "0" * 39),
        (Path("/workspace"), "0" * 41),
        (Path("/workspace"), "g" * 40),
    ],
)
def test_build_plan_rejects_invalid_workspace_or_commit_without_writes(
    tmp_path: Path, workspace: Path, run_commit: str
) -> None:
    before = tuple(tmp_path.iterdir())
    with pytest.raises(run_plan.PreflightError):
        run_plan.build_plan(workspace, run_commit)
    assert tuple(tmp_path.iterdir()) == before


def test_validate_model_config_success_inventory() -> None:
    result = run_plan.validate_model_config(_model_config())
    assert result == {
        "model_id": run_plan.MODEL_ID,
        "model_revision": run_plan.MODEL_REVISION,
        "architecture": "Qwen3_5ForConditionalGeneration",
        "text_model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "language_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "attention_heads": 24,
        "kv_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "untied_output_head": True,
        "layer_pattern": "3-linear-then-1-full",
        "pattern_repeats": 16,
        "mtp_layers_excluded_by_plan": 1,
    }


@pytest.mark.parametrize(
    "path,bad_value",
    [
        (("architectures",), ["Qwen3_5ForCausalLM"]),
        (("model_type",), "qwen3"),
        (("language_model_only",), True),
        (("tie_word_embeddings",), True),
        (("text_config", "model_type"), "qwen3_5"),
        (("text_config", "dtype"), "float16"),
        (("text_config", "num_hidden_layers"), 63),
        (("text_config", "hidden_size"), 4096),
        (("text_config", "intermediate_size"), 17407),
        (("text_config", "num_attention_heads"), 25),
        (("text_config", "num_key_value_heads"), 8),
        (("text_config", "head_dim"), 128),
        (("text_config", "vocab_size"), 248319),
        (("text_config", "tie_word_embeddings"), True),
        (("text_config", "full_attention_interval"), 3),
        (("text_config", "mtp_num_hidden_layers"), 0),
        (("text_config", "layer_types"), ["full_attention"] * 64),
    ],
)
def test_validate_model_config_fails_every_material_mismatch(
    path: tuple[str, ...], bad_value: object
) -> None:
    config = copy.deepcopy(_model_config())
    target = config
    for key in path[:-1]:
        target = target[key]
        assert isinstance(target, dict)
    target[path[-1]] = bad_value
    with pytest.raises(run_plan.PreflightError):
        run_plan.validate_model_config(config)


@pytest.mark.parametrize("bad_config", [{}, {"text_config": []}, []])
def test_validate_model_config_rejects_missing_or_wrong_shape(bad_config: object) -> None:
    with pytest.raises(run_plan.PreflightError):
        run_plan.validate_model_config(bad_config)


def test_validate_host_snapshot_accepts_exact_boundary_and_documents_units() -> None:
    result = run_plan.validate_host_snapshot(_host_snapshot())
    assert result["advertised_gpu_vram_mib"] == 80_000
    assert result["observed_gpu_vram_mib"] == 80_000
    assert result["ram_decimal_gb"] == 96
    assert result["workspace_free_decimal_gb"] == 300
    assert result["active_compute_pids"] == []


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("gpu_count", 0),
        ("gpu_count", 2),
        ("gpu_vendor", "AMD"),
        ("advertised_gpu_vram_mib", 79_999),
        ("observed_gpu_vram_mib", 79_999),
        ("ram_gb", 95.999),
        ("cpu_cores", 15),
        ("workspace_free_gb", 299.999),
        ("active_compute_pids", [42]),
        ("offer_download_mbps", 499.999),
        ("offer_reliability", 0.979999),
        ("offer_reliability", 1.000001),
        ("offer_hourly_usd", 0),
        ("offer_hourly_usd", 1.200001),
    ],
)
def test_validate_host_snapshot_rejects_floor_policy_failures(key: str, bad_value: object) -> None:
    snapshot = _host_snapshot()
    snapshot[key] = bad_value
    with pytest.raises(run_plan.PreflightError):
        run_plan.validate_host_snapshot(snapshot)


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("gpu_count", True),
        ("cpu_cores", 16.0),
        ("observed_gpu_vram_mib", False),
        ("ram_gb", -1),
        ("workspace_free_gb", math.nan),
        ("offer_download_mbps", math.inf),
        ("offer_reliability", -math.inf),
        ("offer_hourly_usd", -0.1),
        ("active_compute_pids", [-1]),
        ("active_compute_pids", [True]),
        ("active_compute_pids", ""),
    ],
)
def test_validate_host_snapshot_rejects_invalid_numeric_and_pid_shapes(
    key: str, bad_value: object
) -> None:
    snapshot = _host_snapshot()
    snapshot[key] = bad_value
    with pytest.raises(run_plan.PreflightError):
        run_plan.validate_host_snapshot(snapshot)


def test_validate_host_snapshot_rejects_missing_field() -> None:
    snapshot = _host_snapshot()
    del snapshot["ram_gb"]
    with pytest.raises(run_plan.PreflightError):
        run_plan.validate_host_snapshot(snapshot)


@pytest.mark.parametrize("count", [65_536, 1])
def test_exact_tokenizer_preflight_accepts_valid_exact_counts_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    corpus, manifest, marker = _committed_pair(tmp_path)
    before = (corpus.read_bytes(), manifest.read_bytes(), marker.read_bytes())
    fake = _fake_executable(
        tmp_path,
        f"import sys\nassert isinstance(sys.argv, list)\nprint('[]')\nprint('Total number of tokens: {count}')",
    )
    model = _vocab_model(tmp_path)
    original_run = subprocess.run
    calls: list[tuple[object, object]] = []

    def guarded_run(*args: object, **kwargs: object):
        calls.append((args, kwargs))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(run_plan.subprocess, "run", guarded_run)
    result = run_plan.exact_tokenizer_preflight(fake, model, corpus, manifest)

    assert result["token_count"] == count
    assert result["capacity_tokens"] == 65_536
    assert result["committed"] is True
    assert before == (corpus.read_bytes(), manifest.read_bytes(), marker.read_bytes())
    assert len(calls) == 1
    argv = calls[0][0][0]
    kwargs = calls[0][1]
    assert isinstance(argv, list)
    assert kwargs["shell"] is False
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["env"] == {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}


def test_exact_tokenizer_preflight_rejects_65537(tmp_path: Path) -> None:
    corpus, manifest, _ = _committed_pair(tmp_path)
    fake = _fake_executable(tmp_path, "print('Total number of tokens: 65537')")
    with pytest.raises(run_plan.PreflightError, match="outside imatrix capacity"):
        run_plan.exact_tokenizer_preflight(fake, _vocab_model(tmp_path), corpus, manifest)


@pytest.mark.parametrize(
    "body",
    [
        "print('no count')",
        "print('Total number of tokens: nope')",
        "print('Total number of tokens: 1')\nprint('Total number of tokens: 1')",
        "print('Total number of tokens: 0')",
    ],
)
def test_exact_tokenizer_preflight_rejects_malformed_or_ambiguous_output(
    tmp_path: Path, body: str
) -> None:
    corpus, manifest, _ = _committed_pair(tmp_path)
    fake = _fake_executable(tmp_path, body)
    with pytest.raises(run_plan.PreflightError):
        run_plan.exact_tokenizer_preflight(fake, _vocab_model(tmp_path), corpus, manifest)


def test_exact_tokenizer_preflight_rejects_nonzero_and_sanitizes_output(tmp_path: Path) -> None:
    corpus, manifest, _ = _committed_pair(tmp_path)
    fake = _fake_executable(
        tmp_path,
        "import sys\nprint('credential-like-private-data', file=sys.stderr)\nraise SystemExit(7)",
    )
    with pytest.raises(run_plan.PreflightError) as caught:
        run_plan.exact_tokenizer_preflight(fake, _vocab_model(tmp_path), corpus, manifest)
    assert "credential-like-private-data" not in str(caught.value)


def test_exact_tokenizer_preflight_rejects_timeout(tmp_path: Path) -> None:
    corpus, manifest, _ = _committed_pair(tmp_path)
    fake = _fake_executable(
        tmp_path,
        "import time\ntime.sleep(2)\nprint('Total number of tokens: 1')",
    )
    with pytest.raises(run_plan.PreflightError, match="timed out"):
        run_plan.exact_tokenizer_preflight(
            fake,
            _vocab_model(tmp_path),
            corpus,
            manifest,
            timeout_seconds=0.05,
        )


def test_exact_tokenizer_preflight_rejects_invalid_publication_before_exec(tmp_path: Path) -> None:
    corpus, manifest, marker = _committed_pair(tmp_path)
    marker.write_text("{}\n", encoding="utf-8")
    sentinel = tmp_path / "executed"
    fake = _fake_executable(tmp_path, f"from pathlib import Path\nPath({str(sentinel)!r}).touch()")
    with pytest.raises(run_plan.PreflightError, match="not valid and committed"):
        run_plan.exact_tokenizer_preflight(fake, _vocab_model(tmp_path), corpus, manifest)
    assert not sentinel.exists()


def test_exact_tokenizer_preflight_detects_publication_mutation(tmp_path: Path) -> None:
    corpus, manifest, _ = _committed_pair(tmp_path)
    fake = _fake_executable(
        tmp_path,
        "import pathlib,sys\npathlib.Path(sys.argv[sys.argv.index('--file') + 1]).write_text('changed')\nprint('Total number of tokens: 1')",
    )
    with pytest.raises(run_plan.PreflightError, match="mutation"):
        run_plan.exact_tokenizer_preflight(fake, _vocab_model(tmp_path), corpus, manifest)


def _run_cli(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "PYTHONPATH": str(repo / "src")}
    return subprocess.run(
        [sys.executable, "-m", "mixstq.gguf_run_plan", *arguments],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def test_cli_help_json_shell_and_invalid_inputs_have_no_side_effects(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"

    help_result = _run_cli(repo, "--help")
    assert help_result.returncode == 0
    assert "--workspace" in help_result.stdout
    assert "--run-commit" in help_result.stdout

    json_result = _run_cli(
        repo, "--workspace", str(workspace), "--run-commit", RUN_COMMIT, "--format", "json"
    )
    assert json_result.returncode == 0
    assert tuple(json.loads(json_result.stdout)) == run_plan.PHASES

    shell_result = _run_cli(
        repo, "--workspace", str(workspace), "--run-commit", RUN_COMMIT, "--format", "shell"
    )
    assert shell_result.returncode == 0
    assert shell_result.stdout.startswith("set -eu\n")
    assert "# bootstrap" in shell_result.stdout
    assert "# split" in shell_result.stdout

    invalid_sha = _run_cli(repo, "--workspace", str(workspace), "--run-commit", "BAD")
    assert invalid_sha.returncode == 2
    assert "40 lowercase hexadecimal" in invalid_sha.stderr
    invalid_workspace = _run_cli(repo, "--workspace", "relative", "--run-commit", RUN_COMMIT)
    assert invalid_workspace.returncode == 2
    assert "absolute normalized" in invalid_workspace.stderr
    unsafe_input = _run_cli(
        repo,
        "--workspace",
        str(tmp_path / "HF_TOKEN"),
        "--run-commit",
        "hf_credential-like-input",
    )
    assert unsafe_input.returncode == 2

    combined = "".join(
        result.stdout + result.stderr
        for result in (
            help_result,
            json_result,
            shell_result,
            invalid_sha,
            invalid_workspace,
            unsafe_input,
        )
    )
    _assert_safe(combined)
    assert not workspace.exists()
    assert not (tmp_path / "HF_TOKEN").exists()
