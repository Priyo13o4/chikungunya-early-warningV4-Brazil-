from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import logging
import os
import platform
import shutil
import subprocess
from typing import Any

LOGGER = logging.getLogger(__name__)

_VALID_MODES: set[str] = {"auto", "cpu", "macos_metal", "nvidia_cuda"}
_VALID_PREFER_GPU_FOR: set[str] = {"baselines", "bayesian", "both"}


@dataclass(frozen=True)
class ComputeBackendConfig:
    mode: str = "cpu"
    prefer_gpu_for: str = "both"
    fallback: str = "cpu"
    configured: bool = False


@dataclass(frozen=True)
class ComputeBackendCapabilities:
    macos_metal: bool
    nvidia_cuda: bool
    details: dict[str, Any] = field(default_factory=dict)


def parse_compute_backend_config(raw_model_config: dict[str, Any] | None) -> ComputeBackendConfig:
    if not isinstance(raw_model_config, dict):
        return ComputeBackendConfig()

    raw_backend = raw_model_config.get("compute_backend")
    if raw_backend is None:
        return ComputeBackendConfig()
    if not isinstance(raw_backend, dict):
        LOGGER.warning("Invalid compute_backend config (expected mapping); using compatibility CPU defaults")
        return ComputeBackendConfig()

    raw_mode = str(raw_backend.get("mode", "auto")).strip().lower()
    mode = raw_mode if raw_mode in _VALID_MODES else "auto"
    if mode != raw_mode:
        LOGGER.warning("Invalid compute_backend.mode '%s'; falling back to 'auto'", raw_mode)

    raw_prefer = str(raw_backend.get("prefer_gpu_for", "both")).strip().lower()
    prefer_gpu_for = raw_prefer if raw_prefer in _VALID_PREFER_GPU_FOR else "both"
    if prefer_gpu_for != raw_prefer:
        LOGGER.warning("Invalid compute_backend.prefer_gpu_for '%s'; falling back to 'both'", raw_prefer)

    raw_fallback = str(raw_backend.get("fallback", "cpu")).strip().lower()
    fallback = "cpu"
    if raw_fallback != "cpu":
        LOGGER.warning("Unsupported compute_backend.fallback '%s'; forcing 'cpu'", raw_fallback)

    return ComputeBackendConfig(
        mode=mode,
        prefer_gpu_for=prefer_gpu_for,
        fallback=fallback,
        configured=True,
    )


def detect_compute_backend_capabilities() -> ComputeBackendCapabilities:
    metal_available, metal_detail = _detect_macos_metal_capability()
    cuda_available, cuda_detail = _detect_nvidia_cuda_capability()
    return ComputeBackendCapabilities(
        macos_metal=bool(metal_available),
        nvidia_cuda=bool(cuda_available),
        details={
            "platform": platform.system(),
            "macos_metal": metal_detail,
            "nvidia_cuda": cuda_detail,
        },
    )


def resolve_backends(requested: ComputeBackendConfig) -> dict[str, Any]:
    capabilities = detect_compute_backend_capabilities()
    return _resolve_backends_with_capabilities(requested=requested, capabilities=capabilities)


def resolve_effective_backends(
    config: ComputeBackendConfig,
    capabilities: ComputeBackendCapabilities,
) -> dict[str, Any]:
    return _resolve_backends_with_capabilities(requested=config, capabilities=capabilities)


def _resolve_backends_with_capabilities(
    *,
    requested: ComputeBackendConfig,
    capabilities: ComputeBackendCapabilities,
) -> dict[str, Any]:
    baseline = _resolve_backend_for_track(
        track="baseline",
        requested=requested,
        capabilities=capabilities,
    )
    bayesian = _resolve_backend_for_track(
        track="bayesian",
        requested=requested,
        capabilities=capabilities,
    )

    reason_logs: list[str] = []
    reason_logs.extend(baseline["reason_logs"])
    reason_logs.extend(bayesian["reason_logs"])

    for message in reason_logs:
        LOGGER.warning("%s", message)

    baseline_backend = str(baseline.get("effective_backend", "cpu"))
    bayesian_backend = str(bayesian.get("effective_backend", "cpu"))
    fallback_used = bool(baseline.get("fallback_used", False) or bayesian.get("fallback_used", False))

    return {
        "configured": bool(requested.configured),
        "requested": asdict(requested),
        "effective": {
            "baseline_backend": baseline_backend,
            "bayesian_backend": bayesian_backend,
        },
        "baseline_backend": baseline_backend,
        "bayesian_backend": bayesian_backend,
        "fallback": str(requested.fallback),
        "fallback_used": fallback_used,
        "reason_logs": reason_logs,
        "capabilities": {
            "macos_metal": bool(capabilities.macos_metal),
            "nvidia_cuda": bool(capabilities.nvidia_cuda),
            "details": capabilities.details,
        },
        "tracks": {
            "baseline": baseline,
            "bayesian": bayesian,
        },
    }


def _detect_macos_metal_capability() -> tuple[bool, dict[str, Any]]:
    if platform.system() != "Darwin":
        return False, {"reason": "non_macos"}

    details: dict[str, Any] = {"platform": "Darwin", "probes": []}
    torch = _safe_import("torch")
    if torch is not None:
        try:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None:
                is_built = bool(getattr(mps, "is_built", lambda: False)())
                is_available = bool(getattr(mps, "is_available", lambda: False)())
                details["probes"].append(
                    {
                        "source": "torch.backends.mps",
                        "is_built": is_built,
                        "is_available": is_available,
                    }
                )
                if is_built and is_available:
                    return True, details
        except Exception as torch_error:
            details["probes"].append({"source": "torch.backends.mps", "error": str(torch_error)})

    xcrun_path = shutil.which("xcrun")
    if xcrun_path is not None:
        try:
            completed = subprocess.run(
                [xcrun_path, "-f", "metal"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            available = completed.returncode == 0 and bool(completed.stdout.strip())
            details["probes"].append(
                {
                    "source": "xcrun -f metal",
                    "returncode": int(completed.returncode),
                    "stdout": completed.stdout.strip(),
                }
            )
            return available, details
        except Exception as xcrun_error:
            details["probes"].append({"source": "xcrun -f metal", "error": str(xcrun_error)})

    details["reason"] = "metal_probe_not_available"
    return False, details


def _detect_nvidia_cuda_capability() -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {"probes": []}

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        details["cuda_visible_devices"] = cuda_visible_devices
        if cuda_visible_devices.strip() in {"", "-1"}:
            details["reason"] = "cuda_visibility_disabled"

    torch = _safe_import("torch")
    if torch is not None:
        try:
            is_available = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
            device_count = int(getattr(getattr(torch, "cuda", None), "device_count", lambda: 0)())
            details["probes"].append(
                {
                    "source": "torch.cuda",
                    "is_available": is_available,
                    "device_count": device_count,
                }
            )
            if is_available and device_count > 0:
                return True, details
        except Exception as torch_error:
            details["probes"].append({"source": "torch.cuda", "error": str(torch_error)})

    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is not None:
        try:
            completed = subprocess.run(
                [nvidia_smi_path, "-L"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            has_gpu = completed.returncode == 0 and "GPU" in completed.stdout
            details["probes"].append(
                {
                    "source": "nvidia-smi -L",
                    "returncode": int(completed.returncode),
                    "stdout": completed.stdout.strip(),
                }
            )
            if has_gpu:
                return True, details
        except Exception as smi_error:
            details["probes"].append({"source": "nvidia-smi -L", "error": str(smi_error)})

    return False, details


def _resolve_backend_for_track(
    *,
    track: str,
    requested: ComputeBackendConfig,
    capabilities: ComputeBackendCapabilities,
) -> dict[str, Any]:
    reason_logs: list[str] = []
    gpu_preferred_for_track = _is_gpu_preferred_for_track(track=track, prefer_gpu_for=requested.prefer_gpu_for)
    requested_mode = str(requested.mode)

    if requested_mode == "cpu":
        effective_backend = "cpu"
    elif requested_mode == "auto":
        if not gpu_preferred_for_track:
            effective_backend = "cpu"
        elif capabilities.nvidia_cuda:
            effective_backend = "nvidia_cuda"
        elif capabilities.macos_metal:
            effective_backend = "macos_metal"
        else:
            effective_backend = "cpu"
            reason_logs.append(
                f"compute_backend auto selected CPU for {track} track because no GPU backend was detected"
            )
    elif requested_mode == "nvidia_cuda":
        if not gpu_preferred_for_track:
            effective_backend = "cpu"
            reason_logs.append(
                f"compute_backend requested nvidia_cuda but prefer_gpu_for excludes {track}; using CPU"
            )
        elif capabilities.nvidia_cuda:
            effective_backend = "nvidia_cuda"
        else:
            effective_backend = "cpu"
            reason_logs.append(
                f"compute_backend requested nvidia_cuda for {track} but CUDA/NVIDIA is unavailable; using CPU"
            )
    elif requested_mode == "macos_metal":
        if not gpu_preferred_for_track:
            effective_backend = "cpu"
            reason_logs.append(
                f"compute_backend requested macos_metal but prefer_gpu_for excludes {track}; using CPU"
            )
        elif capabilities.macos_metal:
            effective_backend = "macos_metal"
        else:
            effective_backend = "cpu"
            reason_logs.append(
                f"compute_backend requested macos_metal for {track} but Metal is unavailable; using CPU"
            )
    else:
        effective_backend = "cpu"
        reason_logs.append(f"Unrecognized compute backend mode '{requested_mode}' for {track}; using CPU")

    fallback_applied = bool(effective_backend == "cpu" and requested_mode != "cpu")
    return {
        "track": track,
        "requested_mode": requested_mode,
        "gpu_preferred_for_track": gpu_preferred_for_track,
        "effective_backend": effective_backend,
        "fallback_backend": requested.fallback,
        "fallback_used": fallback_applied,
        "reason_logs": reason_logs,
    }


def _is_gpu_preferred_for_track(*, track: str, prefer_gpu_for: str) -> bool:
    preferred = str(prefer_gpu_for).lower()
    if preferred == "both":
        return True
    if preferred == "baselines":
        return track == "baseline"
    if preferred == "bayesian":
        return track == "bayesian"
    return True


def _safe_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None
