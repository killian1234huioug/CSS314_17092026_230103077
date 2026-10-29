"""Collect host telemetry and salted parameters for CS482 Milestone 0.

Run from the repository root:
    python3 src/profile_env.py --student-id YOUR_STUDENT_ID
Writes results/host_profile.json and results/salted_params.json.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from salt import get_student_params  # noqa: E402


def run_command(args):
    """Run a system command; return stripped stdout or None on failure."""
    try:
        result = subprocess.run(
            args, check=False, capture_output=True, text=True, timeout=8
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def sysctl_value(key):
    return run_command(["sysctl", "-n", key])


def int_or_none(value):
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def linux_cpu_info():
    info = {}
    try:
        text = Path("/proc/cpuinfo").read_text(errors="replace")
        first = text.split("\n\n", 1)[0]
        for line in first.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    except OSError:
        pass
    return info


def mac_hardware_info():
    """Return hardware fields where macOS exposes them."""
    chip = sysctl_value("machdep.cpu.brand_string")
    model = sysctl_value("hw.model")
    if not chip or "Apple" not in chip:
        hardware_text = run_command(
            ["system_profiler", "SPHardwareDataType"]
        ) or ""
        for line in hardware_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Chip:"):
                chip = stripped.split(":", 1)[1].strip()
                break
            if stripped.startswith("Processor Name:") and not chip:
                chip = stripped.split(":", 1)[1].strip()
    return chip, model


def get_host_profile():
    system = platform.system()
    machine = platform.machine()
    cpu_model = None
    microarchitecture = None
    base_frequency_mhz = None
    physical_cores = None
    logical_threads = os.cpu_count()
    cache = {
        "l1_data_bytes": None,
        "l2_bytes": None,
        "l3_shared_bytes": None,
        "cache_line_bytes": None,
    }
    detection_notes = []

    if system == "Darwin":
        chip, model = mac_hardware_info()
        cpu_model = chip or model or platform.processor() or "Unknown (macOS)"

        if chip and re.search(r"\bM[1-5]\b", chip, re.IGNORECASE):
            match = re.search(r"\b(M[1-5])\b", chip, re.IGNORECASE)
            microarchitecture = f"Apple {match.group(1).upper()} family (exact core microarchitecture not exposed)"
        else:
            microarchitecture = "Not reliably exposed by macOS command-line telemetry"
        physical_cores = int_or_none(sysctl_value("hw.physicalcpu"))
        logical_threads = int_or_none(sysctl_value("hw.logicalcpu")) or logical_threads
        cache_keys = {
            "l1_data_bytes": "hw.l1dcachesize",
            "l2_bytes": "hw.l2cachesize",
            "l3_shared_bytes": "hw.l3cachesize",
            "cache_line_bytes": "hw.cachelinesize",
        }
        for field, key in cache_keys.items():
            cache[field] = int_or_none(sysctl_value(key))
        if any(v is None for v in cache.values()):
            detection_notes.append(
                "Some cache values are not exposed by sysctl on this macOS version; "
                "null means unavailable, not zero."
            )
        detection_notes.append(
            "Apple Silicon base frequency is not reliably exposed by standard macOS "
            "command-line telemetry; base_frequency_mhz is null rather than guessed."
        )

    elif system == "Linux":
        info = linux_cpu_info()
        cpu_model = (
            info.get("model name")
            or info.get("Hardware")
            or info.get("Processor")
            or platform.processor()
            or "Unknown (Linux)"
        )
        microarchitecture = "Not inferred from model name; see cpu_model"
        base_frequency_mhz = int_or_none(info.get("cpu MHz"))
        try:
            physical_cores = os.cpu_count()
            text = Path("/proc/cpuinfo").read_text(errors="replace")
            pairs = set()
            for block in text.strip().split("\n\n"):
                fields = dict(
                    line.split(":", 1) for line in block.splitlines() if ":" in line
                )
                if "physical id" in fields and "core id" in fields:
                    pairs.add((fields["physical id"].strip(), fields["core id"].strip()))
            if pairs:
                physical_cores = len(pairs)
        except OSError:
            pass
        cache["cache_line_bytes"] = 64
        detection_notes.append(
            "Linux cache sizes may be read from /sys/devices/system/cpu/cpu0/cache; "
            "values are left null if unavailable."
        )
        cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
        if cache_root.exists():
            for entry in cache_root.glob("index*"):
                try:
                    level = (entry / "level").read_text().strip()
                    kind = (entry / "type").read_text().strip()
                    size_text = (entry / "size").read_text().strip().upper()
                    match = re.fullmatch(r"(\d+)([KMG]?)", size_text)
                    if not match:
                        continue
                    size = int(match.group(1))
                    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
                    size *= multiplier
                    if level == "1" and kind == "Data":
                        cache["l1_data_bytes"] = size
                    elif level == "2":
                        cache["l2_bytes"] = size
                    elif level == "3":
                        cache["l3_shared_bytes"] = size
                except OSError:
                    continue
    else:
        cpu_model = platform.processor() or platform.machine() or "Unknown"
        microarchitecture = "Not available through the standard cross-platform APIs"
        detection_notes.append(
            "Cache telemetry is platform-dependent and may be unavailable on this OS."
        )

    if physical_cores is None:
        detection_notes.append(
            "Physical core count could not be determined reliably; see logical_threads."
        )

    return {
        "cpu": {
            "model_name": cpu_model,
            "microarchitecture": microarchitecture,
            "base_frequency_mhz": base_frequency_mhz,
            "physical_core_count": physical_cores,
            "logical_execution_threads": logical_threads,
            "smt_or_hyper_threading": (
                None if physical_cores is None or logical_threads is None
                else logical_threads > physical_cores
            ),
        },
        "cache": cache,
        "operating_system": {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "machine": machine,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "detection_notes": detection_notes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True, help="Student ID used for salting")
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = get_host_profile()
    salted = get_student_params(args.student_id)

    host_path = output_dir / "host_profile.json"
    salted_path = output_dir / "salted_params.json"
    host_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    salted_path.write_text(json.dumps(salted, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] Wrote {host_path}")
    print(f"[OK] Wrote {salted_path}")
    print(json.dumps(salted, indent=2))
    print("Review host_profile.json: unavailable telemetry is recorded as null, not guessed.")


if __name__ == "__main__":
    main()
