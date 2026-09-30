#!/usr/bin/env python3

'''
python search_min_safe_distance.py \
  --low 5 \
  --high 60 \
  --iters 8 \
  --repeats 1 \
  --ego-speed 10 \
  --front-speed 5 \
  --clean
'''

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SCENARIO_RUNNER_DIR = Path("/home/xlyu5/Research/scenario_runner")
PCLA_DIR = Path("/home/xlyu5/Research/PCLA")
CARLA_ROOT = Path("/home/xlyu5/Research/CARLA/carla15")
INTERFUSER_ROOT = PCLA_DIR / "pcla_agents" / "interfuser"

SCENARIO_FILE = PCLA_DIR / "front_brake_handoff_scenario.py"
CONFIG_FILE = PCLA_DIR / "front_brake_handoff.xml"
INTERFUSER_CONFIG_FILE = INTERFUSER_ROOT / "interfuser_config.py"

LOG_DIR = PCLA_DIR / "minimum_safe_distance_logs"


def replace_assignment(text, name, value):
    pattern = rf"^(\s*){re.escape(name)}\s*=\s*.*$"

    def repl(match):
        indent = match.group(1)
        return f"{indent}{name} = {value}"

    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)

    if n != 1:
        raise RuntimeError(f"Could not replace assignment for {name}")

    return new_text


def patch_scenario_file(original_text, args, trigger_gap):
    """
    Patch the user's current scenario file.

    This intentionally works with the current file structure:
    - fixed constants at the top
    - time.sleep(0.05) inside update()
    """

    initial_gap = max(args.initial_gap, trigger_gap + args.initial_gap_margin)

    text = original_text

    text = replace_assignment(
        text,
        "EGO_TARGET_SPEED_BEFORE_TRIGGER",
        f"{args.ego_speed:.6f}"
    )

    text = replace_assignment(
        text,
        "FRONT_TARGET_SPEED",
        f"{args.front_speed:.6f}"
    )

    text = replace_assignment(
        text,
        "TRIGGER_GAP",
        f"{trigger_gap:.6f}"
    )

    text = replace_assignment(
        text,
        "INITIAL_FRONT_GAP",
        f"{initial_gap:.6f}"
    )

    text = replace_assignment(
        text,
        "SCENARIO_TIMEOUT",
        f"{args.scenario_timeout:.6f}"
    )

    # For automated search, make it fast by default.
    # Your scenario currently has time.sleep(0.05); this replaces it temporarily.
    text, n = re.subn(
        r"time\.sleep\(\s*[0-9.]+\s*\)",
        f"time.sleep({args.realtime_sleep:.6f})",
        text,
        count=1
    )

    if n != 1:
        raise RuntimeError("Could not replace time.sleep(...) in scenario file")

    SCENARIO_FILE.write_text(text)

    return initial_gap

def patch_interfuser_config(original_text, args):
    """
    Patch pcla_agents/interfuser/interfuser_config.py so that
    GlobalConfig.max_speed matches the ego speed used in this search.
    """

    text = replace_assignment(
        original_text,
        "max_speed",
        f"{args.ego_speed:.6f}"
    )

    INTERFUSER_CONFIG_FILE.write_text(text)

def build_env():
    env = os.environ.copy()

    extra_paths = [
        str(PCLA_DIR),
        str(INTERFUSER_ROOT),
        str(SCENARIO_RUNNER_DIR),
        str(CARLA_ROOT / "PythonAPI" / "carla"),
    ]

    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        env["PYTHONPATH"] = ":".join(extra_paths + [old_pythonpath])
    else:
        env["PYTHONPATH"] = ":".join(extra_paths)

    env["CARLA_ROOT"] = str(CARLA_ROOT)
    env["SCENARIO_RUNNER_ROOT"] = str(SCENARIO_RUNNER_DIR)
    env["PCLA_ROOT"] = str(PCLA_DIR)
    env["INTERFUSER_ROOT"] = str(INTERFUSER_ROOT)

    return env


def parse_global_result(output):
    for line in output.splitlines():
        if "GLOBAL RESULT" in line:
            if "SUCCESS" in line:
                return "SUCCESS"
            if "FAILURE" in line:
                return "FAILURE"
            if "TIMEOUT" in line:
                return "TIMEOUT"

    return "UNKNOWN"


def parse_collision_result(output):
    """
    Return:
        safe=True  if all collision tests are SUCCESS
        safe=False if any collision test is FAILURE

    We intentionally ignore Timeout here.
    A run can be TIMEOUT but still collision-free.
    """

    collision_lines = []

    for line in output.splitlines():
        if (
            "CollisionTest" in line
            or "EgoCollisionTest" in line
            or "FrontCollisionTest" in line
        ):
            collision_lines.append(line)

    if not collision_lines:
        raise RuntimeError("Could not find CollisionTest lines in ScenarioRunner output")

    if any("FAILURE" in line for line in collision_lines):
        return False, collision_lines

    if any("SUCCESS" in line for line in collision_lines):
        return True, collision_lines

    raise RuntimeError("Found CollisionTest lines, but could not parse SUCCESS/FAILURE")


def run_one_gap(args, original_scenario_text, original_interfuser_config_text, trigger_gap, trial):
    initial_gap = patch_scenario_file(
        original_text=original_scenario_text,
        args=args,
        trigger_gap=trigger_gap
    )

    patch_interfuser_config(
        original_text=original_interfuser_config_text,
        args=args
    )

    env = build_env()

    cmd = [
        sys.executable,
        "scenario_runner.py",
        "--scenario",
        "FrontBrakeHandoff_1",
        "--additionalScenario",
        str(SCENARIO_FILE),
        "--configFile",
        str(CONFIG_FILE),
        "--sync",
        "--reloadWorld",
        "--output",
    ]

    print(
        f"\n=== Test trigger_gap={trigger_gap:.3f} m | "
        f"trial={trial} | initial_gap={initial_gap:.3f} m ===",
        flush=True
    )

    proc = subprocess.run(
        cmd,
        cwd=str(SCENARIO_RUNNER_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.process_timeout
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / (
        f"gap_{trigger_gap:.3f}_trial_{trial}_"
        f"ego_{args.ego_speed:.1f}_front_{args.front_speed:.1f}.log"
    )

    log_path.write_text(proc.stdout)

    try:
        safe, collision_lines = parse_collision_result(proc.stdout)
        parse_error = ""
    except Exception as e:
        safe = False
        collision_lines = []
        parse_error = str(e)

    global_result = parse_global_result(proc.stdout)

    print(
        f"Result: {'SAFE' if safe else 'UNSAFE'} | "
        f"GLOBAL={global_result} | "
        f"returncode={proc.returncode} | "
        f"log={log_path}",
        flush=True
    )

    if parse_error:
        print(f"Parse error: {parse_error}", flush=True)

    return {
        "gap": trigger_gap,
        "trial": trial,
        "initial_gap": initial_gap,
        "ego_speed": args.ego_speed,
        "front_speed": args.front_speed,
        "safe": safe,
        "global_result": global_result,
        "returncode": proc.returncode,
        "parse_error": parse_error,
        "log_path": str(log_path),
    }


def run_repeats(args, original_scenario_text, original_interfuser_config_text, trigger_gap):
    rows = []

    for trial in range(1, args.repeats + 1):
        row = run_one_gap(
            args=args,
            original_scenario_text=original_scenario_text,
            original_interfuser_config_text=original_interfuser_config_text,
            trigger_gap=trigger_gap,
            trial=trial
        )
        rows.append(row)

    # Conservative rule:
    # A gap is safe only if all repeated runs are collision-free.
    safe = all(row["safe"] for row in rows)

    return safe, rows


def append_csv(rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = LOG_DIR / "binary_search_results.csv"
    exists = csv_path.exists()

    fieldnames = [
        "gap",
        "trial",
        "initial_gap",
        "ego_speed",
        "front_speed",
        "safe",
        "global_result",
        "returncode",
        "parse_error",
        "log_path",
    ]

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Search minimum safe trigger distance for your current ScenarioRunner file."
    )

    parser.add_argument("--low", type=float, default=5.0)
    parser.add_argument("--high", type=float, default=60.0)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)

    parser.add_argument("--ego-speed", type=float, default=15.0)
    parser.add_argument("--front-speed", type=float, default=5.0)

    parser.add_argument(
        "--initial-gap",
        type=float,
        default=60.0,
        help="Base initial center-to-center gap."
    )

    parser.add_argument(
        "--initial-gap-margin",
        type=float,
        default=60.0,
        help="Actual initial gap = max(initial_gap, trigger_gap + margin)."
    )

    parser.add_argument(
        "--scenario-timeout",
        type=float,
        default=60.0
    )

    parser.add_argument(
        "--process-timeout",
        type=float,
        default=240.0
    )

    parser.add_argument(
        "--realtime-sleep",
        type=float,
        default=0.0,
        help="Use 0.0 for fast search. Use 0.05 if you want visualization."
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete old logs and CSV before running."
    )

    args = parser.parse_args()

    if not SCENARIO_FILE.exists():
        raise FileNotFoundError(SCENARIO_FILE)

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(CONFIG_FILE)

    if args.clean and LOG_DIR.exists():
        shutil.rmtree(LOG_DIR)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    original_scenario_text = SCENARIO_FILE.read_text()
    original_interfuser_config_text = INTERFUSER_CONFIG_FILE.read_text()

    backup_path = SCENARIO_FILE.parent / (SCENARIO_FILE.name + ".bak_min_search")
    backup_path.write_text(original_scenario_text)

    interfuser_backup_path = INTERFUSER_CONFIG_FILE.parent / (
        INTERFUSER_CONFIG_FILE.name + ".bak_min_search"
    )
    interfuser_backup_path.write_text(original_interfuser_config_text)

    print("\n========== Minimum Safe Distance Search ==========")
    print(f"scenario file : {SCENARIO_FILE}")
    print(f"backup file   : {backup_path}")
    print(f"ego speed     : {args.ego_speed:.3f} m/s")
    print(f"front speed   : {args.front_speed:.3f} m/s")
    print(f"low           : {args.low:.3f} m")
    print(f"high          : {args.high:.3f} m")
    print(f"iters         : {args.iters}")
    print(f"repeats       : {args.repeats}")
    print(f"realtime_sleep: {args.realtime_sleep}")
    print(f"log dir       : {LOG_DIR}")
    print("==================================================\n")

    csv_path = LOG_DIR / "binary_search_results.csv"

    try:
        # Check low bound.
        print("Checking low bound...", flush=True)
        low_safe, rows = run_repeats(args, original_scenario_text, original_interfuser_config_text, args.low)
        csv_path = append_csv(rows)

        if low_safe:
            print(
                f"\nWARNING: low={args.low:.3f} m is already safe. "
                f"Decrease --low if you want a valid unsafe lower bound.",
                flush=True
            )
            print(f"CSV: {csv_path}", flush=True)
            return 3

        # Check high bound.
        print("\nChecking high bound...", flush=True)
        high_safe, rows = run_repeats(args, original_scenario_text, original_interfuser_config_text, args.high)
        csv_path = append_csv(rows)

        if not high_safe:
            print(
                f"\nERROR: high={args.high:.3f} m is still unsafe. "
                f"Increase --high and run again.",
                flush=True
            )
            print(f"CSV: {csv_path}", flush=True)
            return 2

        low = args.low
        high = args.high

        print(
            f"\nStart binary search with bracket: "
            f"low={low:.3f} unsafe, high={high:.3f} safe",
            flush=True
        )

        for i in range(1, args.iters + 1):
            mid = 0.5 * (low + high)

            print(
                f"\n--- Iteration {i}/{args.iters}: testing gap={mid:.3f} m ---",
                flush=True
            )

            mid_safe, rows = run_repeats(args, original_scenario_text, original_interfuser_config_text, mid)
            csv_path = append_csv(rows)

            if mid_safe:
                high = mid
                decision = "SAFE -> high = mid"
            else:
                low = mid
                decision = "UNSAFE -> low = mid"

            print(
                f"Decision: {decision}. "
                f"Current interval: low={low:.3f}, high={high:.3f}",
                flush=True
            )

        print("\n================ Final Result ================")
        print(f"Estimated minimum safe trigger distance: {high:.3f} m center-to-center")
        print(f"Unsafe lower bound neighbor:             {low:.3f} m center-to-center")
        print(f"CSV results:                             {csv_path}")
        print(f"Logs:                                    {LOG_DIR}")
        print("==============================================\n")

        return 0

    finally:
        # Restore your original scenario file after search.
        SCENARIO_FILE.write_text(original_scenario_text)
        INTERFUSER_CONFIG_FILE.write_text(original_interfuser_config_text)

        print(f"\nRestored original scenario file: {SCENARIO_FILE}", flush=True)
        print(f"Restored original InterFuser config file: {INTERFUSER_CONFIG_FILE}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())