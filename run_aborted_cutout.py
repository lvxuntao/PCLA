#!/usr/bin/env python3
"""Launch the InterFuser aborted cut-out scenario from any directory."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_SR = Path("/home/xlyu5/Research/scenario_runner")
DEFAULT_PCLA = Path("/home/xlyu5/Research/PCLA")
DEFAULT_SCENARIO = DEFAULT_PCLA / "interfuser_aborted_cutout_scenario.py"
DEFAULT_CONFIG = DEFAULT_PCLA / "interfuser_aborted_cutout.xml"


def parse_args():
    p = argparse.ArgumentParser(description="Run one aborted cut-out InterFuser test")
    p.add_argument("--initial-gap", type=float, default=12.0)
    p.add_argument("--npc-speed", type=float, default=3.5)
    p.add_argument("--trigger-gap", type=float, default=9.0)
    p.add_argument(
        "--max-follow-time",
        type=float,
        default=8.0,
        help="Trigger cut-out after this many seconds even if trigger-gap is not reached; set 0 to disable.",
    )
    p.add_argument("--cutout-duration", type=float, default=1.0)
    p.add_argument("--abort-progress", type=float, default=0.55)
    p.add_argument("--return-duration", type=float, default=0.6)
    p.add_argument("--npc-brake", type=float, default=0.8)
    p.add_argument("--brake-delay-after-abort", type=float, default=0.35)
    p.add_argument("--brake-duration", type=float, default=1.5)
    p.add_argument("--return-lateral-tolerance", type=float, default=0.25)
    p.add_argument("--npc-after-speed", type=float, default=0.5)
    p.add_argument("--post-observation", type=float, default=6.0)
    p.add_argument("--timeout", type=float, default=45.0)
    p.add_argument("--fixed-dt", type=float, default=0.05)
    p.add_argument("--scenario-runner-dir", type=Path, default=DEFAULT_SR)
    p.add_argument("--scenario-file", type=Path, default=DEFAULT_SCENARIO)
    p.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--scenario-name", default="InterfuserAbortedCutOut_1")
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--no-reload-world", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def validate(a):
    if a.initial_gap <= 0 or a.trigger_gap <= 0:
        raise ValueError("--initial-gap and --trigger-gap must be positive")
    if a.trigger_gap >= a.initial_gap:
        raise ValueError("--trigger-gap must be smaller than --initial-gap")
    if a.max_follow_time < 0:
        raise ValueError("--max-follow-time must be >= 0")
    if not 0.0 < a.abort_progress < 1.0:
        raise ValueError("--abort-progress must be in (0, 1)")
    if a.cutout_duration <= 0 or a.return_duration <= 0:
        raise ValueError("lane-change durations must be positive")
    if not 0.0 <= a.npc_brake <= 1.0:
        raise ValueError("--npc-brake must be in [0, 1]")
    if a.brake_delay_after_abort < 0:
        raise ValueError("--brake-delay-after-abort cannot be negative")
    if a.brake_duration <= 0 or a.fixed_dt <= 0:
        raise ValueError("--brake-duration and --fixed-dt must be positive")
    if a.return_lateral_tolerance <= 0:
        raise ValueError("--return-lateral-tolerance must be positive")
    if a.npc_speed < 0 or a.npc_after_speed < 0:
        raise ValueError("NPC speeds cannot be negative")

    runner = a.scenario_runner_dir / "scenario_runner.py"
    if not runner.exists():
        raise FileNotFoundError(f"ScenarioRunner not found: {runner}")
    if not a.scenario_file.exists():
        raise FileNotFoundError(f"Scenario file not found: {a.scenario_file}")


def ensure_config(path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '''<?xml version="1.0"?>
<scenarios>
  <scenario name="InterfuserAbortedCutOut_1"
            type="InterfuserAbortedCutOut"
            town="Town06">
    <ego_vehicle
        x="397.32"
        y="-24.38"
        z="0.5"
        yaw="179.8"
        model="vehicle.tesla.model3"
        rolename="hero" />
  </scenario>
</scenarios>
''',
        encoding="utf-8",
    )
    print(f"Generated missing config file: {path}")


def main():
    a = parse_args()
    validate(a)
    ensure_config(a.config_file)

    env = os.environ.copy()
    env.update({
        "ACO_INITIAL_GAP": str(a.initial_gap),
        "ACO_NPC_SPEED": str(a.npc_speed),
        "ACO_TRIGGER_GAP": str(a.trigger_gap),
        "ACO_MAX_FOLLOW_TIME": str(a.max_follow_time),
        "ACO_CUTOUT_DURATION": str(a.cutout_duration),
        "ACO_ABORT_PROGRESS": str(a.abort_progress),
        "ACO_RETURN_DURATION": str(a.return_duration),
        "ACO_NPC_BRAKE": str(a.npc_brake),
        "ACO_BRAKE_DELAY_AFTER_ABORT": str(a.brake_delay_after_abort),
        "ACO_BRAKE_DURATION": str(a.brake_duration),
        "ACO_RETURN_LATERAL_TOLERANCE": str(a.return_lateral_tolerance),
        "ACO_NPC_AFTER_SPEED": str(a.npc_after_speed),
        "ACO_POST_OBSERVATION": str(a.post_observation),
        "ACO_TIMEOUT": str(a.timeout),
        "ACO_FIXED_DT": str(a.fixed_dt),
    })

    runner = a.scenario_runner_dir / "scenario_runner.py"
    command = [
        sys.executable,
        str(runner),
        "--scenario", a.scenario_name,
        "--additionalScenario", str(a.scenario_file),
        "--configFile", str(a.config_file),
        "--sync",
        "--output",
    ]
    if not a.no_reload_world:
        command.append("--reloadWorld")
    if a.debug:
        command.append("--debug")

    print("Running ScenarioRunner command:")
    print(" ".join(command))
    print(
        "Parameters: "
        f"initial_gap={a.initial_gap}, npc_speed={a.npc_speed}, "
        f"trigger_gap={a.trigger_gap}, max_follow_time={a.max_follow_time}, "
        f"cutout_duration={a.cutout_duration}, "
        f"abort_progress={a.abort_progress}, return_duration={a.return_duration}, "
        f"npc_brake={a.npc_brake}, "
        f"brake_delay_after_abort={a.brake_delay_after_abort}, "
        f"brake_duration={a.brake_duration}, "
        f"return_lateral_tolerance={a.return_lateral_tolerance}, "
        f"npc_after_speed={a.npc_after_speed}"
    )
    print()

    if a.log is None:
        result = subprocess.run(
            command,
            cwd=a.scenario_runner_dir,
            env=env,
            check=False,
        )
    else:
        a.log.parent.mkdir(parents=True, exist_ok=True)
        with a.log.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command,
                cwd=a.scenario_runner_dir,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        print(f"Complete output written to: {a.log}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
