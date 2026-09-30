#!/usr/bin/env python3
"""
Run the ScenarioRunner-native InterFuser partial cut-in scenario from any directory.

Example:
    python run_partial_cutin.py \
      --partial-progress 0.62 \
      --trigger-gap 8 \
      --lane-change-duration 0.8 \
      --npc-speed 4 \
      --npc-after-speed 1.5
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_SCENARIO_RUNNER = Path("/home/xlyu5/Research/scenario_runner")
DEFAULT_PCLA_DIR = Path("/home/xlyu5/Research/PCLA")
DEFAULT_SCENARIO_FILE = DEFAULT_PCLA_DIR / "interfuser_partial_cutin_scenario_cli.py"
DEFAULT_CONFIG_FILE = DEFAULT_PCLA_DIR / "interfuser_partial_cutin.xml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one InterFuser partial cut-in ScenarioRunner test."
    )

    parser.add_argument("--initial-gap", type=float, default=20.0)
    parser.add_argument("--npc-speed", type=float, default=4.0)
    parser.add_argument("--npc-after-speed", type=float, default=2.0)
    parser.add_argument("--trigger-gap", type=float, default=10.0)
    parser.add_argument("--lane-change-duration", type=float, default=1.0)
    parser.add_argument("--partial-progress", type=float, default=0.58)
    parser.add_argument("--post-observation", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--fixed-dt", type=float, default=0.05)

    parser.add_argument(
        "--scenario-runner-dir",
        type=Path,
        default=DEFAULT_SCENARIO_RUNNER,
    )
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=DEFAULT_SCENARIO_FILE,
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
    )
    parser.add_argument(
        "--scenario-name",
        default="InterfuserPartialCutIn_1",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional file that receives the complete ScenarioRunner output.",
    )
    parser.add_argument(
        "--no-reload-world",
        action="store_true",
        help="Do not pass --reloadWorld.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass --debug to ScenarioRunner.",
    )

    return parser.parse_args()


def validate(args):
    if not 0.0 <= args.partial_progress <= 1.0:
        raise ValueError("--partial-progress must be in [0, 1]")
    if args.initial_gap <= 0:
        raise ValueError("--initial-gap must be positive")
    if args.trigger_gap <= 0:
        raise ValueError("--trigger-gap must be positive")
    if args.lane_change_duration <= 0:
        raise ValueError("--lane-change-duration must be positive")
    if args.npc_speed < 0 or args.npc_after_speed < 0:
        raise ValueError("NPC speeds cannot be negative")
    if args.fixed_dt <= 0:
        raise ValueError("--fixed-dt must be positive")

    runner = args.scenario_runner_dir / "scenario_runner.py"
    if not runner.exists():
        raise FileNotFoundError(f"ScenarioRunner not found: {runner}")
    if not args.scenario_file.exists():
        raise FileNotFoundError(f"Scenario file not found: {args.scenario_file}")
    if not args.config_file.exists():
        args.config_file.parent.mkdir(parents=True, exist_ok=True)
        args.config_file.write_text(
            '''<?xml version="1.0"?>
<scenarios>
  <scenario name="InterfuserPartialCutIn_1"
            type="InterfuserPartialCutIn"
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
        print(f"Generated missing config file: {args.config_file}")


def main():
    args = parse_args()
    validate(args)

    runner = args.scenario_runner_dir / "scenario_runner.py"

    env = os.environ.copy()
    env.update(
        {
            "PC_INITIAL_GAP": str(args.initial_gap),
            "PC_NPC_SPEED": str(args.npc_speed),
            "PC_NPC_AFTER_SPEED": str(args.npc_after_speed),
            "PC_TRIGGER_GAP": str(args.trigger_gap),
            "PC_LANE_CHANGE_DURATION": str(args.lane_change_duration),
            "PC_PARTIAL_PROGRESS": str(args.partial_progress),
            "PC_POST_OBSERVATION": str(args.post_observation),
            "PC_TIMEOUT": str(args.timeout),
            "PC_FIXED_DT": str(args.fixed_dt),
        }
    )

    command = [
        sys.executable,
        str(runner),
        "--scenario",
        args.scenario_name,
        "--additionalScenario",
        str(args.scenario_file),
        "--configFile",
        str(args.config_file),
        "--sync",
        "--output",
    ]

    if not args.no_reload_world:
        command.append("--reloadWorld")
    if args.debug:
        command.append("--debug")

    print("Running ScenarioRunner command:")
    print(" ".join(command))
    print(
        "Parameters: "
        f"initial_gap={args.initial_gap}, "
        f"npc_speed={args.npc_speed}, "
        f"npc_after_speed={args.npc_after_speed}, "
        f"trigger_gap={args.trigger_gap}, "
        f"lane_change_duration={args.lane_change_duration}, "
        f"partial_progress={args.partial_progress}"
    )
    print()

    if args.log is None:
        completed = subprocess.run(
            command,
            cwd=args.scenario_runner_dir,
            env=env,
            check=False,
        )
    else:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                command,
                cwd=args.scenario_runner_dir,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        print(f"Complete output written to: {args.log}")

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
