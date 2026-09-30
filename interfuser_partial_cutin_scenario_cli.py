#!/usr/bin/env python3
"""
ScenarioRunner-native InterFuser partial cut-in regression scenario.

Run from ScenarioRunner:
    cd /home/xlyu5/Research/scenario_runner
    python scenario_runner.py \
      --scenario InterfuserPartialCutIn_1 \
      --additionalScenario /home/xlyu5/Research/PCLA/interfuser_partial_cutin_scenario.py \
      --configFile /home/xlyu5/Research/PCLA/interfuser_partial_cutin.xml \
      --sync --reloadWorld --output
"""

import math
import os
import xml.etree.ElementTree as ET

import carla
import py_trees

from PCLA import PCLA
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest


# =========================
# Experiment configuration
# =========================
PCLA_AGENT = "if_if"
PCLA_ROUTE = "/home/xlyu5/Research/PCLA/town06_partial_cutin_route.xml"

def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None else float(default)


FIXED_DT = env_float("PC_FIXED_DT", 0.05)
NPC_INITIAL_GAP = env_float("PC_INITIAL_GAP", 20.0)
NPC_CRUISE_SPEED = env_float("PC_NPC_SPEED", 4.0)
NPC_AFTER_CUTIN_SPEED = env_float("PC_NPC_AFTER_SPEED", 2.0)
CUTIN_TRIGGER_GAP = env_float("PC_TRIGGER_GAP", 10.0)
LANE_CHANGE_DURATION = env_float("PC_LANE_CHANGE_DURATION", 1.0)
PARTIAL_PROGRESS = env_float("PC_PARTIAL_PROGRESS", 0.58)
POST_CUTIN_OBSERVATION = env_float("PC_POST_OBSERVATION", 5.0)
SCENARIO_TIMEOUT = env_float("PC_TIMEOUT", 45.0)

ROUTE_LENGTH = 210.0
ROUTE_STEP = 8.0


def clamp(value, low, high):
    return max(low, min(high, value))


def get_speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(
        velocity.x * velocity.x
        + velocity.y * velocity.y
        + velocity.z * velocity.z
    )


def angle_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def same_direction(wp1, wp2):
    f1 = wp1.transform.get_forward_vector()
    f2 = wp2.transform.get_forward_vector()
    return f1.x * f2.x + f1.y * f2.y > 0.8


def valid_left_lane(wp):
    left = wp.get_left_lane()
    if left is None:
        return None
    if left.lane_type != carla.LaneType.Driving:
        return None
    if not same_direction(wp, left):
        return None
    return left


def choose_forward_waypoint(current_wp, reference_yaw, distance):
    candidates = [
        wp
        for wp in current_wp.next(distance)
        if wp.lane_type == carla.LaneType.Driving
    ]
    if not candidates:
        return None

    same_lane = [wp for wp in candidates if wp.lane_id == current_wp.lane_id]
    candidates = same_lane if same_lane else candidates

    return min(
        candidates,
        key=lambda wp: angle_diff_deg(
            wp.transform.rotation.yaw, reference_yaw
        ),
    )


def waypoint_ahead(start_wp, distance, step=4.0):
    reference_yaw = start_wp.transform.rotation.yaw
    current_wp = start_wp
    travelled = 0.0

    while travelled < distance:
        increment = min(step, distance - travelled)
        next_wp = choose_forward_waypoint(
            current_wp, reference_yaw, increment
        )
        if next_wp is None:
            raise RuntimeError(
                f"Cannot find waypoint {distance:.1f} m ahead"
            )
        current_wp = next_wp
        travelled += increment

    return current_wp


def write_route_xml(path, town, start_wp):
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id="0", town=town)

    current_wp = start_wp
    reference_yaw = start_wp.transform.rotation.yaw
    point_count = int(ROUTE_LENGTH // ROUTE_STEP) + 1

    for _ in range(point_count):
        tf = current_wp.transform
        ET.SubElement(
            route,
            "waypoint",
            x=f"{tf.location.x:.6f}",
            y=f"{tf.location.y:.6f}",
            z=f"{tf.location.z:.6f}",
            pitch=f"{tf.rotation.pitch:.6f}",
            yaw=f"{tf.rotation.yaw:.6f}",
            roll=f"{tf.rotation.roll:.6f}",
        )

        next_wp = choose_forward_waypoint(
            current_wp, reference_yaw, ROUTE_STEP
        )
        if next_wp is None or next_wp.is_junction:
            break
        current_wp = next_wp

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(f"Generated route: {path}", flush=True)


def smoothstep(value):
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def world_to_vehicle_local(vehicle_transform, target_location):
    dx = target_location.x - vehicle_transform.location.x
    dy = target_location.y - vehicle_transform.location.y
    yaw = math.radians(vehicle_transform.rotation.yaw)

    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return local_x, local_y


def speed_control(current_speed, target_speed):
    error = target_speed - current_speed
    if error > 1.0:
        return 0.55, 0.0
    if error > 0.25:
        return 0.25, 0.0
    if error < -1.0:
        return 0.0, 0.45
    if error < -0.25:
        return 0.0, 0.18
    return 0.08, 0.0


def signed_lateral_offset(ego_lane_wp, vehicle):
    right = ego_lane_wp.transform.get_right_vector()
    delta_x = vehicle.get_location().x - ego_lane_wp.transform.location.x
    delta_y = vehicle.get_location().y - ego_lane_wp.transform.location.y
    return delta_x * right.x + delta_y * right.y


class InterfuserPartialCutInBehavior(AtomicBehavior):
    """
    Ego is controlled by PCLA/InterFuser from the first tick.
    NPC drives in the left lane and performs a continuous partial cut-in.
    """

    def __init__(
        self,
        ego,
        npc,
        ego_start_wp,
        client,
        name="InterfuserPartialCutInBehavior",
    ):
        super().__init__(name)
        self.ego = ego
        self.npc = npc
        self.ego_start_wp = ego_start_wp
        self.client = client

        self.pcla = None
        self.elapsed = 0.0
        self.triggered = False
        self.trigger_time = None
        self.progress = 0.0
        self.path_s = 0.0
        self.safe_after_complete_since = None

    def initialise(self):
        print(
            "[PARAMS] "
            f"initial_gap={NPC_INITIAL_GAP:.2f}, "
            f"npc_speed={NPC_CRUISE_SPEED:.2f}, "
            f"npc_after_speed={NPC_AFTER_CUTIN_SPEED:.2f}, "
            f"trigger_gap={CUTIN_TRIGGER_GAP:.2f}, "
            f"lane_change_duration={LANE_CHANGE_DURATION:.2f}, "
            f"partial_progress={PARTIAL_PROGRESS:.2f}, "
            f"post_observation={POST_CUTIN_OBSERVATION:.2f}",
            flush=True,
        )
        self.elapsed = 0.0
        self.triggered = False
        self.trigger_time = None
        self.progress = 0.0
        self.path_s = 0.0
        self.safe_after_complete_since = None

        write_route_xml(
            PCLA_ROUTE,
            CarlaDataProvider.get_map().name.split("/")[-1],
            self.ego_start_wp,
        )

        if self.pcla is None:
            self.pcla = PCLA(
                PCLA_AGENT,
                self.ego,
                PCLA_ROUTE,
                self.client,
            )

    def update(self):
        ego_action = self.pcla.get_action()
        self.ego.apply_control(ego_action)

        ego_speed = get_speed(self.ego)
        npc_speed = get_speed(self.npc)
        center_gap = self.ego.get_location().distance(
            self.npc.get_location()
        )

        if not self.triggered and center_gap <= CUTIN_TRIGGER_GAP:
            self.triggered = True
            self.trigger_time = self.elapsed
            print(
                f"[CUT-IN TRIGGER] t={self.elapsed:.2f}s, "
                f"center_gap={center_gap:.2f}m",
                flush=True,
            )

        if self.triggered:
            phase = smoothstep(
                (self.elapsed - self.trigger_time) / LANE_CHANGE_DURATION
            )
            self.progress = PARTIAL_PROGRESS * phase
        else:
            self.progress = 0.0

        self.path_s += npc_speed * FIXED_DT
        lookahead = clamp(4.0 + 0.45 * npc_speed, 4.0, 8.0)
        reference_s = NPC_INITIAL_GAP + self.path_s + lookahead
        ego_reference_wp = waypoint_ahead(
            self.ego_start_wp, reference_s
        )

        tf = ego_reference_wp.transform
        right = tf.get_right_vector()
        remaining_left_offset = (
            ego_reference_wp.lane_width * (1.0 - self.progress)
        )
        target = carla.Location(
            x=tf.location.x - right.x * remaining_left_offset,
            y=tf.location.y - right.y * remaining_left_offset,
            z=tf.location.z,
        )

        local_x, local_y = world_to_vehicle_local(
            self.npc.get_transform(), target
        )
        heading_error = math.atan2(local_y, max(local_x, 0.5))
        steer = clamp(1.65 * heading_error, -0.42, 0.42)

        target_speed = (
            NPC_AFTER_CUTIN_SPEED
            if self.triggered
            else NPC_CRUISE_SPEED
        )
        throttle, brake = speed_control(npc_speed, target_speed)
        self.npc.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=steer,
                hand_brake=False,
            )
        )

        lateral_offset = signed_lateral_offset(
            ego_reference_wp, self.npc
        )

        print(
            f"t={self.elapsed:6.2f}s, gap={center_gap:6.2f}m, "
            f"ego_v={ego_speed:5.2f}, npc_v={npc_speed:5.2f}, "
            f"cutin={self.progress:4.2f}, "
            f"npc_lat={lateral_offset:5.2f}m, "
            f"npc_steer={steer:+.2f}, "
            f"ego_thr={ego_action.throttle:.2f}, "
            f"ego_brake={ego_action.brake:.2f}",
            flush=True,
        )

        cutin_complete = (
            self.triggered
            and self.progress >= PARTIAL_PROGRESS * 0.98
        )

        if cutin_complete:
            if self.safe_after_complete_since is None:
                self.safe_after_complete_since = self.elapsed
            elif (
                self.elapsed - self.safe_after_complete_since
                >= POST_CUTIN_OBSERVATION
            ):
                print(
                    "Partial cut-in observation window completed.",
                    flush=True,
                )
                return py_trees.common.Status.SUCCESS
        else:
            self.safe_after_complete_since = None

        self.elapsed += FIXED_DT
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self.pcla is not None:
            try:
                self.pcla.cleanup()
            except RuntimeError:
                pass
            self.pcla = None


class InterfuserPartialCutIn(BasicScenario):
    """ScenarioRunner-native partial cut-in scenario."""

    timeout = SCENARIO_TIMEOUT

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        randomize=False,
        debug_mode=False,
        criteria_enable=True,
    ):
        self._world = world
        self._map = world.get_map()
        self._npc = None
        self._ego_start_wp = None

        super().__init__(
            "InterfuserPartialCutIn",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable,
        )

    def _setup_scenario_trigger(self, config):
        return None

    def _initialize_actors(self, config):
        ego = self.ego_vehicles[0]
        self._ego_start_wp = self._map.get_waypoint(
            ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if self._ego_start_wp is None:
            raise RuntimeError("Could not project ego onto a driving lane")

        ahead_wp = waypoint_ahead(
            self._ego_start_wp, NPC_INITIAL_GAP
        )
        npc_start_wp = valid_left_lane(ahead_wp)
        if npc_start_wp is None:
            raise RuntimeError(
                "Ego road does not have a same-direction left lane"
            )

        self._npc = CarlaDataProvider.request_new_actor(
            "vehicle.tesla.model3",
            npc_start_wp.transform,
            rolename="scenario",
            autopilot=False,
        )
        if self._npc is None:
            raise RuntimeError("Failed to spawn partial cut-in NPC")

        self._npc.set_autopilot(False)
        self.other_actors.append(self._npc)

    def _create_behavior(self):
        client = CarlaDataProvider.get_client()
        behavior = InterfuserPartialCutInBehavior(
            ego=self.ego_vehicles[0],
            npc=self._npc,
            ego_start_wp=self._ego_start_wp,
            client=client,
        )

        try:
            root = py_trees.composites.Sequence(
                name="InterfuserPartialCutInSequence",
                memory=True,
            )
        except TypeError:
            root = py_trees.composites.Sequence(
                "InterfuserPartialCutInSequence"
            )

        root.add_child(behavior)
        return root

    def _create_test_criteria(self):
        return [
            CollisionTest(
                self.ego_vehicles[0],
                terminate_on_failure=True,
                name="EgoCollisionTest",
            ),
            CollisionTest(
                self._npc,
                terminate_on_failure=True,
                name="NPCCollisionTest",
            ),
        ]

    def __del__(self):
        self.remove_all_actors()
