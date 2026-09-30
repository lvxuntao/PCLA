print("===== LOADED ABORTED CUTOUT SCENARIO V4 =====", flush=True)
#!/usr/bin/env python3
"""ScenarioRunner-native aborted cut-out + braking scenario for InterFuser.

NPC starts ahead of ego in the same lane, begins a smooth cut-out to the left,
aborts the lane change at a configurable progress, returns to the ego lane, and
returns to the ego lane, and starts a timed forced brake after a configurable delay. Ego is controlled by PCLA/InterFuser throughout.
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


PCLA_AGENT = os.environ.get("ACO_PCLA_AGENT", "if_if")
PCLA_ROUTE = os.environ.get(
    "ACO_PCLA_ROUTE",
    "/home/xlyu5/Research/PCLA/town06_aborted_cutout_route.xml",
)


def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None else float(default)


FIXED_DT = env_float("ACO_FIXED_DT", 0.05)
NPC_INITIAL_GAP = env_float("ACO_INITIAL_GAP", 12.0)
NPC_CRUISE_SPEED = env_float("ACO_NPC_SPEED", 3.5)
TRIGGER_GAP = env_float("ACO_TRIGGER_GAP", 9.0)
MAX_FOLLOW_TIME = env_float("ACO_MAX_FOLLOW_TIME", 8.0)
CUTOUT_DURATION = env_float("ACO_CUTOUT_DURATION", 1.0)
ABORT_PROGRESS = env_float("ACO_ABORT_PROGRESS", 0.55)
RETURN_DURATION = env_float("ACO_RETURN_DURATION", 0.6)
NPC_BRAKE = env_float("ACO_NPC_BRAKE", 0.8)
BRAKE_DELAY_AFTER_ABORT = env_float("ACO_BRAKE_DELAY_AFTER_ABORT", 0.35)
BRAKE_DURATION = env_float("ACO_BRAKE_DURATION", 1.5)
RETURN_LATERAL_TOLERANCE = env_float("ACO_RETURN_LATERAL_TOLERANCE", 0.25)
NPC_AFTER_SPEED = env_float("ACO_NPC_AFTER_SPEED", 0.5)
POST_OBSERVATION = env_float("ACO_POST_OBSERVATION", 6.0)
SCENARIO_TIMEOUT = env_float("ACO_TIMEOUT", 45.0)

ROUTE_LENGTH = 210.0
ROUTE_STEP = 8.0


def clamp(value, low, high):
    return max(low, min(high, value))


def get_speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def angle_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def choose_forward_waypoint(current_wp, reference_yaw, distance):
    candidates = [
        wp for wp in current_wp.next(distance)
        if wp.lane_type == carla.LaneType.Driving
    ]
    if not candidates:
        return None
    same_lane = [wp for wp in candidates if wp.lane_id == current_wp.lane_id]
    candidates = same_lane if same_lane else candidates
    return min(
        candidates,
        key=lambda wp: angle_diff_deg(wp.transform.rotation.yaw, reference_yaw),
    )


def waypoint_ahead(start_wp, distance, step=4.0):
    reference_yaw = start_wp.transform.rotation.yaw
    current_wp = start_wp
    travelled = 0.0
    while travelled < distance:
        increment = min(step, distance - travelled)
        next_wp = choose_forward_waypoint(current_wp, reference_yaw, increment)
        if next_wp is None:
            raise RuntimeError(f"Cannot find waypoint {distance:.1f} m ahead")
        current_wp = next_wp
        travelled += increment
    return current_wp


def write_route_xml(path, town, start_wp):
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id="0", town=town)
    current_wp = start_wp
    reference_yaw = start_wp.transform.rotation.yaw

    for _ in range(int(ROUTE_LENGTH // ROUTE_STEP) + 1):
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
        next_wp = choose_forward_waypoint(current_wp, reference_yaw, ROUTE_STEP)
        if next_wp is None or next_wp.is_junction:
            break
        current_wp = next_wp

    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def signed_lateral_offset(lane_wp, vehicle):
    right = lane_wp.transform.get_right_vector()
    delta_x = vehicle.get_location().x - lane_wp.transform.location.x
    delta_y = vehicle.get_location().y - lane_wp.transform.location.y
    return delta_x * right.x + delta_y * right.y


class InterfuserAbortedCutOutBehavior(AtomicBehavior):
    def __init__(self, ego, npc, ego_start_wp, client):
        super().__init__("InterfuserAbortedCutOutBehavior")
        self.ego = ego
        self.npc = npc
        self.ego_start_wp = ego_start_wp
        self.client = client
        self.pcla = None
        self.elapsed = 0.0
        self.triggered = False
        self.trigger_time = None
        self.abort_time = None
        self.brake_start_time = None
        self.brake_complete_time = None
        self.phase = "FOLLOW"
        self.lateral_progress = 0.0
        self.path_s = 0.0
        self.return_complete_time = None
        self.after_start_time = None

    def initialise(self):
        print(
            "[PARAMS] "
            f"initial_gap={NPC_INITIAL_GAP:.2f}, npc_speed={NPC_CRUISE_SPEED:.2f}, "
            f"trigger_gap={TRIGGER_GAP:.2f}, max_follow_time={MAX_FOLLOW_TIME:.2f}, "
            f"cutout_duration={CUTOUT_DURATION:.2f}, "
            f"abort_progress={ABORT_PROGRESS:.2f}, return_duration={RETURN_DURATION:.2f}, "
            f"npc_brake={NPC_BRAKE:.2f}, "
            f"brake_delay_after_abort={BRAKE_DELAY_AFTER_ABORT:.2f}, "
            f"brake_duration={BRAKE_DURATION:.2f}, "
            f"return_tolerance={RETURN_LATERAL_TOLERANCE:.2f}, "
            f"npc_after_speed={NPC_AFTER_SPEED:.2f}, "
            f"post_observation={POST_OBSERVATION:.2f}",
            flush=True,
        )
        self.elapsed = 0.0
        self.triggered = False
        self.trigger_time = None
        self.abort_time = None
        self.brake_start_time = None
        self.brake_complete_time = None
        self.phase = "FOLLOW"
        self.lateral_progress = 0.0
        self.path_s = 0.0
        self.return_complete_time = None
        self.after_start_time = None

        write_route_xml(
            PCLA_ROUTE,
            CarlaDataProvider.get_map().name.split("/")[-1],
            self.ego_start_wp,
        )
        if self.pcla is None:
            self.pcla = PCLA(PCLA_AGENT, self.ego, PCLA_ROUTE, self.client)

    def update(self):
        ego_action = self.pcla.get_action()
        self.ego.apply_control(ego_action)

        ego_speed = get_speed(self.ego)
        npc_speed = get_speed(self.npc)
        center_gap = self.ego.get_location().distance(self.npc.get_location())

        distance_trigger = center_gap <= TRIGGER_GAP
        time_trigger = MAX_FOLLOW_TIME > 0.0 and self.elapsed >= MAX_FOLLOW_TIME

        if not self.triggered and (distance_trigger or time_trigger):
            self.triggered = True
            self.trigger_time = self.elapsed
            trigger_reason = "gap" if distance_trigger else "max_follow_time"
            print(
                f"[CUT-OUT TRIGGER] t={self.elapsed:.2f}s, "
                f"center_gap={center_gap:.2f}m, reason={trigger_reason}",
                flush=True,
            )
            self.phase = "CUT_OUT"

        if self.phase == "CUT_OUT":
            normalized = (self.elapsed - self.trigger_time) / CUTOUT_DURATION
            self.lateral_progress = ABORT_PROGRESS * smoothstep(normalized)
            if normalized >= 1.0:
                self.phase = "RETURN_BRAKE"
                self.abort_time = self.elapsed
                self.lateral_progress = ABORT_PROGRESS
                print(
                    f"[ABORT] t={self.elapsed:.2f}s, "
                    f"progress={self.lateral_progress:.2f}, "
                    f"gap={center_gap:.2f}m",
                    flush=True,
                )
        elif self.phase == "RETURN_BRAKE":
            normalized = (self.elapsed - self.abort_time) / RETURN_DURATION
            # After RETURN_DURATION the target stays at the original-lane center,
            # but the phase does not finish until the physical vehicle is there.
            self.lateral_progress = ABORT_PROGRESS * (1.0 - smoothstep(normalized))
        elif self.phase == "BRAKE_IN_LANE":
            self.lateral_progress = 0.0
        elif self.phase == "AFTER":
            self.lateral_progress = 0.0

        self.path_s += npc_speed * FIXED_DT
        lookahead = clamp(4.0 + 0.45 * npc_speed, 4.0, 8.0)
        reference_s = NPC_INITIAL_GAP + self.path_s + lookahead
        lane_reference_wp = waypoint_ahead(self.ego_start_wp, reference_s)

        tf = lane_reference_wp.transform
        right = tf.get_right_vector()
        left_offset = lane_reference_wp.lane_width * self.lateral_progress
        target = carla.Location(
            x=tf.location.x - right.x * left_offset,
            y=tf.location.y - right.y * left_offset,
            z=tf.location.z,
        )

        local_x, local_y = world_to_vehicle_local(self.npc.get_transform(), target)
        heading_error = math.atan2(local_y, max(local_x, 0.5))
        steer_limit = 0.42 if self.phase == "CUT_OUT" else 0.55
        steer = clamp(1.75 * heading_error, -steer_limit, steer_limit)

        time_since_abort = (
            self.elapsed - self.abort_time if self.abort_time is not None else -1.0
        )

        # Start the forced-braking clock only when the delay has elapsed.
        # BRAKE_DURATION is measured from this actual start time, not from abort_time.
        if (
            self.abort_time is not None
            and self.brake_start_time is None
            and time_since_abort >= BRAKE_DELAY_AFTER_ABORT
        ):
            self.brake_start_time = self.elapsed
            print(
                f"[BRAKE START] t={self.elapsed:.2f}s, "
                f"npc_brake={NPC_BRAKE:.2f}, gap={center_gap:.2f}m",
                flush=True,
            )

        braking = (
            self.brake_start_time is not None
            and self.brake_complete_time is None
            and self.elapsed - self.brake_start_time < BRAKE_DURATION
        )

        if braking:
            throttle = 0.0
            brake = clamp(NPC_BRAKE, 0.0, 1.0)
        else:
            target_speed = (
                NPC_AFTER_SPEED if self.phase == "AFTER" else NPC_CRUISE_SPEED
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

        lateral_offset = signed_lateral_offset(lane_reference_wp, self.npc)

        # Do not claim that the return is complete merely because the requested
        # return duration elapsed. Wait until the physical actor is actually
        # close to the original lane center.
        if (
            self.phase == "RETURN_BRAKE"
            and self.abort_time is not None
            and self.elapsed - self.abort_time >= RETURN_DURATION
            and abs(lateral_offset) <= RETURN_LATERAL_TOLERANCE
        ):
            self.return_complete_time = self.elapsed
            self.lateral_progress = 0.0
            if self.brake_complete_time is not None:
                self.phase = "AFTER"
                self.after_start_time = self.elapsed
            else:
                self.phase = "BRAKE_IN_LANE"
            print(
                f"[RETURN COMPLETE] t={self.elapsed:.2f}s, "
                f"actual_lateral_offset={lateral_offset:.2f}m, "
                f"gap={center_gap:.2f}m",
                flush=True,
            )

        # Complete the forced braking exactly BRAKE_DURATION seconds after
        # BRAKE START. Do not enter AFTER until the physical return is also complete.
        if (
            self.brake_start_time is not None
            and self.brake_complete_time is None
            and self.elapsed - self.brake_start_time >= BRAKE_DURATION
        ):
            self.brake_complete_time = self.elapsed
            print(
                f"[BRAKE COMPLETE] t={self.elapsed:.2f}s, gap={center_gap:.2f}m",
                flush=True,
            )
            if self.return_complete_time is not None:
                self.phase = "AFTER"
                self.after_start_time = self.elapsed

        npc_yaw = self.npc.get_transform().rotation.yaw
        print(
            f"t={self.elapsed:6.2f}s, phase={self.phase:14s}, "
            f"gap={center_gap:6.2f}m, ego_v={ego_speed:5.2f}, "
            f"npc_v={npc_speed:5.2f}, progress={self.lateral_progress:4.2f}, "
            f"npc_lat={lateral_offset:6.2f}m, npc_yaw={npc_yaw:7.2f}, "
            f"npc_steer={steer:+.2f}, npc_brake={brake:.2f}, "
            f"ego_thr={ego_action.throttle:.2f}, ego_brake={ego_action.brake:.2f}",
            flush=True,
        )

        if (
            self.phase == "AFTER"
            and self.after_start_time is not None
            and self.elapsed - self.after_start_time >= POST_OBSERVATION
        ):
            print("Aborted cut-out observation window completed.", flush=True)
            return py_trees.common.Status.SUCCESS

        self.elapsed += FIXED_DT
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self.pcla is not None:
            try:
                self.pcla.cleanup()
            except RuntimeError:
                pass
            self.pcla = None


class InterfuserAbortedCutOut(BasicScenario):
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
            "InterfuserAbortedCutOut",
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

        npc_start_wp = waypoint_ahead(self._ego_start_wp, NPC_INITIAL_GAP)
        left_wp = npc_start_wp.get_left_lane()
        if (
            left_wp is None
            or left_wp.lane_type != carla.LaneType.Driving
        ):
            raise RuntimeError("The selected road has no driving lane on the left")

        spawn_tf = npc_start_wp.transform
        spawn_tf.location.z += 0.35
        self._npc = CarlaDataProvider.request_new_actor(
            "vehicle.tesla.model3",
            spawn_tf,
            rolename="scenario",
            autopilot=False,
        )
        if self._npc is None:
            raise RuntimeError("Failed to spawn aborted-cutout NPC")

        self._npc.set_autopilot(False)
        self.other_actors.append(self._npc)

    def _create_behavior(self):
        behavior = InterfuserAbortedCutOutBehavior(
            ego=self.ego_vehicles[0],
            npc=self._npc,
            ego_start_wp=self._ego_start_wp,
            client=CarlaDataProvider.get_client(),
        )
        try:
            root = py_trees.composites.Sequence(
                name="InterfuserAbortedCutOutSequence",
                memory=True,
            )
        except TypeError:
            root = py_trees.composites.Sequence("InterfuserAbortedCutOutSequence")
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
