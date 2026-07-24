'''
cd /home/xlyu5/Research/scenario_runner

python scenario_runner.py \
  --scenario FrontBrakeHandoff_1 \
  --additionalScenario /home/xlyu5/Research/PCLA/front_brake_handoff_scenario0.py \
  --configFile /home/xlyu5/Research/PCLA/front_brake_handoff.xml \
  --sync \
  --reloadWorld \
  --output
'''

import math
import time
import carla
import py_trees

from PCLA import PCLA

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    AtomicBehavior,
    KeepVelocity,
    StopVehicle
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest

# ============================================================
# Experiment config
# ============================================================

EGO_TARGET_SPEED_BEFORE_TRIGGER = 12.0
FRONT_TARGET_SPEED = 5.0

# center-to-center distance trigger, matching your current script
TRIGGER_GAP = 14.0

# front initial center-to-center distance ahead of ego
INITIAL_FRONT_GAP = 60.0

# PCLA / InterFuser
PCLA_AGENT = "if_if"
PCLA_ROUTE = "/home/xlyu5/Research/PCLA/sample_route.xml"

# Scenario timeout
SCENARIO_TIMEOUT = 60.0


# ============================================================
# Helper functions
# ============================================================

def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def center_distance(ego_vehicle, front_vehicle):
    return ego_vehicle.get_location().distance(front_vehicle.get_location())


def spawn_actor_ahead_of_ego(ego_vehicle, model, gap):
    ego_tf = ego_vehicle.get_transform()
    forward = ego_tf.get_forward_vector()

    front_location = ego_tf.location + carla.Location(
        x=forward.x * gap,
        y=forward.y * gap,
        z=0.5
    )

    front_transform = carla.Transform(
        front_location,
        ego_tf.rotation
    )

    actor = CarlaDataProvider.request_new_actor(
        model,
        front_transform,
        rolename="scenario",
        autopilot=False
    )

    if actor is None:
        raise RuntimeError("Failed to spawn front vehicle.")

    actor.set_autopilot(False)

    return actor


# ============================================================
# Main behavior
# ============================================================

class EgoExternalThenInterFuserFrontBrake(AtomicBehavior):
    """
    Before trigger:
        ego is externally controlled at EGO_TARGET_SPEED_BEFORE_TRIGGER.
        front is kept at FRONT_TARGET_SPEED by ScenarioRunner KeepVelocity.

    Trigger:
        center distance <= TRIGGER_GAP.

    After trigger:
        front uses ScenarioRunner StopVehicle.
        ego is immediately handed over to InterFuser/PCLA.
    """

    def __init__(
        self,
        ego_vehicle,
        front_vehicle,
        carla_map,
        client,
        name="EgoExternalThenInterFuserFrontBrake"
    ):
        super().__init__(name)

        self.ego_vehicle = ego_vehicle
        self.front_vehicle = front_vehicle
        self.carla_map = carla_map
        self.client = client

        self.brake_triggered = False
        self.finished = False

        self.pcla = None

        self.keep_velocity = KeepVelocity(
            self.front_vehicle,
            FRONT_TARGET_SPEED,
            force_speed=False,
            duration=float("inf"),
            distance=float("inf")
        )

        self.ego_keep_velocity = KeepVelocity(
            self.ego_vehicle,
            EGO_TARGET_SPEED_BEFORE_TRIGGER,
            force_speed=False,
            duration=float("inf"),
            distance=float("inf")
        )

        self.stop_vehicle = StopVehicle(
            self.front_vehicle,
            brake_value=1.0
        )

        self.last_mode = "INIT"
        self.last_distance = None
        self.last_ego_control = carla.VehicleControl()
        self.last_front_control = carla.VehicleControl()

    def initialise(self):
        self.keep_velocity.initialise()
        self.ego_keep_velocity.initialise()
        self.stop_vehicle.initialise()

        if self.pcla is None:
            self.pcla = PCLA(PCLA_AGENT, self.ego_vehicle, PCLA_ROUTE, self.client)

        self.brake_triggered = False
        self.finished = False
        self.last_mode = "INIT"

    def update(self):
        distance = center_distance(self.ego_vehicle, self.front_vehicle)
        ego_speed = get_speed(self.ego_vehicle)
        front_speed = get_speed(self.front_vehicle)

        self.last_distance = distance

        # Latch trigger
        self.brake_triggered = self.brake_triggered or (distance <= TRIGGER_GAP)

        # Always warm up InterFuser/PCLA.
        # Before trigger we do not apply it.
        # After trigger we immediately apply it.
        pcla_action = self.pcla.get_action()
        time.sleep(0.05)
        
        if self.brake_triggered:
            # Front full brake
            self.stop_vehicle.update()

            # Ego handed over to InterFuser
            ego_control = pcla_action
            self.ego_vehicle.apply_control(ego_control)

            self.last_mode = "TRIGGERED_INTERFUSER"
            self.last_ego_control = ego_control

        else:
            # Before trigger, both front and ego are controlled by ScenarioRunner KeepVelocity.
            # Both use force_speed=False, so they use vehicle controls instead of directly forcing velocity.
            self.keep_velocity.update()
            self.ego_keep_velocity.update()

            try:
                ego_control = self.ego_vehicle.get_control()
            except RuntimeError:
                ego_control = carla.VehicleControl()

            self.last_mode = "PRE_TRIGGER_KEEPVELOCITY"
            self.last_ego_control = ego_control

        try:
            self.last_front_control = self.front_vehicle.get_control()
        except RuntimeError:
            self.last_front_control = carla.VehicleControl()

        print(
            f"distance={distance:.2f}m, "
            f"ego_v={ego_speed:.2f}m/s, "
            f"front_v={front_speed:.2f}m/s, "
            f"triggered={self.brake_triggered}, "
            f"mode={self.last_mode}, "
            f"ego_control=(thr={self.last_ego_control.throttle:.2f}, "
            f"brake={self.last_ego_control.brake:.2f}, "
            f"steer={self.last_ego_control.steer:.2f}), "
            f"front_control=(thr={self.last_front_control.throttle:.2f}, "
            f"brake={self.last_front_control.brake:.2f}, "
            f"steer={self.last_front_control.steer:.2f})"
        )

        if self.brake_triggered:
            if ego_speed < 0.1 and front_speed < 0.1:
                print("Both vehicles stopped safely.")
                self.finished = True
                return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self.pcla is not None:
            try:
                self.pcla.cleanup()
            except RuntimeError:
                pass
            self.pcla = None


# ============================================================
# Scenario class discovered by ScenarioRunner
# ============================================================

class FrontBrakeHandoff(BasicScenario):
    """
    Custom ScenarioRunner scenario.

    This class is loaded by scenario_runner.py via --additionalScenario.
    """

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        randomize=False,
        debug_mode=False,
        criteria_enable=True
    ):
        self._world = world
        self._map = world.get_map()
        self._front_vehicle = None

        super().__init__(
            "FrontBrakeHandoff",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

    def _setup_scenario_trigger(self, config):
        return None

    def _initialize_actors(self, config):
        ego_vehicle = self.ego_vehicles[0]

        self._front_vehicle = spawn_actor_ahead_of_ego(
            ego_vehicle=ego_vehicle,
            model="vehicle.tesla.model3",
            gap=INITIAL_FRONT_GAP
        )

        self.other_actors.append(self._front_vehicle)

    def _create_behavior(self):
        ego_vehicle = self.ego_vehicles[0]

        try:
            client = CarlaDataProvider.get_client()
        except AttributeError:
            client = carla.Client("localhost", 2000)
            client.set_timeout(10.0)

        behavior = EgoExternalThenInterFuserFrontBrake(
            ego_vehicle=ego_vehicle,
            front_vehicle=self._front_vehicle,
            carla_map=self._map,
            client=client
        )

        try:
            root = py_trees.composites.Sequence(
                name="FrontBrakeHandoffSequence",
                memory=True
            )
        except TypeError:
            root = py_trees.composites.Sequence("FrontBrakeHandoffSequence")

        root.add_child(behavior)

        print(
            "returning behavior tree with children:",
            [child.name for child in root.children],
            flush=True
        )

        return root

    def _create_test_criteria(self):
        criteria = [
            CollisionTest(
                self.ego_vehicles[0],
                name="EgoCollisionTest"
            ),
            CollisionTest(
                self._front_vehicle,
                name="FrontCollisionTest"
            )
        ]
        return criteria

    def __del__(self):
        self.remove_all_actors()