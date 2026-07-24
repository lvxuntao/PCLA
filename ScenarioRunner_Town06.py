import math
import time
import xml.etree.ElementTree as ET

import carla
import py_trees

from PCLA import PCLA
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior


TOWN = "Town06"
FIXED_DT = 0.05

EGO_TARGET_ROUTE = "./town06_straight_route.xml"

INITIAL_FRONT_GAP = 30.0
FRONT_TARGET_SPEED = 5.0
BRAKE_TRIGGER_GAP = 12.0

ROUTE_LENGTH = 220.0
ROUTE_STEP = 10.0


def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def angle_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def safe_destroy(actor):
    if actor is None:
        return

    try:
        if actor.is_alive:
            actor.destroy()
    except RuntimeError:
        pass


def choose_forward_waypoint(current_wp, reference_yaw, distance):
    """
    Choose a forward waypoint that best preserves the current lane direction.
    This avoids accidentally taking a ramp/branch.
    """
    next_wps = current_wp.next(distance)

    if len(next_wps) == 0:
        return None

    next_wps = [
        wp for wp in next_wps
        if wp.lane_type == carla.LaneType.Driving
    ]

    if len(next_wps) == 0:
        return None

    same_lane_wps = [
        wp for wp in next_wps
        if wp.lane_id == current_wp.lane_id
    ]

    candidates = same_lane_wps if len(same_lane_wps) > 0 else next_wps

    return min(
        candidates,
        key=lambda wp: angle_diff_deg(wp.transform.rotation.yaw, reference_yaw)
    )


def evaluate_straight_length(start_wp, step=5.0, max_length=250.0, max_yaw_error=8.0):
    """
    Estimate how long the lane ahead stays roughly straight and non-junction.
    """
    if start_wp is None or start_wp.is_junction:
        return 0.0

    reference_yaw = start_wp.transform.rotation.yaw
    current_wp = start_wp
    total = 0.0

    while total < max_length:
        next_wp = choose_forward_waypoint(current_wp, reference_yaw, step)

        if next_wp is None:
            break

        if next_wp.is_junction:
            break

        yaw_error = angle_diff_deg(next_wp.transform.rotation.yaw, reference_yaw)
        if yaw_error > max_yaw_error:
            break

        total += step
        current_wp = next_wp

    return total


def find_town06_straight_start(carla_map, min_length=120.0):
    """
    Search all spawn points and pick the one with the longest straight lane ahead.
    """
    spawn_points = carla_map.get_spawn_points()

    best_wp = None
    best_length = -1.0
    best_index = -1

    for i, sp in enumerate(spawn_points):
        wp = carla_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if wp is None:
            continue

        length = evaluate_straight_length(
            wp,
            step=5.0,
            max_length=250.0,
            max_yaw_error=8.0
        )

        if length > best_length:
            best_length = length
            best_wp = wp
            best_index = i

    if best_wp is None or best_length < min_length:
        raise RuntimeError(
            f"Could not find a sufficiently long straight road in {TOWN}. "
            f"Best length = {best_length:.1f} m"
        )

    loc = best_wp.transform.location
    yaw = best_wp.transform.rotation.yaw

    print(
        f"Selected Town06 straight spawn index={best_index}, "
        f"straight_length≈{best_length:.1f}m, "
        f"road_id={best_wp.road_id}, lane_id={best_wp.lane_id}, "
        f"x={loc.x:.2f}, y={loc.y:.2f}, yaw={yaw:.1f}"
    )

    return best_wp


def waypoint_ahead(start_wp, distance, step=5.0):
    """
    Move forward along the same straight-ish lane.
    """
    reference_yaw = start_wp.transform.rotation.yaw
    current_wp = start_wp
    traveled = 0.0

    while traveled < distance:
        remaining = distance - traveled
        d = min(step, remaining)

        next_wp = choose_forward_waypoint(current_wp, reference_yaw, d)

        if next_wp is None:
            raise RuntimeError(f"Cannot find waypoint {distance:.1f}m ahead.")

        current_wp = next_wp
        traveled += d

    return current_wp


def spawn_vehicle_at_waypoint(world, vehicle_bp, wp, max_tries=10):
    """
    Spawn a vehicle on a CARLA waypoint.
    If the exact point is occupied, move slightly forward and retry.
    """
    current_wp = wp

    for _ in range(max_tries):
        transform = current_wp.transform
        transform.location.z += 0.5

        actor = world.try_spawn_actor(vehicle_bp, transform)
        if actor is not None:
            return actor

        current_wp = waypoint_ahead(current_wp, 3.0, step=3.0)

    raise RuntimeError("Failed to spawn vehicle after multiple attempts.")


def spawn_front_vehicle(world, ego_start_wp, vehicle_bp, gap=30.0):
    front_wp = waypoint_ahead(ego_start_wp, gap, step=5.0)
    front_vehicle = spawn_vehicle_at_waypoint(world, vehicle_bp, front_wp)
    return front_vehicle


def write_route_xml(route_path, town, start_wp, route_length=220.0, step=10.0):
    """
    Generate a simple Leaderboard-style route XML along the selected straight lane.

    This is important because PCLA/InterFuser needs a route compatible with Town06,
    not the old Town02 sample_route.xml.
    """
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", id="0", town=town)

    current_wp = start_wp
    reference_yaw = start_wp.transform.rotation.yaw

    num_points = int(route_length // step) + 1

    for i in range(num_points):
        tf = current_wp.transform
        loc = tf.location
        rot = tf.rotation

        ET.SubElement(
            route,
            "waypoint",
            x=f"{loc.x:.6f}",
            y=f"{loc.y:.6f}",
            z=f"{loc.z:.6f}",
            pitch=f"{rot.pitch:.6f}",
            yaw=f"{rot.yaw:.6f}",
            roll=f"{rot.roll:.6f}",
        )

        next_wp = choose_forward_waypoint(current_wp, reference_yaw, step)
        if next_wp is None:
            break

        if next_wp.is_junction:
            break

        current_wp = next_wp

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(route_path, encoding="utf-8", xml_declaration=True)

    print(f"Generated route XML: {route_path}")


def lane_follow_steer(vehicle, carla_map, lookahead=5.0):
    """
    A simple waypoint-based steering controller for the front vehicle.
    This keeps the front car roughly following its lane before braking.
    """
    loc = vehicle.get_location()
    wp = carla_map.get_waypoint(
        loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if wp is None:
        return 0.0

    reference_yaw = wp.transform.rotation.yaw
    next_wp = choose_forward_waypoint(wp, reference_yaw, lookahead)

    if next_wp is None:
        return 0.0

    target = next_wp.transform.location
    transform = vehicle.get_transform()

    dx = target.x - transform.location.x
    dy = target.y - transform.location.y

    yaw = math.radians(transform.rotation.yaw)

    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

    angle = math.atan2(local_y, max(local_x, 1e-3))

    return clamp(1.2 * angle, -0.4, 0.4)


class FrontFullBrakeBehavior(AtomicBehavior):
    """
    ScenarioRunner-style custom atomic behavior.

    Before trigger:
        front follows lane and tries to reach target_speed through throttle/brake.

    After trigger:
        front applies full brake = 1.0.

    Trigger condition:
        ego-front distance <= trigger_gap.
    """

    def __init__(
        self,
        front_vehicle,
        ego_vehicle,
        carla_map,
        target_speed=5.0,
        trigger_gap=12.0,
        name="FrontFullBrakeBehavior"
    ):
        super().__init__(name)
        self.front_vehicle = front_vehicle
        self.ego_vehicle = ego_vehicle
        self.carla_map = carla_map
        self.target_speed = target_speed
        self.trigger_gap = trigger_gap
        self.brake_triggered = False

    def update(self):
        distance = self.ego_vehicle.get_location().distance(
            self.front_vehicle.get_location()
        )

        front_speed = get_speed(self.front_vehicle)

        if distance <= self.trigger_gap:
            self.brake_triggered = True

        if self.brake_triggered:
            control = carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                steer=lane_follow_steer(self.front_vehicle, self.carla_map)
            )
        else:
            error = self.target_speed - front_speed

            if error > 0.5:
                throttle = 0.35
                brake = 0.0
            elif error > 0.1:
                throttle = 0.15
                brake = 0.0
            elif error < -0.3:
                throttle = 0.0
                brake = 0.2
            else:
                throttle = 0.0
                brake = 0.0

            control = carla.VehicleControl(
                throttle=throttle,
                brake=brake,
                steer=lane_follow_steer(self.front_vehicle, self.carla_map)
            )

        self.front_vehicle.apply_control(control)

        return py_trees.common.Status.RUNNING


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    world = client.load_world(TOWN)
    traffic_manager = client.get_trafficmanager(8000)

    original_sync = None
    original_delta = None

    ego_vehicle = None
    front_vehicle = None
    pcla = None

    try:
        settings = world.get_settings()
        original_sync = settings.synchronous_mode
        original_delta = settings.fixed_delta_seconds

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)
        world.tick()

        carla_map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        ego_bp = blueprint_library.filter("model3")[0]
        ego_bp.set_attribute("role_name", "hero")

        front_bp = blueprint_library.filter("model3")[0]
        front_bp.set_attribute("role_name", "scenario")

        # =========================
        # Find a long straight segment in Town06
        # =========================
        ego_start_wp = find_town06_straight_start(carla_map, min_length=120.0)

        # Generate a Town06 route that matches this straight segment.
        write_route_xml(
            route_path=EGO_TARGET_ROUTE,
            town=TOWN,
            start_wp=ego_start_wp,
            route_length=ROUTE_LENGTH,
            step=ROUTE_STEP
        )

        # =========================
        # Ego vehicle
        # =========================
        ego_vehicle = spawn_vehicle_at_waypoint(world, ego_bp, ego_start_wp)
        ego_vehicle.set_autopilot(False)
        world.tick()

        # =========================
        # Front vehicle
        # =========================
        front_vehicle = spawn_front_vehicle(
            world,
            ego_start_wp,
            front_bp,
            gap=INITIAL_FRONT_GAP
        )
        front_vehicle.set_autopilot(False)
        world.tick()

        # Move spectator above the experiment.
        spectator = world.get_spectator()
        ego_tf = ego_vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(
                ego_tf.location + carla.Location(z=55.0),
                carla.Rotation(pitch=-90.0, yaw=ego_tf.rotation.yaw)
            )
        )

        # =========================
        # InterFuser / PCLA ego controller
        # =========================
        agent = "if_if"
        route = EGO_TARGET_ROUTE

        pcla = PCLA(agent, ego_vehicle, route, client)

        # =========================
        # ScenarioRunner-style front behavior
        # =========================
        front_behavior = FrontFullBrakeBehavior(
            front_vehicle=front_vehicle,
            ego_vehicle=ego_vehicle,
            carla_map=carla_map,
            target_speed=FRONT_TARGET_SPEED,
            trigger_gap=BRAKE_TRIGGER_GAP
        )

        print("\nRunning Town06 straight-road front full brake scenario")
        print("Town: Town06")
        print("Ego: InterFuser/PCLA")
        print("Front: ScenarioRunner AtomicBehavior")
        print(f"Initial front gap: {INITIAL_FRONT_GAP:.1f} m")
        print(f"Front target speed: {FRONT_TARGET_SPEED:.1f} m/s")
        print(f"Brake trigger gap: {BRAKE_TRIGGER_GAP:.1f} m")
        print(f"Route: {EGO_TARGET_ROUTE}\n")

        step = 0

        while True:
            timestamp = step * FIXED_DT

            # =========================
            # Ego is controlled only by InterFuser/PCLA
            # =========================
            ego_action = pcla.get_action()
            ego_vehicle.apply_control(ego_action)

            # =========================
            # Front is controlled by ScenarioRunner-style behavior
            # =========================
            front_behavior.update()

            distance = ego_vehicle.get_location().distance(
                front_vehicle.get_location()
            )

            ego_speed = get_speed(ego_vehicle)
            front_speed = get_speed(front_vehicle)

            print(
                f"t={timestamp:.2f}s, "
                f"distance={distance:.2f}m, "
                f"ego_v={ego_speed:.2f}m/s, "
                f"front_v={front_speed:.2f}m/s, "
                f"triggered={front_behavior.brake_triggered}, "
                f"ego_control=(thr={ego_action.throttle:.2f}, "
                f"brake={ego_action.brake:.2f}, "
                f"steer={ego_action.steer:.2f})"
            )

            if front_behavior.brake_triggered:
                if distance < 2.0:
                    print("Crash detected by distance threshold.")
                    break

                if ego_speed < 0.1 and front_speed < 0.1:
                    print("Both vehicles stopped safely.")
                    break

            world.tick()
            step += 1

    finally:
        print("\nCleaning up...")

        if pcla is not None:
            try:
                pcla.cleanup()
            except RuntimeError:
                pass

        safe_destroy(ego_vehicle)
        safe_destroy(front_vehicle)

        try:
            settings = world.get_settings()
            settings.synchronous_mode = original_sync if original_sync is not None else False
            settings.fixed_delta_seconds = original_delta
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass

        time.sleep(0.5)
        print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")