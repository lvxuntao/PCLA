import carla
import time
import math


# ============================================================
# Basic utilities
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def get_speed(vehicle):
    """
    Total vehicle speed magnitude [m/s].
    """
    v = vehicle.get_velocity()

    return math.sqrt(
        v.x ** 2
        + v.y ** 2
        + v.z ** 2
    )


def get_longitudinal_acceleration(vehicle):
    """
    CARLA acceleration projected onto the vehicle's
    current forward direction.
    """

    acceleration = vehicle.get_acceleration()
    tf = vehicle.get_transform()

    theta = math.radians(tf.rotation.yaw)

    fx = math.cos(theta)
    fy = math.sin(theta)

    return (
        acceleration.x * fx
        + acceleration.y * fy
    )


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def smoothstep(r):
    """
    Cubic lane-change profile:

        s(r) = 3 r^2 - 2 r^3
    """
    r = clamp(r, 0.0, 1.0)

    return (
        3.0 * r ** 2
        - 2.0 * r ** 3
    )


# ============================================================
# Waypoint utilities
# ============================================================

def get_same_lane_next(wp, distance):
    """
    Advance along the same road and lane.
    """

    candidates = wp.next(distance)

    if not candidates:
        return None

    for candidate in candidates:

        if (
            candidate.road_id == wp.road_id
            and
            candidate.lane_id == wp.lane_id
        ):
            return candidate

    return None


def find_adjacent_driving_lane(waypoint):
    """
    Find a same-direction adjacent driving lane.

    Returns:
        adjacent_waypoint, side
    """

    forward = waypoint.transform.get_forward_vector()

    candidates = [
        ("left", waypoint.get_left_lane()),
        ("right", waypoint.get_right_lane())
    ]

    for side, candidate in candidates:

        if candidate is None:
            continue

        if candidate.lane_type != carla.LaneType.Driving:
            continue

        other_forward = (
            candidate.transform.get_forward_vector()
        )

        dot = (
            forward.x * other_forward.x
            + forward.y * other_forward.y
            + forward.z * other_forward.z
        )

        if dot > 0.9:
            return candidate, side

    return None, None


def get_adjacent_lane_on_side(waypoint, side):

    if side == "left":
        candidate = waypoint.get_left_lane()
    else:
        candidate = waypoint.get_right_lane()

    if candidate is None:
        return None

    if candidate.lane_type != carla.LaneType.Driving:
        return None

    f1 = waypoint.transform.get_forward_vector()
    f2 = candidate.transform.get_forward_vector()

    dot = (
        f1.x * f2.x
        + f1.y * f2.y
        + f1.z * f2.z
    )

    if dot <= 0.9:
        return None

    return candidate


# ============================================================
# Find suitable two-lane road
# ============================================================

def find_suitable_two_lane_segment(
    carla_map,
    spawn_points,
    npc_initial_distance,
    required_length,
    check_step=2.0,
    max_heading_change=6.0
):
    """
    Find a long, approximately straight, same-direction
    two-lane road segment.
    """

    print("\nSearching for suitable two-lane straight road...")

    for spawn_index, spawn_tf in enumerate(spawn_points):

        start_wp = carla_map.get_waypoint(
            spawn_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if start_wp is None:
            continue

        if start_wp.is_junction:
            continue

        adjacent_start, side = find_adjacent_driving_lane(
            start_wp
        )

        if adjacent_start is None:
            continue

        initial_yaw = start_wp.transform.rotation.yaw

        ego_wp = start_wp
        npc_wp = adjacent_start

        traveled = 0.0
        valid = True

        ego_at_npc_start = None
        npc_at_npc_start = None

        while traveled < required_length:

            if ego_wp.is_junction or npc_wp.is_junction:
                valid = False
                break

            heading_change = abs(
                angle_diff_deg(
                    ego_wp.transform.rotation.yaw,
                    initial_yaw
                )
            )

            if heading_change > max_heading_change:
                valid = False
                break

            expected_adjacent = get_adjacent_lane_on_side(
                ego_wp,
                side
            )

            if expected_adjacent is None:
                valid = False
                break

            if expected_adjacent.lane_id != npc_wp.lane_id:
                valid = False
                break

            if (
                ego_at_npc_start is None
                and traveled >= npc_initial_distance
            ):
                ego_at_npc_start = ego_wp
                npc_at_npc_start = npc_wp

            next_ego = get_same_lane_next(
                ego_wp,
                check_step
            )

            next_npc = get_same_lane_next(
                npc_wp,
                check_step
            )

            if next_ego is None or next_npc is None:
                valid = False
                break

            ego_wp = next_ego
            npc_wp = next_npc

            traveled += check_step

        if not valid:
            continue

        if (
            ego_at_npc_start is None
            or npc_at_npc_start is None
        ):
            continue

        print("\nFound suitable road:")
        print(f"  spawn index   : {spawn_index}")
        print(f"  road id       : {start_wp.road_id}")
        print(f"  ego lane id   : {start_wp.lane_id}")
        print(f"  NPC lane id   : {adjacent_start.lane_id}")
        print(f"  adjacent side : {side}")
        print(f"  checked       : {traveled:.1f} m")

        return {
            "ego_start_wp": start_wp,
            "npc_source_wp": npc_at_npc_start,
            "npc_target_wp": ego_at_npc_start,
            "adjacent_side": side
        }

    return None


# ============================================================
# Bounding-box geometry
# ============================================================

def get_vehicle_corners(vehicle):
    """
    Return four footprint corners of CARLA bounding box
    in world coordinates.
    """

    tf = vehicle.get_transform()
    bb = vehicle.bounding_box

    center = tf.transform(
        bb.location
    )

    theta = math.radians(
        tf.rotation.yaw
    )

    c = math.cos(theta)
    s = math.sin(theta)

    half_length = bb.extent.x
    half_width = bb.extent.y

    local_corners = [
        (+half_length, +half_width),
        (+half_length, -half_width),
        (-half_length, +half_width),
        (-half_length, -half_width)
    ]

    corners = []

    for lx, ly in local_corners:

        wx = (
            center.x
            + lx * c
            - ly * s
        )

        wy = (
            center.y
            + lx * s
            + ly * c
        )

        corners.append(
            (wx, wy)
        )

    return corners


# ============================================================
# Ego forward corridor detection
# ============================================================

def npc_enters_ego_corridor(ego, npc):
    """
    ACC trigger condition:

    At least one NPC bounding-box corner is

        1. ahead of ego front
        2. inside ego's lateral swept corridor
    """

    ego_tf = ego.get_transform()

    ego_center = ego_tf.transform(
        ego.bounding_box.location
    )

    ego_theta = math.radians(
        ego_tf.rotation.yaw
    )

    fx = math.cos(ego_theta)
    fy = math.sin(ego_theta)

    nx = -math.sin(ego_theta)
    ny = math.cos(ego_theta)

    ego_half_length = (
        ego.bounding_box.extent.x
    )

    ego_half_width = (
        ego.bounding_box.extent.y
    )

    npc_corners = get_vehicle_corners(
        npc
    )

    for px, py in npc_corners:

        dx = px - ego_center.x
        dy = py - ego_center.y

        longitudinal = (
            dx * fx
            + dy * fy
        )

        lateral = (
            dx * nx
            + dy * ny
        )

        if (
            longitudinal >= ego_half_length
            and
            abs(lateral) <= ego_half_width
        ):
            return True

    return False


# ============================================================
# PI Controller
# ============================================================

class PIController:

    def __init__(
        self,
        kp,
        ki,
        dt,
        integral_min=-10.0,
        integral_max=10.0
    ):

        self.kp = kp
        self.ki = ki
        self.dt = dt

        self.integral = 0.0

        self.integral_min = integral_min
        self.integral_max = integral_max

    def reset(self):

        self.integral = 0.0

    def step(self, error):

        self.integral += (
            error * self.dt
        )

        self.integral = clamp(
            self.integral,
            self.integral_min,
            self.integral_max
        )

        return (
            self.kp * error
            + self.ki * self.integral
        )


# ============================================================
# Ego controller
# ============================================================

class EgoController:

    def __init__(
        self,
        vehicle,
        dt,

        v_ref=8.0,
        Tc=1.0,

        kd=0.30,
        kv=0.80,
        time_headway=1.2,
        eta=3.0,

        throttle_kp=0.35,
        throttle_ki=0.10,

        brake_kp=0.25,
        brake_ki=0.08
    ):

        self.vehicle = vehicle
        self.dt = dt

        # Cruise
        self.v_ref = v_ref
        self.Tc = Tc

        # ACC
        self.kd = kd
        self.kv = kv
        self.time_headway = time_headway
        self.eta = eta

        self.acc_active = False

        self.throttle_pi = PIController(
            throttle_kp,
            throttle_ki,
            dt
        )

        self.brake_pi = PIController(
            brake_kp,
            brake_ki,
            dt
        )

    # ========================================================
    # Longitudinal gap
    # ========================================================

    def compute_longitudinal_gap(
        self,
        npc_x,
        npc_y,
        npc_theta,
        npc_vehicle
    ):
        """
        d = ego front -> NPC rear
        measured along ego heading.

        NPC orientation is explicitly considered.
        """

        ego_tf = self.vehicle.get_transform()

        ego_center = ego_tf.transform(
            self.vehicle.bounding_box.location
        )

        ego_theta = math.radians(
            ego_tf.rotation.yaw
        )

        fx = math.cos(ego_theta)
        fy = math.sin(ego_theta)

        dx = npc_x - ego_center.x
        dy = npc_y - ego_center.y

        center_projection = (
            dx * fx
            + dy * fy
        )

        ego_half_length = (
            self.vehicle.bounding_box.extent.x
        )

        npc_half_length = (
            npc_vehicle.bounding_box.extent.x
        )

        npc_half_width = (
            npc_vehicle.bounding_box.extent.y
        )

        delta_theta = (
            npc_theta - ego_theta
        )

        npc_projected_extent = (
            npc_half_length
            * abs(
                math.cos(delta_theta)
            )
            +
            npc_half_width
            * abs(
                math.sin(delta_theta)
            )
        )

        d = (
            center_projection
            - ego_half_length
            - npc_projected_extent
        )

        return d

    # ========================================================
    # Main controller
    # ========================================================

    def run_step(
        self,
        v,
        a,
        vL,
        npc_theta,
        npc_x,
        npc_y,
        npc_vehicle
    ):

        intrusion = npc_enters_ego_corridor(
            self.vehicle,
            npc_vehicle
        )

        # ====================================================
        # CRUISE -> ACC
        # ====================================================

        if (
            intrusion
            and not self.acc_active
        ):

            self.acc_active = True

            self.throttle_pi.reset()
            self.brake_pi.reset()

            print()
            print(
                ">>> NPC bounding box entered "
                "ego forward corridor."
            )

            print(
                ">>> Controller switches: "
                "CRUISE -> ACC"
            )
            print()

        # ====================================================
        # Outer controller
        # ====================================================

        if not self.acc_active:

            mode = "CRUISE"

            d = None
            desired_gap = None

            # -----------------------------------------------
            # Cruise:
            #
            # a_hat = 1/Tc * (v_ref - v)
            # -----------------------------------------------

            a_hat = (
                self.v_ref - v
            ) / self.Tc

        else:

            mode = "ACC"

            d = (
                self.compute_longitudinal_gap(
                    npc_x=npc_x,
                    npc_y=npc_y,
                    npc_theta=npc_theta,
                    npc_vehicle=npc_vehicle
                )
            )

            desired_gap = (
                self.time_headway * v
                + self.eta
            )

            # -----------------------------------------------
            # ACC:
            #
            # a_hat =
            # kd(d - th*v - eta)
            # +
            # kv(vL - v)
            # -----------------------------------------------

            a_hat = (
                self.kd
                * (
                    d
                    - self.time_headway * v
                    - self.eta
                )
                +
                self.kv
                * (
                    vL - v
                )
            )

        # ====================================================
        # Actuator controller
        # ====================================================

        throttle = 0.0
        brake = 0.0

        if a_hat >= 0.0:

            error = (
                a_hat - a
            )

            throttle = clamp(
                self.throttle_pi.step(
                    error
                ),
                0.0,
                1.0
            )

            self.brake_pi.reset()

        else:

            error = (
                a - a_hat
            )

            brake = clamp(
                self.brake_pi.step(
                    error
                ),
                0.0,
                1.0
            )

            self.throttle_pi.reset()

        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=0.0
        )

        debug = {
            "mode": mode,
            "v": v,
            "vL": vL,
            "a": a,
            "a_hat": a_hat,
            "d": d,
            "desired_gap": desired_gap,
            "intrusion": intrusion,
            "throttle": throttle,
            "brake": brake
        }

        return control, debug


# ============================================================
# Collision monitor
# ============================================================

class CollisionMonitor:

    def __init__(
        self,
        world,
        vehicle
    ):

        self.collided = False
        self.other_actor = None

        sensor_bp = (
            world
            .get_blueprint_library()
            .find(
                "sensor.other.collision"
            )
        )

        self.sensor = world.spawn_actor(
            sensor_bp,
            carla.Transform(),
            attach_to=vehicle
        )

        self.sensor.listen(
            self._callback
        )

    def _callback(self, event):

        self.collided = True
        self.other_actor = event.other_actor

        print()
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        print(
            f"COLLISION WITH: "
            f"{event.other_actor.type_id}"
        )

        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        print()

    def destroy(self):

        if self.sensor is not None:

            self.sensor.stop()
            self.sensor.destroy()

            self.sensor = None


# ============================================================
# Visualization
# ============================================================

def draw_vehicle_bbox(
    world,
    vehicle,
    color,
    thickness,
    life_time
):
    """
    Draw 3D bounding box around vehicle.
    """

    vertices = (
        vehicle.bounding_box
        .get_world_vertices(
            vehicle.get_transform()
        )
    )

    edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),

        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),

        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7)
    ]

    for i, j in edges:

        world.debug.draw_line(
            vertices[i],
            vertices[j],

            thickness=thickness,

            color=color,

            life_time=life_time
        )


def draw_npc_corners(
    world,
    npc,
    life_time
):
    """
    Draw P0-P3 corners used for intrusion detection.
    """

    corners = get_vehicle_corners(
        npc
    )

    color = carla.Color(
        255,
        0,
        255
    )

    z = (
        npc.get_location().z
        + 0.7
    )

    for index, (x, y) in enumerate(
        corners
    ):

        p = carla.Location(
            x=x,
            y=y,
            z=z
        )

        world.debug.draw_point(
            p,
            size=0.13,
            color=color,
            life_time=life_time
        )

        world.debug.draw_string(
            p
            + carla.Location(
                z=0.25
            ),

            f"P{index}",

            draw_shadow=True,

            color=color,

            life_time=life_time
        )


def draw_ego_corridor(
    world,
    ego,
    length,
    color,
    life_time
):
    """
    Ego forward straight swept corridor.
    """

    tf = ego.get_transform()

    center = tf.transform(
        ego.bounding_box.location
    )

    theta = math.radians(
        tf.rotation.yaw
    )

    fx = math.cos(theta)
    fy = math.sin(theta)

    nx = -math.sin(theta)
    ny = math.cos(theta)

    half_length = (
        ego.bounding_box.extent.x
    )

    half_width = (
        ego.bounding_box.extent.y
    )

    front_x = (
        center.x
        + half_length * fx
    )

    front_y = (
        center.y
        + half_length * fy
    )

    z = (
        center.z
        + 0.12
    )

    left_start = carla.Location(
        x=(
            front_x
            + half_width * nx
        ),

        y=(
            front_y
            + half_width * ny
        ),

        z=z
    )

    right_start = carla.Location(
        x=(
            front_x
            - half_width * nx
        ),

        y=(
            front_y
            - half_width * ny
        ),

        z=z
    )

    left_end = carla.Location(
        x=(
            left_start.x
            + length * fx
        ),

        y=(
            left_start.y
            + length * fy
        ),

        z=z
    )

    right_end = carla.Location(
        x=(
            right_start.x
            + length * fx
        ),

        y=(
            right_start.y
            + length * fy
        ),

        z=z
    )

    world.debug.draw_line(
        left_start,
        left_end,

        thickness=0.08,

        color=color,

        life_time=life_time
    )

    world.debug.draw_line(
        right_start,
        right_end,

        thickness=0.08,

        color=color,

        life_time=life_time
    )

    # Cross-lines every 5 m
    for distance in range(
        0,
        int(length) + 1,
        5
    ):

        cx = (
            front_x
            + distance * fx
        )

        cy = (
            front_y
            + distance * fy
        )

        p1 = carla.Location(
            x=(
                cx
                + half_width * nx
            ),

            y=(
                cy
                + half_width * ny
            ),

            z=z
        )

        p2 = carla.Location(
            x=(
                cx
                - half_width * nx
            ),

            y=(
                cy
                - half_width * ny
            ),

            z=z
        )

        world.debug.draw_line(
            p1,
            p2,

            thickness=0.02,

            color=color,

            life_time=life_time
        )


def draw_longitudinal_gap(
    world,
    ego,
    d,
    life_time
):
    """
    Draw d along ego heading.
    """

    if d is None:
        return

    tf = ego.get_transform()

    center = tf.transform(
        ego.bounding_box.location
    )

    theta = math.radians(
        tf.rotation.yaw
    )

    fx = math.cos(theta)
    fy = math.sin(theta)

    half_length = (
        ego.bounding_box.extent.x
    )

    start = carla.Location(
        x=(
            center.x
            + half_length * fx
        ),

        y=(
            center.y
            + half_length * fy
        ),

        z=(
            center.z
            + 1.6
        )
    )

    end = carla.Location(
        x=(
            start.x
            + d * fx
        ),

        y=(
            start.y
            + d * fy
        ),

        z=start.z
    )

    color = carla.Color(
        255,
        165,
        0
    )

    world.debug.draw_line(
        start,
        end,

        thickness=0.12,

        color=color,

        life_time=life_time
    )

    world.debug.draw_point(
        start,

        size=0.13,

        color=color,

        life_time=life_time
    )

    world.debug.draw_point(
        end,

        size=0.13,

        color=color,

        life_time=life_time
    )

    middle = carla.Location(
        x=(
            start.x
            + end.x
        ) / 2.0,

        y=(
            start.y
            + end.y
        ) / 2.0,

        z=(
            start.z
            + 0.3
        )
    )

    world.debug.draw_string(
        middle,

        f"d={d:.2f}m",

        draw_shadow=True,

        color=color,

        life_time=life_time
    )


def draw_vehicle_information(
    world,
    ego,
    npc,
    debug,
    npc_theta,
    cutin_progress,
    life_time
):
    """
    Draw state text above vehicles.
    """

    ego_location = (
        ego.get_location()
        + carla.Location(
            z=2.8
        )
    )

    npc_location = (
        npc.get_location()
        + carla.Location(
            z=2.8
        )
    )

    if debug["mode"] == "ACC":

        ego_color = carla.Color(
            255,
            80,
            80
        )

    else:

        ego_color = carla.Color(
            50,
            255,
            50
        )

    if debug["d"] is None:

        d_text = "---"

    else:

        d_text = (
            f"{debug['d']:.2f}m"
        )

    ego_text = (
        f"{debug['mode']} | "
        f"v={debug['v']:.2f} | "
        f"a={debug['a']:.2f} | "
        f"a_hat={debug['a_hat']:.2f} | "
        f"d={d_text}"
    )

    npc_text = (
        f"NPC | "
        f"vL={debug['vL']:.2f} | "
        f"theta={math.degrees(npc_theta):.1f}deg | "
        f"cutin={cutin_progress:.2f}"
    )

    world.debug.draw_string(
        ego_location,

        ego_text,

        draw_shadow=True,

        color=ego_color,

        life_time=life_time
    )

    world.debug.draw_string(
        npc_location,

        npc_text,

        draw_shadow=True,

        color=carla.Color(
            255,
            255,
            0
        ),

        life_time=life_time
    )


def update_spectator(
    world,
    ego,
    npc
):
    """
    Camera follows ego from farther behind.

    It does NOT use the ego/NPC midpoint, so the camera
    remains visually stable during the cut-in.
    """

    spectator = (
        world.get_spectator()
    )

    ego_tf = (
        ego.get_transform()
    )

    ego_location = (
        ego_tf.location
    )

    ego_yaw = (
        ego_tf.rotation.yaw
    )

    theta = math.radians(
        ego_yaw
    )

    # --------------------------------------------------------
    # Camera farther behind
    # --------------------------------------------------------

    camera_distance = 18.0
    camera_height = 9.0

    camera_location = carla.Location(
        x=(
            ego_location.x
            - camera_distance
            * math.cos(theta)
        ),

        y=(
            ego_location.y
            - camera_distance
            * math.sin(theta)
        ),

        z=(
            ego_location.z
            + camera_height
        )
    )

    camera_rotation = carla.Rotation(
        pitch=-24.0,
        yaw=ego_yaw,
        roll=0.0
    )

    spectator.set_transform(
        carla.Transform(
            camera_location,
            camera_rotation
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # CARLA
    # ========================================================

    HOST = "localhost"
    PORT = 2000

    MAP_NAME = "Town06"

    DT = 0.05

    # ========================================================
    # Scenario
    # ========================================================

    NPC_LONGITUDINAL_SPEED = 5.0

    NPC_INITIAL_DISTANCE = 20.0

    CUTIN_START_TIME = 5.0

    CUTIN_DURATION = 2.0

    SIMULATION_DURATION = 20.0

    # ========================================================
    # Cruise
    # ========================================================

    V_REF = 8.0

    TC = 1.0

    # ========================================================
    # ACC
    # ========================================================

    KD = 0.30

    KV = 0.80

    TIME_HEADWAY = 1.2

    ETA = 3.0

    # ========================================================
    # PI
    # ========================================================

    THROTTLE_KP = 0.35
    THROTTLE_KI = 0.10

    BRAKE_KP = 0.25
    BRAKE_KI = 0.08

    # ========================================================
    # Visualization
    # ========================================================

    VISUALIZE = True

    CORRIDOR_LENGTH = 50.0

    # ========================================================
    # Required road length
    # ========================================================

    required_length = (
        NPC_INITIAL_DISTANCE
        +
        NPC_LONGITUDINAL_SPEED
        * SIMULATION_DURATION
        +
        30.0
    )

    # ========================================================
    # Connect to CARLA
    # ========================================================

    client = carla.Client(
        HOST,
        PORT
    )

    client.set_timeout(
        10.0
    )

    print(
        f"\nLoading {MAP_NAME}..."
    )

    client.load_world(
        MAP_NAME
    )

    world = (
        client.get_world()
    )

    traffic_manager = (
        client.get_trafficmanager(
            8000
        )
    )

    original_settings = (
        world.get_settings()
    )

    ego = None
    npc = None
    collision_monitor = None

    try:

        # ====================================================
        # Synchronous + rendering
        # ====================================================

        settings = (
            world.get_settings()
        )

        settings.synchronous_mode = True

        settings.fixed_delta_seconds = DT

        # Ensure vehicle 3D meshes are rendered
        settings.no_rendering_mode = False

        world.apply_settings(
            settings
        )

        traffic_manager.set_synchronous_mode(
            True
        )

        # Give renderer a couple frames after map loading
        world.tick()
        world.tick()

        # ====================================================
        # Blueprints
        # ====================================================

        blueprint_library = (
            world.get_blueprint_library()
        )

        ego_candidates = (
            blueprint_library.filter(
                "vehicle.tesla.model3"
            )
        )

        if not ego_candidates:

            ego_candidates = (
                blueprint_library.filter(
                    "*model3*"
                )
            )

        if not ego_candidates:

            raise RuntimeError(
                "Cannot find Tesla Model 3 blueprint."
            )

        # Separate blueprint objects for colors
        ego_bp = ego_candidates[0]

        npc_candidates = (
            blueprint_library.filter(
                "vehicle.tesla.model3"
            )
        )

        npc_bp = npc_candidates[0]

        if ego_bp.has_attribute("color"):

            ego_bp.set_attribute(
                "color",
                "0,80,255"
            )

        if npc_bp.has_attribute("color"):

            npc_bp.set_attribute(
                "color",
                "255,40,40"
            )

        carla_map = (
            world.get_map()
        )

        spawn_points = (
            carla_map.get_spawn_points()
        )

        # ====================================================
        # Find road
        # ====================================================

        road = (
            find_suitable_two_lane_segment(
                carla_map=carla_map,

                spawn_points=spawn_points,

                npc_initial_distance=
                    NPC_INITIAL_DISTANCE,

                required_length=
                    required_length,

                check_step=2.0,

                max_heading_change=6.0
            )
        )

        if road is None:

            raise RuntimeError(
                "Cannot find suitable "
                "straight two-lane road."
            )

        ego_start_wp = (
            road["ego_start_wp"]
        )

        npc_source_wp = (
            road["npc_source_wp"]
        )

        npc_target_wp = (
            road["npc_target_wp"]
        )

        # ====================================================
        # Spawn ego
        # ====================================================

        ego_transform = carla.Transform(
            ego_start_wp
            .transform
            .location
            +
            carla.Location(
                z=0.2
            ),

            ego_start_wp
            .transform
            .rotation
        )

        ego = world.try_spawn_actor(
            ego_bp,
            ego_transform
        )

        if ego is None:

            raise RuntimeError(
                "Failed to spawn ego vehicle."
            )

        ego.set_autopilot(
            False
        )

        ego.set_simulate_physics(
            True
        )

        # ====================================================
        # Spawn NPC
        # ====================================================

        npc_transform = carla.Transform(
            npc_source_wp
            .transform
            .location
            +
            carla.Location(
                z=0.2
            ),

            npc_source_wp
            .transform
            .rotation
        )

        npc = world.try_spawn_actor(
            npc_bp,
            npc_transform
        )

        if npc is None:

            raise RuntimeError(
                "Failed to spawn NPC vehicle."
            )

        npc.set_autopilot(
            False
        )

        # ====================================================
        # IMPORTANT:
        #
        # NPC is a kinematic/scripted vehicle.
        #
        # The actor and its 3D mesh still exist and render,
        # but CARLA physics does not fight set_transform().
        # ====================================================

        npc.set_simulate_physics(
            False
        )

        # Give renderer time to instantiate both meshes
        world.tick()
        world.tick()
        world.tick()

        print()
        print(
            f"Ego actor: "
            f"id={ego.id}, "
            f"type={ego.type_id}"
        )

        print(
            f"NPC actor: "
            f"id={npc.id}, "
            f"type={npc.type_id}"
        )

        print(
            f"NPC initial location: "
            f"{npc.get_location()}"
        )

        # ====================================================
        # Collision sensor
        # ====================================================

        collision_monitor = (
            CollisionMonitor(
                world,
                ego
            )
        )

        # ====================================================
        # Controller
        # ====================================================

        controller = (
            EgoController(
                vehicle=ego,

                dt=DT,

                v_ref=V_REF,

                Tc=TC,

                kd=KD,

                kv=KV,

                time_headway=
                    TIME_HEADWAY,

                eta=ETA,

                throttle_kp=
                    THROTTLE_KP,

                throttle_ki=
                    THROTTLE_KI,

                brake_kp=
                    BRAKE_KP,

                brake_ki=
                    BRAKE_KI
            )
        )

        # ====================================================
        # NPC trajectory states
        # ====================================================

        source_wp = (
            npc_source_wp
        )

        target_wp = (
            npc_target_wp
        )

        # Use actual spawned pose as initial scripted position
        npc_initial_tf = (
            npc.get_transform()
        )

        previous_npc_x = (
            npc_initial_tf.location.x
        )

        previous_npc_y = (
            npc_initial_tf.location.y
        )

        current_npc_yaw = (
            npc_initial_tf.rotation.yaw
        )

        sim_time = 0.0

        frame = 0

        # ====================================================
        # Setup print
        # ====================================================

        print()
        print(
            "===================================================="
        )
        print(
            "Two-lane cut-in experiment"
        )
        print(
            "===================================================="
        )
        print(
            f"Map                 : {MAP_NAME}"
        )
        print(
            f"dt                  : {DT:.3f} s"
        )
        print(
            f"Ego target speed    : {V_REF:.2f} m/s"
        )
        print(
            f"NPC longitudinal v  : "
            f"{NPC_LONGITUDINAL_SPEED:.2f} m/s"
        )
        print(
            f"NPC initial distance: "
            f"{NPC_INITIAL_DISTANCE:.2f} m"
        )
        print(
            f"Cut-in starts       : "
            f"{CUTIN_START_TIME:.2f} s"
        )
        print(
            f"Cut-in duration     : "
            f"{CUTIN_DURATION:.2f} s"
        )
        print()
        print(
            "CRUISE:"
        )
        print(
            "  a_hat = (v_ref - v) / Tc"
        )
        print()
        print(
            "ACC:"
        )
        print(
            "  a_hat = "
            "kd(d - th*v - eta) + kv(vL - v)"
        )
        print()
        print(
            "ACC trigger:"
        )
        print(
            "  first NPC bbox corner entering "
            "ego forward corridor"
        )
        print()
        print(
            "NPC:"
        )
        print(
            "  rendered CARLA vehicle actor"
        )
        print(
            "  physics disabled"
        )
        print(
            "  pose controlled by set_transform()"
        )
        print(
            "===================================================="
        )
        print()

        # ====================================================
        # Simulation loop
        # ====================================================

        while (
            sim_time
            <= SIMULATION_DURATION
        ):

            # =================================================
            # 1. Advance reference lane positions
            # =================================================

            step_distance = (
                NPC_LONGITUDINAL_SPEED
                * DT
            )

            next_source_wp = (
                get_same_lane_next(
                    source_wp,
                    step_distance
                )
            )

            next_target_wp = (
                get_same_lane_next(
                    target_wp,
                    step_distance
                )
            )

            if (
                next_source_wp is None
                or
                next_target_wp is None
            ):

                print(
                    "\nNPC reference trajectory "
                    "reached end of road."
                )

                break

            source_wp = (
                next_source_wp
            )

            target_wp = (
                next_target_wp
            )

            # =================================================
            # 2. Cut-in progress
            # =================================================

            if (
                sim_time
                < CUTIN_START_TIME
            ):

                raw_progress = 0.0

            else:

                raw_progress = (
                    sim_time
                    - CUTIN_START_TIME
                ) / CUTIN_DURATION

            cutin_progress = (
                smoothstep(
                    raw_progress
                )
            )

            # =================================================
            # 3. NPC scripted position
            # =================================================

            source_loc = (
                source_wp
                .transform
                .location
            )

            target_loc = (
                target_wp
                .transform
                .location
            )

            npc_x = (
                source_loc.x
                +
                cutin_progress
                * (
                    target_loc.x
                    - source_loc.x
                )
            )

            npc_y = (
                source_loc.y
                +
                cutin_progress
                * (
                    target_loc.y
                    - source_loc.y
                )
            )

            # IMPORTANT:
            # use actual waypoint z directly;
            # do not continuously add +0.3 here
            npc_z = (
                source_loc.z
                +
                cutin_progress
                * (
                    target_loc.z
                    - source_loc.z
                )
            )

            # =================================================
            # 4. NPC world-frame velocity
            # =================================================

            motion_dx = (
                npc_x
                - previous_npc_x
            )

            motion_dy = (
                npc_y
                - previous_npc_y
            )

            npc_vx = (
                motion_dx
                / DT
            )

            npc_vy = (
                motion_dy
                / DT
            )

            motion_distance = math.sqrt(
                motion_dx ** 2
                +
                motion_dy ** 2
            )

            # =================================================
            # 5. NPC heading from actual path tangent
            # =================================================

            if (
                motion_distance
                > 1e-6
            ):

                current_npc_yaw = (
                    math.degrees(
                        math.atan2(
                            motion_dy,
                            motion_dx
                        )
                    )
                )

            # =================================================
            # 6. Move the rendered NPC actor
            # =================================================

            npc_pose = carla.Transform(
                carla.Location(
                    x=npc_x,
                    y=npc_y,
                    z=npc_z
                ),

                carla.Rotation(
                    yaw=current_npc_yaw
                )
            )

            npc.set_transform(
                npc_pose
            )

            previous_npc_x = (
                npc_x
            )

            previous_npc_y = (
                npc_y
            )

            # =================================================
            # 7. Ego state
            # =================================================

            v = get_speed(
                ego
            )

            a = (
                get_longitudinal_acceleration(
                    ego
                )
            )

            ego_tf_now = (
                ego.get_transform()
            )

            ego_theta = math.radians(
                ego_tf_now
                .rotation
                .yaw
            )

            ego_fx = math.cos(
                ego_theta
            )

            ego_fy = math.sin(
                ego_theta
            )

            # =================================================
            # 8. NPC longitudinal speed for ACC
            #
            # vL = v_NPC projected onto ego heading
            # =================================================

            vL = (
                npc_vx
                * ego_fx
                +
                npc_vy
                * ego_fy
            )

            # =================================================
            # 9. NPC pose/state
            # =================================================

            npc_tf_now = (
                npc.get_transform()
            )

            npc_theta = math.radians(
                npc_tf_now
                .rotation
                .yaw
            )

            npc_center = (
                npc_tf_now.transform(
                    npc.bounding_box.location
                )
            )

            # =================================================
            # 10. Controller
            # =================================================

            control, debug = (
                controller.run_step(
                    v=v,

                    a=a,

                    vL=vL,

                    npc_theta=
                        npc_theta,

                    npc_x=
                        npc_center.x,

                    npc_y=
                        npc_center.y,

                    npc_vehicle=
                        npc
                )
            )

            ego.apply_control(
                control
            )

            # =================================================
            # 11. Visualization
            # =================================================

            if VISUALIZE:

                life_time = (
                    DT * 1.5
                )

                # Camera
                update_spectator(
                    world,
                    ego,
                    npc
                )

                # ---------------------------------------------
                # Corridor
                # ---------------------------------------------

                if (
                    debug["mode"]
                    == "CRUISE"
                ):

                    corridor_color = (
                        carla.Color(
                            0,
                            255,
                            255
                        )
                    )

                else:

                    corridor_color = (
                        carla.Color(
                            255,
                            60,
                            60
                        )
                    )

                # draw_ego_corridor(
                #     world=world,

                #     ego=ego,

                #     length=
                #         CORRIDOR_LENGTH,

                #     color=
                #         corridor_color,

                #     life_time=
                #         life_time
                # )

                # ---------------------------------------------
                # Ego bbox
                # ---------------------------------------------

                # draw_vehicle_bbox(
                #     world=world,

                #     vehicle=ego,

                #     color=carla.Color(
                #         0,
                #         255,
                #         0
                #     ),

                #     thickness=0.06,

                #     life_time=
                #         life_time
                # )

                # ---------------------------------------------
                # NPC bbox
                # ---------------------------------------------

                # draw_vehicle_bbox(
                #     world=world,

                #     vehicle=npc,

                #     color=carla.Color(
                #         255,
                #         165,
                #         0
                #     ),

                #     thickness=0.06,

                #     life_time=
                #         life_time
                # )

                # ---------------------------------------------
                # NPC corners
                # ---------------------------------------------

                # draw_npc_corners(
                #     world=world,

                #     npc=npc,

                #     life_time=
                #         life_time
                # )

                # ---------------------------------------------
                # Distance d
                # ---------------------------------------------

                # if (
                #     debug["d"]
                #     is not None
                # ):

                #     draw_longitudinal_gap(
                #         world=world,

                #         ego=ego,

                #         d=
                #             debug["d"],

                #         life_time=
                #             life_time
                #     )

                # ---------------------------------------------
                # State text
                # ---------------------------------------------

                draw_vehicle_information(
                    world=world,

                    ego=ego,

                    npc=npc,

                    debug=debug,

                    npc_theta=
                        npc_theta,

                    cutin_progress=
                        cutin_progress,

                    life_time=
                        life_time
                )

            # =================================================
            # 12. Terminal output
            # =================================================

            print_interval = max(
                1,
                int(
                    0.25
                    / DT
                )
            )

            if (
                frame
                % print_interval
                == 0
            ):

                if (
                    debug["d"]
                    is None
                ):

                    d_text = (
                        "  --- "
                    )

                else:

                    d_text = (
                        f"{debug['d']:6.2f}"
                    )

                print(
                    f"t={sim_time:5.2f} | "
                    f"{debug['mode']:6s} | "
                    f"v={debug['v']:5.2f} | "
                    f"vL={debug['vL']:5.2f} | "
                    f"d={d_text} | "
                    f"a={debug['a']:6.2f} | "
                    f"a_hat={debug['a_hat']:6.2f} | "
                    f"T={debug['throttle']:.2f} | "
                    f"B={debug['brake']:.2f} | "
                    f"thetaL="
                    f"{math.degrees(npc_theta):7.2f} | "
                    f"cutin="
                    f"{cutin_progress:.3f}"
                )

            # =================================================
            # 13. Collision
            # =================================================

            if (
                collision_monitor
                is not None
                and
                collision_monitor.collided
            ):

                print(
                    f"\nExperiment terminated "
                    f"at t={sim_time:.2f}s "
                    f"due to collision."
                )

                break

            # =================================================
            # 14. Tick
            # =================================================

            world.tick()

            sim_time += DT

            frame += 1

        # ====================================================
        # Result
        # ====================================================

        print()
        print(
            "===================================================="
        )

        if (
            collision_monitor
            is not None
            and
            collision_monitor.collided
        ):

            print(
                "RESULT: COLLISION"
            )

        else:

            print(
                "RESULT: NO COLLISION"
            )

        print(
            "===================================================="
        )

    finally:

        print(
            "\nCleaning up..."
        )

        if (
            collision_monitor
            is not None
        ):

            collision_monitor.destroy()

        if ego is not None:

            ego.destroy()

        if npc is not None:

            npc.destroy()

        traffic_manager.set_synchronous_mode(
            False
        )

        world.apply_settings(
            original_settings
        )

        time.sleep(
            0.5
        )

        print(
            "Done."
        )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nInterrupted by user."
        )