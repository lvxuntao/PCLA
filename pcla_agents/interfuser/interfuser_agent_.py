import os
import json
import copy
import datetime
import pathlib
import time
import importlib
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
# Compatibility for old InterFuser helper code that still uses np.int.
# Keep this local to this GT-controller agent; no repo files need to be edited.
if not hasattr(np, "int"):
    np.int = int
from PIL import Image
from easydict import EasyDict

from torchvision import transforms
from leaderboard_codes import autonomous_agent1 as autonomous_agent
# Neural-network perception is intentionally disabled in this GT-controller version.
# from pcla_agents.interfuser.timm.models import create_model
from utils import lidar_to_histogram_features, transform_2d_points
from planner import RoutePlanner
from interfuser_controller import InterfuserController
from render import render, render_self_car, render_waypoints
from tracker import Tracker
from pcla_agents.interfuser.timm.data.heatmap_utils import generate_heatmap
from pcla_agents.interfuser.timm.data.det_utils import generate_det_data

import math
import yaml

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        rgb_left = input_data['rgb_left']
        rgb_right = input_data['rgb_right']
        rgb_focus = input_data['rgb_focus']
        map = input_data['map']
        surface = np.zeros((600, 1200, 3),np.uint8)
        surface[:, :800] = rgb
        surface[:400,800:1200] = map
        surface[400:600,800:1000] = input_data['map_t1']
        surface[400:600,1000:1200] = input_data['map_t2']
        surface[:150,:200] = input_data['rgb_left']
        surface[:150, 600:800] = input_data['rgb_right']
        surface[:150, 325:475] = input_data['rgb_focus']
        surface = cv2.putText(surface, input_data['control'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][0], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][1], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)

        surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

        surface = cv2.putText(surface, 'Future Prediction', (940,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,385), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '0', (1170,385), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (960,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '1', (970,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '2', (1170,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)

        surface[:150,198:202]=0
        surface[:150,323:327]=0
        surface[:150,473:477]=0
        surface[:150,598:602]=0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255


        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()


def get_entry_point():
    return "InterfuserAgent"

def load_module_from_path(module_name, file_path):
    """
    Loads a module from a given file path using importlib.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot find module '{module_name}' at '{file_path}'")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class InterfuserAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):

        self._hic = DisplayInterface()
        # self._hic = None
        self.lidar_processed = list()
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.tracker = Tracker()

        self.input_buffer = {
            "rgb": deque(),
            "rgb_left": deque(),
            "rgb_right": deque(),
            "rgb_rear": deque(),
            "lidar": deque(),
            "gps": deque(),
            "thetas": deque(),
        }

        self.config = load_module_from_path("MainModel", path_to_conf_file).GlobalConfig()
        self.skip_frames = self.config.skip_frames
        self.controller = InterfuserController(self.config)

        # Controller-only mode:
        # Do NOT load the InterFuser neural network.
        # The perception outputs used by the controller will be generated from
        # CARLA ground-truth actors in run_step().
        self.ensemble = False
        self.net = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None
        self.prev_surround_map = None

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(
                map(
                    lambda x: "%02d" % x,
                    (now.month, now.day, now.hour, now.minute, now.second),
                )
            )

            print(string)

            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Store both GPS and world-coordinate global plans, matching MapAgent.
        The official data collector's MapAgent uses _plan_gps_HACK for
        _waypoint_planner.set_route(..., gps=True).
        """
        try:
            super().set_global_plan(global_plan_gps, global_plan_world_coord)
        except AttributeError:
            pass
        self._global_plan = global_plan_gps
        self._global_plan_world_coord = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps
        self._plan_HACK = global_plan_world_coord

    def _init(self):
        # Original InterFuser route planner used by tick() for next command and target point.
        route_for_planner = getattr(self, "_plan_gps_HACK", None)
        if route_for_planner is None:
            route_for_planner = self._global_plan

        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(route_for_planner, True)

        # Data-collection-compatible waypoint planner.
        # Official MapAgent does:
        #   self._waypoint_planner = RoutePlanner(4.0, 50)
        #   self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        # Keep it separate from _route_planner because run_step() mutates/pops route.
        self._waypoint_planner = RoutePlanner(4.0, 50.0)
        self._waypoint_planner.set_route(route_for_planner, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):

        rgb = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        pos = self._get_position(result)

        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data

        lidar_unprocessed = lidar_data[:, :3]
        lidar_unprocessed[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_unprocessed,
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
        )
        lidar_processed = lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        result["gps"] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_command"] = next_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        return result

    def collect_actor_data_gt(self, ego_vehicle, max_distance=50.0):
        """
        Collect CARLA ground-truth surrounding actors in the same format used by
        InterFuser's data collection code. This replaces AI perception.
        """
        data = {}
        world = ego_vehicle.get_world()
        ego_loc = ego_vehicle.get_location()

        vehicles = world.get_actors().filter("*vehicle*")
        for actor in vehicles:
            if actor.id == ego_vehicle.id or not actor.is_alive:
                continue

            loc = actor.get_location()
            if loc.distance(ego_loc) > max_distance:
                continue

            ori = actor.get_transform().rotation.get_forward_vector()
            box = actor.bounding_box.extent
            vel = actor.get_velocity()

            data[str(actor.id)] = {
                "loc": [loc.x, loc.y, loc.z],
                "ori": [ori.x, ori.y, ori.z],
                "box": [box.x, box.y],
                "vel": [vel.x, vel.y, vel.z],
                "tpe": 0,
            }

        walkers = world.get_actors().filter("*walker*")
        for actor in walkers:
            if not actor.is_alive:
                continue

            loc = actor.get_location()
            if loc.distance(ego_loc) > max_distance:
                continue

            ori = actor.get_transform().rotation.get_forward_vector()
            box = actor.bounding_box.extent
            vel = actor.get_velocity()

            data[str(actor.id)] = {
                "loc": [loc.x, loc.y, loc.z],
                "ori": [ori.x, ori.y, ori.z],
                "box": [box.x, box.y],
                "vel": [vel.x, vel.y, vel.z],
                "tpe": 1,
            }

        return data

    def build_route_waypoints(self, tick_data, num_waypoints=10):
        """
        Build GT waypoint labels in the same way as InterFuser data collection +
        dataset loading.

        Data generation path in the official repo:
            AutoPilot.run_step()
              -> near_node, near_command = self._waypoint_planner.run_step(gps)
              -> save()
              -> "future_waypoints": self._waypoint_planner.get_future_waypoints(50)

        Dataset path:
            for waypoint in measurements["future_waypoints"][:10]:
                new_loc = R.T @ ([waypoint_x - ego_x, waypoint_y - ego_y])
            pad missing waypoints with [10000, 10000]

        This function intentionally does NOT use CARLA lane-center wp.next(),
        and does NOT resample the path. It follows the official data pipeline.
        """
        gps = np.array(tick_data["gps"], dtype=np.float32)

        # Important: the official AutoPilot calls run_step(gps) before saving
        # future_waypoints. run_step() pops route nodes that have already been passed.
        self._waypoint_planner.run_step(gps)

        future_waypoints = self._waypoint_planner.get_future_waypoints(50)

        ego_x, ego_y = gps[0], gps[1]
        ego_theta = tick_data["compass"]
        R = np.array(
            [
                [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
                [np.sin(np.pi / 2 + ego_theta),  np.cos(np.pi / 2 + ego_theta)],
            ],
            dtype=np.float32,
        )

        command_waypoints = []
        for i in range(min(num_waypoints, len(future_waypoints))):
            waypoint = future_waypoints[i]
            new_loc = R.T.dot(
                np.array([waypoint[0] - ego_x, waypoint[1] - ego_y], dtype=np.float32)
            )
            command_waypoints.append(new_loc.reshape(1, 2).astype(np.float32))

        # Same padding behavior as carla_dataset.py.
        for _ in range(num_waypoints - len(command_waypoints)):
            command_waypoints.append(np.array([10000.0, 10000.0], dtype=np.float32).reshape(1, 2))

        pred_waypoints = np.concatenate(command_waypoints, axis=0).astype(np.float32)

        if np.isnan(pred_waypoints).any():
            pred_waypoints[np.isnan(pred_waypoints)] = 0.0

        if os.environ.get("DEBUG_GT_WAYPOINTS", "0") == "1":
            print("future_waypoints[:10] =", future_waypoints[:10])
            print("pred_waypoints =", pred_waypoints)

        return pred_waypoints

    def angle_diff_deg(self, a, b):
        return abs((a - b + 180.0) % 360.0 - 180.0)


    def choose_lane_continuation(self, current_wp, distance):
        """
        Choose the next waypoint that best continues the current lane.
        This avoids accidentally taking a ramp or branch.
        """
        next_wps = current_wp.next(distance)

        if len(next_wps) == 0:
            return None

        # Prefer driving lanes only.
        next_wps = [
            wp for wp in next_wps
            if wp.lane_type == carla.LaneType.Driving
        ]

        if len(next_wps) == 0:
            return None

        # Prefer same lane_id if possible.
        same_lane_wps = [
            wp for wp in next_wps
            if wp.lane_id == current_wp.lane_id
        ]

        candidates = same_lane_wps if len(same_lane_wps) > 0 else next_wps

        ref_yaw = current_wp.transform.rotation.yaw

        return min(
            candidates,
            key=lambda wp: self.angle_diff_deg(wp.transform.rotation.yaw, ref_yaw)
        )


    def world_to_ego_local_lane_point(self, ego_vehicle, target_loc):
        """
        Convert a CARLA world location to InterFuser controller waypoint format.

        Output:
            [lateral, forward]

        For a point directly ahead of ego:
            lateral ≈ 0
            forward > 0
        """
        ego_tf = ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        yaw = math.radians(ego_tf.rotation.yaw)

        dx = target_loc.x - ego_loc.x
        dy = target_loc.y - ego_loc.y

        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy

        return np.array([lateral, -forward], dtype=np.float32)


    def smooth_lane_center_waypoints(self, pred_waypoints, alpha=0.35):
        """
        Smooth lane-center waypoints to reduce controller steering jitter.

        alpha closer to 1.0:
            less smoothing

        alpha closer to 0.0:
            more smoothing
        """
        pred_waypoints = pred_waypoints.astype(np.float32)

        if not hasattr(self, "prev_lane_center_waypoints"):
            self.prev_lane_center_waypoints = None

        if self.prev_lane_center_waypoints is None:
            self.prev_lane_center_waypoints = pred_waypoints.copy()
            return pred_waypoints

        # If waypoint sequence jumps too much, reset instead of blending.
        jump = np.linalg.norm(pred_waypoints[0] - self.prev_lane_center_waypoints[0])
        if jump > 5.0:
            self.prev_lane_center_waypoints = pred_waypoints.copy()
            return pred_waypoints

        smoothed = alpha * pred_waypoints + (1.0 - alpha) * self.prev_lane_center_waypoints
        self.prev_lane_center_waypoints = smoothed.copy()

        return smoothed.astype(np.float32)


    def build_lane_center_waypoints(
        self,
        ego_vehicle,
        num_waypoints=10,
        spacing=1.0,
        first_distance=3.5,
        smooth=True
    ):
        """
        Build stable lane-center waypoints from the current CARLA lane.

        With spacing=1.0 and first_distance=2.0, this generates:
            2m, 3m, 4m, ..., 11m ahead

        This avoids the first points being hidden inside/under the ego car.
        It also avoids duplicate padding when CARLA waypoint.next() fails.
        """
        world = ego_vehicle.get_world()
        carla_map = world.get_map()

        ego_loc = ego_vehicle.get_location()
        ego_wp = carla_map.get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # Emergency fallback: straight line in ego frame.
        if ego_wp is None:
            pred_waypoints = np.array(
                [[0.0, first_distance + spacing * i] for i in range(num_waypoints)],
                dtype=np.float32
            )
            return pred_waypoints

        command_waypoints = []
        last_valid_local_wp = None

        ref_yaw = ego_wp.transform.rotation.yaw

        for i in range(num_waypoints):
            dist = first_distance + spacing * i

            next_wps = ego_wp.next(dist)

            chosen_wp = None

            if len(next_wps) > 0:
                # Prefer driving lanes.
                next_wps = [
                    wp for wp in next_wps
                    if wp.lane_type == carla.LaneType.Driving
                ]

            if len(next_wps) > 0:
                # Prefer same lane id.
                same_lane_wps = [
                    wp for wp in next_wps
                    if wp.lane_id == ego_wp.lane_id
                ]

                candidates = same_lane_wps if len(same_lane_wps) > 0 else next_wps

                chosen_wp = min(
                    candidates,
                    key=lambda wp: self.angle_diff_deg(wp.transform.rotation.yaw, ref_yaw)
                )

            if chosen_wp is not None:
                local_wp = self.world_to_ego_local_lane_point(
                    ego_vehicle,
                    chosen_wp.transform.location
                )
                last_valid_local_wp = local_wp.copy()
            else:
                # Fallback: keep generating unique forward points.
                # This avoids repeated duplicate waypoints.
                local_wp = np.array([0.0, dist], dtype=np.float32)

                # If we already have a valid lane-center point, preserve its lateral offset.
                if last_valid_local_wp is not None:
                    local_wp[0] = last_valid_local_wp[0]

            command_waypoints.append(local_wp.reshape(1, 2).astype(np.float32))

        pred_waypoints = np.concatenate(command_waypoints, axis=0).astype(np.float32)

        if np.isnan(pred_waypoints).any():
            pred_waypoints[np.isnan(pred_waypoints)] = 0.0

        if smooth:
            pred_waypoints = self.smooth_lane_center_waypoints(
                pred_waypoints,
                alpha=0.35
            )

        if os.environ.get("DEBUG_GT_WAYPOINTS", "0") == "1":
            print("lane_center shape =", pred_waypoints.shape)
            print("lane_center pred_waypoints =", pred_waypoints)

        return pred_waypoints

    def get_gt_affordances(self, ego_vehicle):
        """
        Ground-truth values for the non-BEV controller inputs.
        For your straight-road sudden-brake test, red light and stop sign are disabled.
        """
        world = ego_vehicle.get_world()
        carla_map = world.get_map()
        wp = carla_map.get_waypoint(ego_vehicle.get_location())

        # InterfuserController treats smaller junction values as junction-like;
        # for straight road, use 1.0 when not in junction, 0.0 when in junction.
        is_junction = 0.0 if wp is not None and wp.is_junction else 1.0

        # Disable traffic light and stop sign for this controlled verification case.
        traffic_light_state = 0.0
        stop_sign = 1.0

        return is_junction, traffic_light_state, stop_sign

    def build_gt_traffic_meta(self, ego_vehicle, tick_data):
        """
        Generate InterFuser's 20 x 20 x 7 traffic meta grid from CARLA ground truth.
        """
        loc = ego_vehicle.get_location()

        actors_data = self.collect_actor_data_gt(ego_vehicle)

        # Important: this should match InterFuser data collection format:
        # x/y are CARLA world coordinates; theta is the IMU compass in radians.
        # generate_heatmap() also expects the hazard-id lists saved by the
        # original autopilot data collector. In this GT-controller test, we mark
        # all nearby GT vehicles/walkers as present and leave traffic lights /
        # stop signs disabled.
        vehicle_ids = [
            int(_id) for _id, item in actors_data.items()
            if item.get("tpe", 0) == 0
        ]
        pedestrian_ids = [
            int(_id) for _id, item in actors_data.items()
            if item.get("tpe", 0) == 1
        ]
        measurements = {
            "x": loc.x,
            "y": loc.y,
            "theta": tick_data["compass"],
            "is_vehicle_present": vehicle_ids,
            "is_bike_present": [],
            "is_lane_vehicle_present": [],
            "is_junction_vehicle_present": [],
            "is_pedestrian_present": pedestrian_ids,
            "is_red_light_present": [],
            "is_stop_sign_present": [],
        }

        if len(actors_data) == 0:
            return np.zeros((400, 7), dtype=np.float32)

        heatmap = generate_heatmap(measurements, copy.deepcopy(actors_data))
        traffic_meta = generate_det_data(
            heatmap, measurements, copy.deepcopy(actors_data)
        ).reshape(400, 7)

        return traffic_meta.astype(np.float32)

    @torch.no_grad()
    def run_step(self, input_data, timestamp, vehicle=None):
        if not self.initialized:
            self._init()

        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control

        tick_data = self.tick(input_data)

        velocity = tick_data["speed"]

        # PCLA passes the ego CARLA actor as vehicle. If not provided, try the
        # leaderboard-style self._vehicle fallback.
        ego_vehicle = vehicle if vehicle is not None else getattr(self, "_vehicle", None)
        if ego_vehicle is None:
            raise RuntimeError(
                "GT-controller mode requires the ego CARLA vehicle actor. "
                "Call run_step(input_data, timestamp, vehicle=ego_vehicle)."
            )

        # ===== Controller-only replacement for InterFuser neural perception =====
        # pred_waypoints = self.build_route_waypoints(tick_data)
        pred_waypoints = self.build_lane_center_waypoints(
            ego_vehicle,
            num_waypoints=10,
            spacing=1.0,
            smooth=True
        )
        is_junction, traffic_light_state, stop_sign = self.get_gt_affordances(ego_vehicle)
        traffic_meta = self.build_gt_traffic_meta(ego_vehicle, tick_data)

        # Keep the variable name used by the original visualization/controller code.
        self.traffic_meta_moving_avg = traffic_meta

        tick_data["raw"] = traffic_meta
        tick_data["bev_feature"] = None

        steer, throttle, brake, meta_infos = self.controller.run_step(
            velocity,
            pred_waypoints,
            is_junction,
            traffic_light_state,
            stop_sign,
            self.traffic_meta_moving_avg,
        )

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        surround_map, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20)
        surround_map = surround_map[:400, 160:560]
        surround_map = np.stack([surround_map, surround_map, surround_map], 2)

        self_car_map = render_self_car(
            loc=np.array([0, 0]),
            ori=np.array([0, -1]),
            box=np.array([2.45, 1.0]),
            color=[1, 1, 0], pixels_per_meter=20
        )[:400, 160:560]

        pred_waypoints = pred_waypoints.reshape(-1, 2)
        safe_index = 10
        for i in range(10):
            if pred_waypoints[i, 0] ** 2 + pred_waypoints[i, 1] ** 2> (meta_infos[3]+0.5) ** 2:
                safe_index = i
                break
        wp1 = render_waypoints(pred_waypoints[:safe_index], pixels_per_meter=20, color=(0, 255, 0))[:400, 160:560]
        wp2 = render_waypoints(pred_waypoints[safe_index:], pixels_per_meter=20, color=(255, 0, 0))[:400, 160:560]
        wp = wp1 + wp2

        surround_map = np.clip(
            (
                surround_map.astype(np.float32)
                + self_car_map.astype(np.float32)
                + wp.astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)

        map_t1, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=1)
        map_t1 = map_t1[:400, 160:560]
        map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
        map_t1 = np.clip(map_t1.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        map_t1 = cv2.resize(map_t1, (200, 200))
        map_t2, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=2)
        map_t2 = map_t2[:400, 160:560]
        map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
        map_t2 = np.clip(map_t2.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        map_t2 = cv2.resize(map_t2, (200, 200))


        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control
            self.prev_surround_map = surround_map

        tick_data["map"] = self.prev_surround_map
        tick_data["map_t1"] = map_t1
        tick_data["map_t2"] = map_t2
        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]

        tick_data["rgb"] = cv2.resize(tick_data["rgb"], (800, 600))
        tick_data["rgb_left"] = cv2.resize(tick_data["rgb_left"], (200, 150))
        tick_data["rgb_right"] = cv2.resize(tick_data["rgb_right"], (200, 150))
        tick_data["rgb_focus"] = cv2.resize(tick_data["rgb_raw"][244:356, 344:456], (150, 150))
        tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
            control.throttle,
            control.steer,
            control.brake,
        )
        tick_data["meta_infos"] = meta_infos
        tick_data["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (
            box_info["car"],
            box_info["bike"],
            box_info["pedestrian"],
        )
        tick_data["mes"] = "speed: %.2f" % velocity
        tick_data["time"] = "time: %.3f" % timestamp
        # surface = self._hic.run_interface(tick_data)
        # tick_data["surface"] = surface
        if self._hic is not None:
            surface = self._hic.run_interface(tick_data)
            tick_data["surface"] = surface
        else:
            tick_data["surface"] = np.zeros((600, 1200, 3), dtype=np.uint8)

        # if SAVE_PATH is not None:
        #     self.save(tick_data)
        if False and SAVE_PATH is not None:
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step // self.skip_frames
        Image.fromarray(tick_data["surface"]).save(
            self.save_path / "meta" / ("%04d.jpg" % frame)
        )
        return

    def destroy(self):
        if hasattr(self, "nets"):
            del self.nets
        if hasattr(self, "net"):
            del self.net
        if hasattr(self, "_hic") and self._hic is not None:
            self._hic._quit()