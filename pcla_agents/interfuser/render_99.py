import numpy as np
import cv2
import math

reweight_array = np.array([1.0, 3.5, 3.5, 2.0, 3.5, 2.0, 8.0])


def add_rect(img, loc, ori, box, value, pixels_per_meter, max_distance, color):
    img_size = max_distance * pixels_per_meter * 2
    vet_ori = np.array([-ori[1], ori[0]])
    hor_offset = box[0] * ori
    vet_offset = box[1] * vet_ori
    left_up = (loc + hor_offset + vet_offset + max_distance) * pixels_per_meter
    left_down = (loc + hor_offset - vet_offset + max_distance) * pixels_per_meter
    right_up = (loc - hor_offset + vet_offset + max_distance) * pixels_per_meter
    right_down = (loc - hor_offset - vet_offset + max_distance) * pixels_per_meter
    left_up = np.around(left_up).astype(int)
    left_down = np.around(left_down).astype(int)
    right_down = np.around(right_down).astype(int)
    right_up = np.around(right_up).astype(int)
    left_up = list(left_up)
    left_down = list(left_down)
    right_up = list(right_up)
    right_down = list(right_down)
    color = [int(x) for x in value * color]
    cv2.fillConvexPoly(img, np.array([left_up, left_down, right_down, right_up]), color)
    return img


def convert_grid_to_xy(i, j):
    x = j - 9.5
    y = 17.5 - i
    return x, y


def find_peak_box(data):
    det_data = np.zeros((22, 22, 7))
    det_data[1:21, 1:21] = data
    det_data[19:21, 1:21, 0] -= 0.1
    res = []
    for i in range(1, 21):
        for j in range(1, 21):
            if det_data[i, j, 0] > 0.9 or (
                det_data[i, j, 0] > 0.4
                and det_data[i, j, 0] > det_data[i, j - 1, 0]
                and det_data[i, j, 0] > det_data[i, j + 1, 0]
                and det_data[i, j, 0] > det_data[i + 1, j + 1, 0]
                and det_data[i, j, 0] > det_data[i - 1, j + 1, 0]
                and det_data[i, j, 0] > det_data[i + 1, j - 1, 0]
                and det_data[i, j, 0] > det_data[i + 1, j + 1, 0]
                and det_data[i, j, 0] > det_data[i - 1, j, 0]
                and det_data[i, j, 0] > det_data[i + 1, j, 0]
            ):
                res.append((i - 1, j - 1))
    box_info = {"car": [], "bike": [], "pedestrian": []}
    for instance in res:
        i, j = instance
        box = np.array(det_data[i + 1, j + 1, 4:6])
        if box[0] > 2.0:
            box_info["car"].append((i, j))
        elif box[0] / box[1] > 1.5:
            box_info["bike"].append((i, j))
        else:
            box_info["pedestrian"].append((i, j))
    return res, box_info


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _actor_type_from_box(box):
    if box[0] > 2.0:
        return "car"
    if box[1] > 1e-6 and box[0] / box[1] > 1.5:
        return "bike"
    return "pedestrian"


def _extract_actor_states(det_data):
    """
    Extract actor pose using the same coordinates and reweighting as render().
    """
    weighted_data = det_data * reweight_array
    box_ids, _ = find_peak_box(weighted_data)

    actor_states = []
    for i, j in box_ids:
        center_x, center_y = convert_grid_to_xy(i, j)
        loc = np.array(
            [
                center_x + weighted_data[i, j, 1],
                -(center_y + weighted_data[i, j, 2]),
            ],
            dtype=float,
        )
        theta = float(weighted_data[i, j, 3] * np.pi)
        box = np.array(weighted_data[i, j, 4:6], dtype=float)
        actor_states.append(
            {
                "poi": (i, j),
                "loc": loc,
                "theta": theta,
                "box": box,
                "type": _actor_type_from_box(box),
            }
        )
    return actor_states


def estimate_actor_yaw_rates(
    det_data,
    previous_actor_states,
    dt=0.05,
    max_match_distance=3.0,
):
    """
    Match current actors to the previous frame and estimate yaw rate:
        yaw_rate = wrapped(current_yaw - previous_yaw) / dt

    Matching uses nearest position among objects of the same type because the
    same actor can move to a different 20x20 grid cell between frames.
    """
    current_actor_states = _extract_actor_states(det_data)
    yaw_rate_map = {}

    if (
        previous_actor_states is None
        or len(previous_actor_states) == 0
        or dt <= 0
    ):
        return yaw_rate_map, current_actor_states

    used_previous_indices = set()

    for current_state in current_actor_states:
        best_previous_index = None
        best_distance = float("inf")

        for previous_index, previous_state in enumerate(previous_actor_states):
            if previous_index in used_previous_indices:
                continue
            if previous_state["type"] != current_state["type"]:
                continue

            distance = float(
                np.linalg.norm(current_state["loc"] - previous_state["loc"])
            )
            if distance < best_distance:
                best_distance = distance
                best_previous_index = previous_index

        if (
            best_previous_index is not None
            and best_distance <= max_match_distance
        ):
            previous_state = previous_actor_states[best_previous_index]
            yaw_delta = _wrap_angle(
                current_state["theta"] - previous_state["theta"]
            )
            yaw_rate_map[current_state["poi"]] = yaw_delta / dt
            used_previous_indices.add(best_previous_index)

    return yaw_rate_map, current_actor_states


def _constant_turn_pose(initial_loc, initial_theta, speed, yaw_rate, t):
    """
    Constant-turn-rate and constant-speed circular prediction.
    """
    if abs(yaw_rate) < 1e-6:
        future_theta = initial_theta
        future_loc = initial_loc + t * speed * np.array(
            [math.cos(initial_theta), math.sin(initial_theta)]
        )
        return future_loc, future_theta

    future_theta = initial_theta + yaw_rate * t
    future_loc = initial_loc + np.array(
        [
            speed
            / yaw_rate
            * (math.sin(future_theta) - math.sin(initial_theta)),
            speed
            / yaw_rate
            * (math.cos(initial_theta) - math.cos(future_theta)),
        ]
    )
    return future_loc, future_theta


def render_self_car(loc, ori, box, pixels_per_meter=5, max_distance=18, color=None):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.uint8)
    if color is None:
        color = np.array([1, 1, 1])
        new_img = add_rect(
            img, loc, ori, box, 255, pixels_per_meter, max_distance, color
        )
        return new_img[:, :, 0]
    else:
        color = np.array(color)
        new_img = add_rect(
            img, loc, ori, box, 255, pixels_per_meter, max_distance, color
        )
        return new_img


def render(
    det_data,
    pixels_per_meter=5,
    max_distance=18,
    t=0,
    yaw_rate_map=None,
):
    det_data = det_data * reweight_array
    box_ids, box_info = find_peak_box(det_data)
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.uint8)
    for poi in box_ids:
        i, j = poi
        if poi in box_info["bike"]:
            speed = max(4, det_data[i, j, 6])
        else:
            speed = det_data[i, j, 6]
        center_x, center_y = convert_grid_to_xy(i, j)
        act_img = np.zeros((img_size, img_size, 3), np.uint8)
        theta = det_data[i, j, 3] * np.pi
        ori = np.array([math.cos(theta), math.sin(theta)])

        # Original straight-line future pose: always keep and render it.
        loc_x = center_x + det_data[i, j, 1] + t * speed * ori[0]
        loc_y = center_y + det_data[i, j, 2] - t * speed * ori[1]
        loc = np.array([loc_x, -loc_y])

        box = np.array(det_data[i, j, 4:6])
        box[1] = max(0.4, box[1])
        if box[0] < 1.5:
            box = box * 2
        color = np.array([1, 1, 1])
        act_img = add_rect(
            act_img, loc, ori, box, 255, pixels_per_meter, max_distance, color
        )

        # Additional circular future pose based on the previous-frame yaw
        # change. Collision checking therefore uses the union of:
        #   1. original constant-heading straight prediction
        #   2. constant-yaw-rate circular prediction
        if yaw_rate_map is not None and t > 0:
            yaw_rate = yaw_rate_map.get(poi)
            if yaw_rate is not None and abs(yaw_rate) >= 1e-6:
                initial_loc = np.array(
                    [
                        center_x + det_data[i, j, 1],
                        -(center_y + det_data[i, j, 2]),
                    ],
                    dtype=float,
                )
                curved_loc, curved_theta = _constant_turn_pose(
                    initial_loc=initial_loc,
                    initial_theta=theta,
                    speed=speed,
                    yaw_rate=yaw_rate,
                    t=t,
                )
                curved_ori = np.array(
                    [math.cos(curved_theta), math.sin(curved_theta)]
                )
                act_img = add_rect(
                    act_img,
                    curved_loc,
                    curved_ori,
                    box,
                    255,
                    pixels_per_meter,
                    max_distance,
                    color,
                )

        act_img = np.clip(act_img, 0, 255)
        img = img + act_img
    img = np.clip(img, 0, 255)[:, :, 0]
    img = img.astype(np.uint8)

    box_info["car"] = len(box_info["car"])
    box_info["bike"] = len(box_info["bike"])
    box_info["pedestrian"] = len(box_info["pedestrian"])
    return img, box_info


def render_waypoints(waypoints, pixels_per_meter=5, max_distance=18, color=(0, 255, 0)):
    img_size = max_distance * pixels_per_meter * 2
    img = np.zeros((img_size, img_size, 3), np.uint8)
    for i in range(len(waypoints)):
        new_loc = waypoints[i]
        new_loc = new_loc * pixels_per_meter + pixels_per_meter * max_distance
        new_loc = np.around(new_loc)
        new_loc = tuple(new_loc.astype(int))
        img = cv2.circle(img, new_loc, 6, color, -1)
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img
