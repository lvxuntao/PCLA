import carla
import time
from PCLA import PCLA
import math

def get_speed(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)

def main():

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    client.load_world("Town02")

    pcla = None
    settings = None
    vehicle = None
    front_vehicle = None

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(8000)

        # 同步模式
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)

        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05

        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]

        spawn_points = world.get_map().get_spawn_points()

        # =========================
        # ego vehicle（Interfuser控制）
        # =========================
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[31])

        world.tick()

        # =========================
        # 前车（低速行驶）
        # =========================
        ego_transform = vehicle.get_transform()
        forward_vector = ego_transform.get_forward_vector()

        front_location = ego_transform.location + carla.Location(
            x=forward_vector.x * 15,
            y=forward_vector.y * 15,
            z=0.5
        )

        front_transform = carla.Transform(front_location, ego_transform.rotation)

        front_vehicle = world.try_spawn_actor(vehicle_bp, front_transform)

        if front_vehicle is None:
            print("❌ Failed to spawn front vehicle")
            return

        # 不使用 autopilot
        front_vehicle.set_autopilot(False)

        # =========================
        # Interfuser agent
        # =========================
        agent = "if_if"
        route = "./sample_route.xml"

        pcla = PCLA(agent, vehicle, route, client)

        print("\n🚗 Testing Interfuser (low-speed lead vehicle + sudden brake)\n")

        # =========================
        # 参数设置
        # =========================
        target_speed = 4.5   # 前车速度（略低于5 m/s）
        brake_triggered = False

        while True:

            # =========================
            # ego：完全黑盒
            # =========================
            ego_action = pcla.get_action()
            vehicle.apply_control(ego_action)

            # =========================
            # 前车：低速巡航控制
            # =========================
            current_speed = get_speed(front_vehicle)

            if not brake_triggered:
                if current_speed < target_speed:
                    throttle = 0.3
                else:
                    throttle = 0.0

                front_vehicle.apply_control(carla.VehicleControl(
                    throttle=throttle,
                    brake=0.0
                ))

            # =========================
            # 触发急刹（基于距离）
            # =========================
            distance = vehicle.get_location().distance(front_vehicle.get_location())

            if distance < 12 and not brake_triggered:
                print(f">>> Sudden brake triggered! Distance = {distance:.2f} m")

                front_vehicle.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0
                ))

                brake_triggered = True

            # =========================
            # 检查是否安全停车或碰撞
            # =========================
            if brake_triggered:
                ego_speed = get_speed(vehicle)
                if ego_speed < 0.1:
                    print("✅ Successfully stop!")
                    break
                elif distance < 2.0:  # 距离小于2米算碰撞
                    print("❌ Crash!")
                    break

            world.tick()

    finally:
        print("\n🧹 Cleaning up...")

        if pcla is not None:
            pcla.cleanup()

        if vehicle is not None:
            vehicle.destroy()

        if front_vehicle is not None:
            front_vehicle.destroy()

        if settings is not None:
            settings.synchronous_mode = False
            world.apply_settings(settings)

        time.sleep(0.5)
        print("✅ Done.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
