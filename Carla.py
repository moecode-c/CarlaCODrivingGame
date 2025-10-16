'''
carla_4way_intersection.py
Simulates a 4-way intersection controller in CARLA.

How it works (high level):
 - Find traffic lights near a chosen center (auto-detects intersection).
 - Choose 4 closest lights and group them into two opposing groups (NS / EW).
 - Set traffic lights to manual control and cycle:
     NS green -> NS yellow -> EW green -> EW yellow -> repeat
 - Spawn vehicles at map spawn points approaching the intersection and enable autopilot.
 - Count vehicles near each light (simple queue estimator) for extension into adaptive control.

Run:
  - Start CARLA server first.
  - python carla_4way_intersection.py
'''
import carla
import random
import time
import math
import sys
import threading
from collections import defaultdict
import argparse
import os
import platform
try:
    import msvcrt  # windows-only keyboard
except Exception:
    msvcrt = None

# Windows continuous key state polling (for smoother controls)
_get_async_key_state = None
if platform.system().lower().startswith('win'):
    try:
        import ctypes
        _get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
    except Exception:
        _get_async_key_state = None

def _is_down(vk_code: int) -> bool:
    """Return True if a virtual key is currently held down (Windows-only)."""
    if _get_async_key_state is None:
        return False
    try:
        return (_get_async_key_state(vk_code) & 0x8000) != 0
    except Exception:
        return False

# First‑person spectator follow helpers
def _scale(v, s):
    return carla.Location(x=v.x * s, y=v.y * s, z=v.z * s)

def _add(a, b):
    return carla.Location(x=a.x + b.x, y=a.y + b.y, z=a.z + b.z)

def _lerp(a: carla.Location, b: carla.Location, alpha: float) -> carla.Location:
    """Linear interpolate between two Locations."""
    t = max(0.0, min(1.0, alpha))
    return carla.Location(
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        z=a.z + (b.z - a.z) * t,
    )

def _lerp_angle_deg(a: float, b: float, alpha: float) -> float:
    """Interpolate angles in degrees, shortest path wrap-around aware."""
    t = max(0.0, min(1.0, alpha))
    # Normalize to [0,360)
    a = a % 360.0
    b = b % 360.0
    delta = (b - a)
    # Wrap to [-180,180)
    if delta > 180.0:
        delta -= 360.0
    elif delta < -180.0:
        delta += 360.0
    return (a + delta * t)

# ------------------- Configuration -------------------
HOST = '127.0.0.1'
PORT = 2000
# Intersection center override: set to None to auto-detect based on traffic lights
INTERSECTION_CENTER = None  # e.g., carla.Location(x=10.0, y=200.0, z=0.0) or None
# How many vehicles to maintain in the scenario
NUM_VEHICLES = 40
# Distances (meters)
APPROACH_RADIUS = 120.0   # spawn vehicles within this radius of intersection centroid
QUEUE_DISTANCE = 30.0     # distance to check vehicles waiting near a light
# Timings (seconds) - change these to tune behavior
GREEN_TIME = 15.0
YELLOW_TIME = 3.0
# Safety margin when spawning to avoid collisions
SPAWN_RETRY = 10

# ------------------- Utility functions -------------------
def distance(loc1, loc2):
    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y
    dz = loc1.z - loc2.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def angle_between(pivot, target):
    dx = target.x - pivot.x
    dy = target.y - pivot.y
    ang = math.degrees(math.atan2(dy, dx)) % 360.0
    return ang

def group_lights_by_opposite(traffic_lights, center):
    """
    Given a list of 4 traffic lights and the center, group them into two opposing
    groups. We use the angle around center to pair roughly opposite lights.
    Returns (group_A, group_B) where each is a list of traffic light actors.
    """
    lights_with_angle = [(tl, angle_between(center, tl.get_transform().location)) for tl in traffic_lights]
    # sort by angle
    lights_with_angle.sort(key=lambda x: x[1])
    # pair index 0 with 2, 1 with 3 (since sorted by angle)
    groupA = [lights_with_angle[0][0], lights_with_angle[2][0]]
    groupB = [lights_with_angle[1][0], lights_with_angle[3][0]]
    return groupA, groupB

# ------------------- Connection helpers -------------------
def wait_for_carla(host: str, port: int, per_try_timeout: float = 2.0, retries: int = 30):
    """
    Try to connect to CARLA and return (client, world).
    - per_try_timeout: timeout in seconds for each attempt
    - retries: number of attempts (total wait ~ per_try_timeout * retries)
    """
    client = carla.Client(host, port)
    client.set_timeout(per_try_timeout)
    last_err = None
    for i in range(1, retries + 1):
        try:
            world = client.get_world()
            # Successful connection
            return client, world
        except Exception as e:
            last_err = e
            if i % 5 == 1:
                print(f"[INFO] Waiting for CARLA at {host}:{port} (attempt {i}/{retries})...")
            time.sleep(1.0)
    # If here, failed to connect
    raise RuntimeError(
        f"Could not connect to CARLA at {host}:{port} after {retries} attempts (~{int(retries*(per_try_timeout+1))}s). "
        "Ensure CarlaUE4 is running, not blocked by a firewall, and the port matches (default 2000)."
    ) from last_err

# ------------------- CARLA scenario functions -------------------
def find_intersection_lights(world, center=None, max_candidates=4, search_radius=50.0):
    """
    Find traffic lights near center. If center is None, compute centroid of all traffic lights
    and pick the densest cluster of lights as the intersection.
    Returns up to max_candidates lights (usually 4).
    """
    all_tls = world.get_actors().filter('traffic.traffic_light*')
    if len(all_tls) == 0:
        return []

    if center is None:
        # compute centroid of all lights first
        locs = [tl.get_transform().location for tl in all_tls]
        centroid = carla.Location(
            x=sum(l.x for l in locs)/len(locs),
            y=sum(l.y for l in locs)/len(locs),
            z=sum(l.z for l in locs)/len(locs)
        )
    else:
        centroid = center

    # choose lights within radius, otherwise choose closest N
    close = [(tl, distance(centroid, tl.get_transform().location)) for tl in all_tls]
    close.sort(key=lambda x: x[1])
    # pick those within search_radius or the top N
    selected = [tl for (tl, d) in close if d <= search_radius]
    if len(selected) < max_candidates:
        selected = [tl for (tl, d) in close[:max_candidates]]
    else:
        selected = selected[:max_candidates]
    return selected

def set_manual_mode_for_lights(traffic_lights, manual=True):
    for tl in traffic_lights:
        try:
            # Many CARLA versions have 'set_manual_control' or 'set_manual_mode' name differences.
            if hasattr(tl, 'set_manual_control'):
                tl.set_manual_control(manual)
            elif hasattr(tl, 'set_manual_mode'):
                tl.set_manual_mode(manual)
            else:
                # If none available, we'll still call set_state and hope traffic manager respects it.
                pass
        except Exception as e:
            print(f"[WARN] set_manual_mode error on {tl}: {e}")

def set_state(tl, state):
    """Safely set a traffic light state."""
    try:
        tl.set_state(state)
    except Exception as e:
        # older/newer APIs might have slightly different expectations - print but continue
        print(f"[WARN] could not set_state for {tl}: {e}")

def spawn_vehicles(world, blueprint_library, spawn_points, num_vehicles, intersection_center):
    spawned = []
    blueprint_choices = [b for b in blueprint_library.filter('vehicle.*') if int(b.get_attribute('number_of_wheels')) == 4]
    random.shuffle(spawn_points)
    i = 0
    for sp in spawn_points:
        if len(spawned) >= num_vehicles:
            break
        # only spawn vehicles on spawn points that are outside a small radius from the center (approaches)
        if distance(sp.location, intersection_center) > 10.0 and distance(sp.location, intersection_center) <= APPROACH_RADIUS:
            blueprint = random.choice(blueprint_choices)
            # set a random color if available
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            # try to spawn
            attempts = 0
            while attempts < SPAWN_RETRY:
                try:
                    vehicle = world.try_spawn_actor(blueprint, sp)
                    if vehicle:
                        vehicle.set_autopilot(True)  # let CARLA traffic manager drive it
                        spawned.append(vehicle)
                        break
                except Exception:
                    pass
                attempts += 1
    return spawned

def select_spawn_points_for_approaches(world, center, radius=APPROACH_RADIUS):
    """
    Pick spawn points that lie on roads approaching the center, by filtering map spawn points
    by distance to center.
    """
    spawn_points = world.get_map().get_spawn_points()
    # Fallback: some OpenDRIVE worlds don't provide spawn points; derive from waypoints
    if not spawn_points:
        spawn_points = _fallback_spawn_points_from_waypoints(world)
    # filter points at distance approx between 20 and radius
    candidates = [sp for sp in spawn_points if 20.0 < distance(sp.location, center) <= radius]
    # sort by distance (farther first to give space for vehicles to drive in)
    candidates.sort(key=lambda sp: distance(sp.location, center), reverse=True)
    return candidates

def count_vehicles_near_light(world, tl, dist=QUEUE_DISTANCE):
    """Count actors of class vehicle within dist of the traffic light stop line."""
    tl_loc = tl.get_transform().location
    vehicles = world.get_actors().filter('vehicle.*')
    cnt = 0
    for v in vehicles:
        if distance(v.get_transform().location, tl_loc) <= dist:
            cnt += 1
    return cnt

def spawn_ego_vehicle(world, blueprint_library, near_location=None):
    """Spawn a drivable ego vehicle (role_name=hero) near a location if provided."""
    vehicle_bp_list = blueprint_library.filter('vehicle.*')
    # Prefer a simple car
    prefer = [bp for bp in vehicle_bp_list if 'model3' in bp.id or 'mustang' in bp.id or 'a2' in bp.id]
    blueprint = prefer[0] if prefer else vehicle_bp_list[0]
    if blueprint.has_attribute('color'):
        color = blueprint.get_attribute('color').recommended_values[0]
        blueprint.set_attribute('color', color)
    blueprint.set_attribute('role_name', 'hero')

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        spawn_points = _fallback_spawn_points_from_waypoints(world)
    if near_location:
        spawn_points.sort(key=lambda sp: distance(sp.location, near_location))
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, sp)
        if vehicle:
            return vehicle
    raise RuntimeError("Failed to spawn ego vehicle; no free spawn points.")

def _fallback_spawn_points_from_waypoints(world, spacing: float = 15.0, max_points: int = 200):
    """Generate spawn transforms from map waypoints when get_spawn_points() is empty.

    spacing: meters between waypoints to sample
    max_points: cap to avoid excessive transforms
    """
    amap = world.get_map()
    try:
        waypoints = amap.generate_waypoints(spacing)
    except Exception:
        waypoints = []
    transforms = []
    seen = set()
    for wp in waypoints:
        tf = wp.transform
        # Nudge up a bit to avoid ground clipping
        tf.location.z = tf.location.z + 0.2
        key = (round(tf.location.x, 1), round(tf.location.y, 1), round(tf.rotation.yaw, 1))
        if key in seen:
            continue
        seen.add(key)
        transforms.append(tf)
        if len(transforms) >= max_points:
            break
    if not transforms:
        # As a last resort, drop a single transform at origin facing +X
        transforms = [carla.Transform(carla.Location(x=0.0, y=0.0, z=0.3), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))]
    return transforms

def _vehicle_speed_kmh(vehicle) -> float:
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z) * 3.6

def drive_loop_keyboard(vehicle, stop_event, tick_dt=0.02, max_speed_kmh=40.0, easy=False):
    """Smoother WASD control with continuous key sampling, rate limits, and speed limiter.

    Controls:
      - W: accelerate (forward or reverse if reverse mode)
      - S: brake (and small reverse if held in reverse mode)
      - A/D: steer left/right (smooth, with auto-centering)
      - Space: full brake
      - R: toggle reverse
      - Q: quit drive loop
    """
    print("[DRIVE] W/S throttle & brake, A/D steer, Space brake, R reverse toggle, Q quit")
    # Virtual key codes
    VK_W, VK_S, VK_A, VK_D = 0x57, 0x53, 0x41, 0x44
    VK_SPACE, VK_R, VK_Q = 0x20, 0x52, 0x51

    reverse = False
    throttle = 0.0
    brake = 0.0
    steer = 0.0
    steer_target = 0.0
    throttle_target = 0.0
    brake_target = 0.0

    # Tunable rates (per tick)
    if easy:
        accel_rate = 0.04
        brake_rate = 0.06
        steer_rate = 0.06
        return_rate = 0.08
    else:
        accel_rate = 0.08
        brake_rate = 0.10
        steer_rate = 0.10
        return_rate = 0.12

    prev_r_down = False
    prev_q_down = False

    while not stop_event.is_set():
        # Key states (Windows). If not on Windows, fallback to msvcrt pulses.
        w = _is_down(VK_W)
        s = _is_down(VK_S)
        a = _is_down(VK_A)
        d = _is_down(VK_D)
        space = _is_down(VK_SPACE)
        r_down = _is_down(VK_R)
        q_down = _is_down(VK_Q)

        # Fallback for non-Windows: process occasional msvcrt keys
        if _get_async_key_state is None and msvcrt and msvcrt.kbhit():
            key = msvcrt.getch()
            if key in (b'Q', b'q'):
                q_down = True
            elif key in (b'W', b'w'):
                w = True
            elif key in (b'S', b's'):
                s = True
            elif key in (b'A', b'a'):
                a = True
            elif key in (b'D', b'd'):
                d = True
            elif key in (b'R', b'r'):
                r_down = True
            elif key == b' ':
                space = True

        # Toggle reverse on R edge
        if r_down and not prev_r_down:
            reverse = not reverse
        prev_r_down = r_down

        # Quit on Q edge
        if q_down and not prev_q_down:
            stop_event.set()
            break
        prev_q_down = q_down

        # Targets
        if space:
            brake_target = 1.0
            throttle_target = 0.0
        else:
            if w:
                throttle_target = min(1.0, throttle_target + accel_rate)
                brake_target = max(0.0, brake_target - brake_rate)
            elif s:
                # Treat S primarily as brake
                throttle_target = max(0.0, throttle_target - accel_rate)
                brake_target = min(1.0, brake_target + brake_rate)
            else:
                # Natural decay for coasting
                throttle_target = max(0.0, throttle_target - accel_rate*0.5)
                brake_target = max(0.0, brake_target - brake_rate*0.5)

        # Steering target with auto-centering
        if a and not d:
            steer_target = max(-1.0, steer_target - steer_rate)
        elif d and not a:
            steer_target = min(1.0, steer_target + steer_rate)
        else:
            # Return to center smoothly
            if steer_target > 0:
                steer_target = max(0.0, steer_target - return_rate)
            elif steer_target < 0:
                steer_target = min(0.0, steer_target + return_rate)

        # Apply speed limiter
        speed_kmh = _vehicle_speed_kmh(vehicle)
        if speed_kmh > max_speed_kmh:
            throttle_target = min(throttle_target, 0.2)
            if speed_kmh > max_speed_kmh + 10.0:
                brake_target = min(1.0, max(brake_target, 0.2))

        # Move current values toward targets
        def _approach(val, tgt, rate):
            if val < tgt:
                return min(tgt, val + rate)
            elif val > tgt:
                return max(tgt, val - rate)
            return val

        throttle = _approach(throttle, throttle_target, accel_rate)
        brake = _approach(brake, brake_target, brake_rate)
        # Non-linear steering response to reduce sensitivity near center
        steer = _approach(steer, steer_target, steer_rate)
        steer_output = max(-1.0, min(1.0, steer**3))

        control = carla.VehicleControl(throttle=throttle, steer=steer_output, brake=brake, reverse=reverse)
        vehicle.apply_control(control)
        time.sleep(tick_dt)

def attach_first_person_camera(world, ego_vehicle):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    # Driver's seat position (adjust as needed for your car model)
    camera_transform = carla.Transform(
        carla.Location(x=0.4, y=0.0, z=1.2),  # x: forward, y: left/right, z: up
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))
    print("[INFO] First-person camera attached to ego vehicle.")
    return camera

def follow_first_person_spectator(world, vehicle, stop_event, x=0.5, y=0.0, z=1.2, smooth=0.15):
    """
    Moves the CARLA spectator to a driver-seat like pose on every tick,
    so the simulator window shows first-person view.
    """
    spectator = world.get_spectator()
    prev_loc = None
    prev_yaw = None
    while not stop_event.is_set() and vehicle.is_alive:
        try:
            v_tf = vehicle.get_transform()
            fwd = v_tf.get_forward_vector()
            up = v_tf.get_up_vector()
            right = v_tf.get_right_vector()
            offset = _add(_scale(fwd, x), _add(_scale(right, y), _scale(up, z)))
            target_loc = _add(v_tf.location, offset)
            target_yaw = v_tf.rotation.yaw
            if prev_loc is None:
                prev_loc = target_loc
            if prev_yaw is None:
                prev_yaw = target_yaw
            alpha = max(0.0, min(1.0, smooth))
            cam_loc = _lerp(prev_loc, target_loc, alpha)
            cam_yaw = _lerp_angle_deg(prev_yaw, target_yaw, alpha)
            spectator.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=v_tf.rotation.pitch, yaw=cam_yaw, roll=0.0)))
            prev_loc, prev_yaw = cam_loc, cam_yaw
            world.wait_for_tick(timeout=0.1)
        except Exception:
            # Keep trying until stop
            time.sleep(0.05)

def follow_third_person_spectator(world, vehicle, stop_event, back=6.0, up=2.5, right=0.0, smooth=0.25):
    """
    Positions the spectator a few meters behind and above the vehicle, following its heading
    for a classic third-person chase camera.
    """
    spectator = world.get_spectator()
    prev_loc = None
    prev_yaw = None
    while not stop_event.is_set() and vehicle.is_alive:
        try:
            v_tf = vehicle.get_transform()
            fwd = v_tf.get_forward_vector()
            upv = v_tf.get_up_vector()
            rightv = v_tf.get_right_vector()
            # Camera location = vehicle location - back * forward + up * up + right * right
            target_loc = _add(
                _add(v_tf.location, _scale(fwd, -abs(back))),
                _add(_scale(upv, abs(up)), _scale(rightv, right))
            )
            target_yaw = v_tf.rotation.yaw
            if prev_loc is None:
                prev_loc = target_loc
            if prev_yaw is None:
                prev_yaw = target_yaw
            alpha = max(0.0, min(1.0, smooth))
            cam_loc = _lerp(prev_loc, target_loc, alpha)
            cam_yaw = _lerp_angle_deg(prev_yaw, target_yaw, alpha)
            # Slightly pitch down to see the car
            cam_rot = carla.Rotation(pitch=v_tf.rotation.pitch - 5.0, yaw=cam_yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
            prev_loc, prev_yaw = cam_loc, cam_yaw
            world.wait_for_tick(timeout=0.1)
        except Exception:
            time.sleep(0.05)

# ------------------- Main controller logic -------------------
def controller_loop(world, groupA, groupB, stop_event):
    """
    Cycle traffic lights: groupA green -> yellow -> groupB green -> yellow
    """
    print("[INFO] Starting controller loop. Press Ctrl+C to stop.")
    while not stop_event.is_set():
        # GROUP A GREEN
        for tl in groupA:
            set_state(tl, carla.TrafficLightState.Green)
        for tl in groupB:
            set_state(tl, carla.TrafficLightState.Red)
        print("[STATE] A: GREEN | B: RED")
        t0 = time.time()
        while time.time() - t0 < GREEN_TIME and not stop_event.is_set():
            # could add adaptive logic using count_vehicles_near_light here
            time.sleep(0.5)
        if stop_event.is_set(): break

        # GROUP A YELLOW
        for tl in groupA:
            set_state(tl, carla.TrafficLightState.Yellow)
        print("[STATE] A: YELLOW")
        t0 = time.time()
        while time.time() - t0 < YELLOW_TIME and not stop_event.is_set():
            time.sleep(0.2)
        if stop_event.is_set(): break

        # GROUP B GREEN
        for tl in groupB:
            set_state(tl, carla.TrafficLightState.Green)
        for tl in groupA:
            set_state(tl, carla.TrafficLightState.Red)
        print("[STATE] B: GREEN | A: RED")
        t0 = time.time()
        while time.time() - t0 < GREEN_TIME and not stop_event.is_set():
            time.sleep(0.5)
        if stop_event.is_set(): break

        # GROUP B YELLOW
        for tl in groupB:
            set_state(tl, carla.TrafficLightState.Yellow)
        print("[STATE] B: YELLOW")
        t0 = time.time()
        while time.time() - t0 < YELLOW_TIME and not stop_event.is_set():
            time.sleep(0.2)

# ------------------- Entrypoint -------------------
def main():
    # CLI overrides for host/port and waits
    parser = argparse.ArgumentParser(description="CARLA 4-way intersection controller")
    parser.add_argument("--host", default=HOST, help="CARLA host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=PORT, help="CARLA port (default 2000)")
    parser.add_argument("--connect-timeout", type=float, default=2.0, help="Per-attempt timeout when connecting (s)")
    parser.add_argument("--retries", type=int, default=30, help="Number of connection attempts")
    parser.add_argument("--town", default=None, help="Load a specific small town (e.g., Town01 or Town03) for a simpler map")
    parser.add_argument("--drive", action='store_true', help="Spawn an ego vehicle you can drive (WASD)")
    parser.add_argument("--no-traffic", action='store_true', help="Do not spawn background vehicles")
    parser.add_argument("--camera", choices=["first", "third"], default="third", help="Camera mode when driving (default third)")
    parser.add_argument("--minimal-visuals", action='store_true', help="Try to hide buildings/props for a cleaner view (if supported)")
    parser.add_argument("--camera-smooth", type=float, default=0.25, help="Camera smoothing [0..1], higher is snappier (default 0.25)")
    parser.add_argument("--easy-drive", action='store_true', help="Gentler steering and throttle/brake response for easier control")
    parser.add_argument("--max-speed-kmh", type=float, default=25.0, help="Speed limiter in km/h (default 25)")
    parser.add_argument("--opendrive", type=str, default=None, help="Load a custom OpenDRIVE .xodr map (replaces current world)")
    args = parser.parse_args()

    # Make it immediate: default to drive mode and no traffic; only default town if no OpenDRIVE
    if args.town is None and not args.opendrive:
        args.town = "Town01"
    args.drive = True
    args.no_traffic = True

    spawned_vehicles = []
    groupA = []
    groupB = []
    world = None
    original_settings = None

    # Connect with retries to avoid immediate timeout
    client, world = wait_for_carla(args.host, args.port, per_try_timeout=args.connect_timeout, retries=args.retries)
    original_settings = world.get_settings()

    # Load OpenDRIVE map if provided, else load a smaller town if requested
    if args.opendrive:
        xodr_path = os.path.abspath(args.opendrive)
        if not os.path.exists(xodr_path):
            raise FileNotFoundError(f"OpenDRIVE file not found: {xodr_path}")
        print(f"[INFO] Loading OpenDRIVE map from {xodr_path} ...")
        previous_timeout = args.connect_timeout
        try:
            client.set_timeout(120.0)
            with open(xodr_path, 'r', encoding='utf-8') as f:
                xodr_text = f.read()
            # Generation parameters if available (CARLA 0.9.12+)
            gen_params = None
            try:
                gen_params = carla.OpendriveGenerationParameters()
                # Mild defaults for a flat minimal map
                gen_params.vertex_distance = 2.0
                gen_params.max_road_length = 50.0
                gen_params.wall_height = 0.0
                gen_params.additional_width = 0.0
                gen_params.smooth_junctions = True
                gen_params.enable_mesh_visibility = True
            except Exception:
                pass
            if gen_params is not None and hasattr(client, 'generate_opendrive_world'):
                world = client.generate_opendrive_world(xodr_text, gen_params)
            elif hasattr(client, 'generate_opendrive_world'):
                world = client.generate_opendrive_world(xodr_text)
            else:
                raise RuntimeError("This CARLA build does not support OpenDRIVE generation via client.generate_opendrive_world().")
            time.sleep(1.0)
            original_settings = world.get_settings()
            print("[INFO] OpenDRIVE map loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load OpenDRIVE map: {e}")
            print("[HINT] Ensure the .xodr is valid OpenDRIVE and matches your CARLA version. Falling back to existing world.")
        finally:
            client.set_timeout(previous_timeout)

    # Load a smaller town if requested (but not when using OpenDRIVE)
    if (not args.opendrive) and args.town:
        current_map = world.get_map().name if world.get_map() else ""
        if args.town not in current_map:
            print(f"[INFO] Loading map {args.town} (this can take a while the first time)...")
            # Temporarily increase timeout to allow map loading
            previous_timeout = args.connect_timeout
            try:
                client.set_timeout(120.0)
                world = client.load_world(args.town)
                # brief pause to allow map to initialize
                time.sleep(1.0)
                original_settings = world.get_settings()
            except Exception as e:
                print(f"[ERROR] Failed to load map '{args.town}': {e}")
                print("[HINT] Ensure the map exists in your CARLA build. Try --town Town01 or run without --town to keep the current map.")
                # Re-raise so the user sees the error if they supplied an invalid town
                raise
            finally:
                # Restore the shorter per-call timeout for subsequent RPCs
                client.set_timeout(previous_timeout)

    # Try to enable minimal visuals by toggling map layers (if available in this CARLA build)
    if getattr(args, 'minimal_visuals', False):
        try:
            amap = world.get_map()
            if hasattr(amap, 'layer_names') and hasattr(amap, 'toggle_layer'):  # newer APIs
                # Common visual layers that add clutter
                layers = [
                    'Buildings', 'Props', 'Foliage', 'Decals', 'Walls', 'ParkedVehicles', 'StreetLights'
                ]
                for layer in layers:
                    try:
                        if layer in amap.layer_names:
                            amap.toggle_layer(layer)  # hide if visible
                    except Exception:
                        pass
                print("[INFO] Minimal visuals attempted (hid map layers where supported).")
            else:
                print("[INFO] Minimal visuals not supported on this CARLA build; proceeding normally.")
        except Exception as e:
            print(f"[WARN] Minimal visuals failed: {e}")

    try:
        # find intersection lights
        if INTERSECTION_CENTER is None:
            center = None
        else:
            center = INTERSECTION_CENTER

        traffic_lights = find_intersection_lights(world, center=center, max_candidates=4, search_radius=60.0)

        # compute center (centroid) for grouping/spawning
        if len(traffic_lights) >= 1:
            centroid = carla.Location(
                x=sum(tl.get_transform().location.x for tl in traffic_lights)/len(traffic_lights),
                y=sum(tl.get_transform().location.y for tl in traffic_lights)/len(traffic_lights),
                z=sum(tl.get_transform().location.z for tl in traffic_lights)/len(traffic_lights)
            )
            print(f"[INFO] intersection centroid at {centroid}")
        else:
            # Fallback: use the first spawn point or derive from waypoints; else origin
            spawns = world.get_map().get_spawn_points()
            if not spawns:
                spawns = _fallback_spawn_points_from_waypoints(world)
            centroid = (spawns[0].location if spawns else carla.Location(x=0.0, y=0.0, z=0.0))
            print(f"[INFO] No traffic lights found; using fallback centroid at {centroid}")

        # If we found more than 4, trim to 4 closest
        if len(traffic_lights) > 4:
            traffic_lights = traffic_lights[:4]

        # group into two opposing groups
        if len(traffic_lights) >= 2:
            if len(traffic_lights) == 2:
                # easy: groupA is first, groupB is second
                groupA = [traffic_lights[0]]
                groupB = [traffic_lights[1]]
            else:
                groupA, groupB = group_lights_by_opposite(traffic_lights, centroid)

        if groupA or groupB:
            print("[INFO] Group A lamps:")
            for tl in groupA:
                print("  ", tl.id, tl.get_transform().location)
            print("[INFO] Group B lamps:")
            for tl in groupB:
                print("  ", tl.id, tl.get_transform().location)
        else:
            print("[INFO] No traffic light groups configured (map may not have signals). Skipping controller.")

        # set traffic lights to manual control if possible
        if groupA or groupB:
            set_manual_mode_for_lights(groupA + groupB, manual=True)

        # spawn vehicles on approaches
        blueprint_library = world.get_blueprint_library()
        spawn_points = select_spawn_points_for_approaches(world, centroid, radius=APPROACH_RADIUS)
        print(f"[INFO] Found {len(spawn_points)} candidate spawn points for approaches.")
        if not args.no_traffic:
            spawned_vehicles = spawn_vehicles(world, blueprint_library, spawn_points, NUM_VEHICLES, centroid)
            print(f"[INFO] Spawned {len(spawned_vehicles)} vehicles.")
        else:
            print("[INFO] Skipping background traffic spawn (--no-traffic)")

        # controller thread (only if running traffic lights scenario)
        stop_event = threading.Event()
        controller_t = None
        if (not args.drive) and (groupA or groupB):
            controller_t = threading.Thread(target=controller_loop, args=(world, groupA, groupB, stop_event), daemon=True)
            controller_t.start()

        # main monitoring loop: prints queue lengths every few seconds
        try:
            if args.drive:
                # spawn ego and let user drive
                ego = spawn_ego_vehicle(world, blueprint_library, near_location=centroid)
                print(f"[INFO] Spawned ego vehicle id={ego.id}")
                # Choose camera mode
                # If easy-drive and user left default camera-smooth, make it a bit smoother
                cam_smooth = args.camera_smooth
                if args.easy_drive and abs(cam_smooth - 0.25) < 1e-6:
                    cam_smooth = 0.18
                if args.camera == 'first':
                    cam_thread = threading.Thread(target=follow_first_person_spectator, args=(world, ego, stop_event), kwargs={"smooth": cam_smooth}, daemon=True)
                else:
                    cam_thread = threading.Thread(target=follow_third_person_spectator, args=(world, ego, stop_event), kwargs={"smooth": cam_smooth}, daemon=True)
                cam_thread.start()
                # Optional: also attach an RGB sensor that records frames to disk
                # camera = attach_first_person_camera(world, ego)
                try:
                    drive_loop_keyboard(ego, stop_event, max_speed_kmh=args.max_speed_kmh, easy=args.easy_drive)
                finally:
                    stop_event.set()
                    try:
                        ego.destroy()
                    except Exception:
                        pass
                    # try:
                    #     camera.destroy()
                    # except Exception:
                    #     pass
            else:
                while True:
                    qA = sum(count_vehicles_near_light(world, tl, QUEUE_DISTANCE) for tl in groupA)
                    qB = sum(count_vehicles_near_light(world, tl, QUEUE_DISTANCE) for tl in groupB)
                    print(f"[MONITOR] Queue length A={qA} | B={qB} | Vehicles total={len(world.get_actors().filter('vehicle.*'))}")
                    time.sleep(4.0)
        except KeyboardInterrupt:
            print("[INFO] Interrupted. Stopping controller.")
            stop_event.set()
            if controller_t:
                controller_t.join(timeout=5.0)
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
    finally:
        # reset traffic lights to automatic/manual off
        try:
            if groupA or groupB:
                set_manual_mode_for_lights(groupA + groupB, manual=False)
        except Exception:
            pass
        # destroy spawned vehicles
        try:
            for v in spawned_vehicles:
                v.destroy()
        except Exception:
            pass
        # restore original settings if changed
        try:
            if world and original_settings is not None:
                world.apply_settings(original_settings)
        except Exception:
            pass
        print("[INFO] Cleaned up, exiting.")

if __name__ == '__main__':
    main()
