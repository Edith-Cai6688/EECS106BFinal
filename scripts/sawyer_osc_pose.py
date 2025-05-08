import robosuite as suite
from robosuite.controllers import load_composite_controller_config
# from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.stack import Stack
import numpy as np
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import inspect
import matplotlib.pyplot as plt
root_folder = Path(__file__).parent.parent
import os

robot_name = "Sawyer"
controller_path = "./new_config.json"
controller_config = load_composite_controller_config(robot=robot_name, controller=str(controller_path))

print(controller_config)
env = suite.make(
    env_name="Stack",
    robots=robot_name,
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    camera_names=["agentview"],
    camera_heights=256,
    camera_widths=256,
    use_camera_obs=True,
    control_freq=20,
    horizon=20000,
    render_camera=None,
)
pos_err_history = {"x": [], "y": [], "z": []}
rot_err_history = []

obs = env.reset()
cubeA_main_id = env.sim.model.body_name2id("cubeA_main")
cubeB_main_id = env.sim.model.body_name2id("cubeB_main")
eef_id = env.sim.model.body_name2id("gripper0_right_eef")
base_id = env.sim.model.body_name2id("robot0_base")

pos_base = env.sim.data.body_xpos[base_id]
rot_base = env.sim.data.body_xmat[base_id].reshape(3, 3)
pos_cubeA = env.sim.data.body_xpos[cubeA_main_id]
print("actual cube A is in ", pos_cubeA)
pos_cubeB = env.sim.data.body_xpos[cubeB_main_id]
print("actual cube B is in ", pos_cubeB)
pos_eef = env.sim.data.body_xpos[eef_id]
rot_cubeA = env.sim.data.body_xmat[cubeA_main_id].reshape(3, 3)

action = np.zeros(env.action_spec[0].shape[0])
APPROACH_DISTANCE = 0.1
GRASP_HEIGHT = 0.05

def get_pos(body_id):
    return env.sim.data.body_xpos[body_id] - pos_base

def get_rot(body_id, rotm_basic):
    return env.sim.data.body_xmat[body_id].reshape((3,3)) @ rotm_basic

# === move to target helper function ===

def has_reached_target(target_pos, target_rotm, epsilon_pos=0.0075, epsilon_rot=0.03):
    current_pos = env.sim.data.body_xpos[eef_id] - pos_base
    current_rotm = env.sim.data.body_xmat[eef_id].reshape(3, 3)

    pos_err = np.linalg.norm(current_pos - target_pos)
    # rot_err = np.linalg.norm(R.from_matrix(current_rotm).as_rotvec() - R.from_matrix(target_rotm).as_rotvec())
    rot_err = R.from_matrix(current_rotm.T @ target_rotm).magnitude()
    return pos_err < epsilon_pos and rot_err < epsilon_rot

def has_reached_target_pos(target_pos, epsilon_pos=0.0075):
    current_pos = env.sim.data.body_xpos[eef_id] - pos_base
    pos_err = np.linalg.norm(current_pos - target_pos)
    return pos_err < epsilon_pos

def compute_reward(target_pos, target_rotm, reached_thresh=0.0075, rot_thresh=0.03):
    current_pos = env.sim.data.body_xpos[eef_id] - pos_base
    current_rotm = env.sim.data.body_xmat[eef_id].reshape(3, 3)

    # Compute errors
    pos_err = np.linalg.norm(current_pos - target_pos)
    rot_err = R.from_matrix(current_rotm.T @ target_rotm).magnitude()

    reward = - (10.0 * pos_err + 2.0 * rot_err)

    if pos_err < reached_thresh and rot_err < rot_thresh:
        reward += 10.0

    return reward

def get_rotm_basic_4A():
    posA = get_pos(cubeA_main_id)
    posB = get_pos(cubeB_main_id)
    rotA = env.sim.data.body_xmat[cubeA_main_id].reshape(3, 3)
    eef_rotm_now = env.sim.data.body_xmat[eef_id].reshape(3, 3)

    local_x_dir = rotA[:, 0]
    local_y_dir = rotA[:, 1]

    vec_to_B = posB - posA

    projected_x = np.abs(np.dot(vec_to_B, local_x_dir))
    projected_y = np.abs(np.dot(vec_to_B, local_y_dir))

    if projected_x > projected_y:
        preferred_dir = local_x_dir
        alternative_rotm = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
    else:
        preferred_dir = local_y_dir
        alternative_rotm = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

    eef_forward_dir = eef_rotm_now[:, 0]  
    cos_angle = np.dot(preferred_dir, eef_forward_dir)

    if cos_angle < 0:
        alternative_rotm[:, 0] *= -1
        alternative_rotm[:, 1] *= -1

    return alternative_rotm

def get_rotm_basic_4B():
    rotB = env.sim.data.body_xmat[cubeB_main_id].reshape(3, 3)
    local_x_dir = rotB[:, 0]
    local_y_dir = rotB[:, 1]
    eef_rotm_now = env.sim.data.body_xmat[eef_id].reshape(3, 3)
    eef_forward_dir = eef_rotm_now[:, 0]

    cos_x = np.abs(np.dot(eef_forward_dir, local_x_dir))
    cos_y = np.abs(np.dot(eef_forward_dir, local_y_dir))

    if cos_x > cos_y:
        rotm_basic = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
    else:
        rotm_basic = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

    return rotm_basic

def add_noise(pos, rot_vec, pos_noise=0.005, rot_noise=0.02):
    noisy_pos = pos + np.random.uniform(-pos_noise, pos_noise, size=3)
    noisy_rot = rot_vec + np.random.uniform(-rot_noise, rot_noise, size=3)
    return noisy_pos, noisy_rot



def is_robot_lost_control(target_pos, target_rotm, pos_threshold=3, rot_threshold=10):
    current_pos = env.sim.data.body_xpos[eef_id] - pos_base
    current_rotm = env.sim.data.body_xmat[eef_id].reshape(3, 3)

    pos_err = np.linalg.norm(current_pos - target_pos)
    rot_err = np.linalg.norm(R.from_matrix(current_rotm).as_rotvec() - R.from_matrix(target_rotm).as_rotvec())

    if pos_err > pos_threshold or rot_err > rot_threshold:
        return True
    else:
        return False

def constant_dist_waypoints(start_p, goal_p, step=0.05):
    vec = goal_p - start_p
    dist = np.linalg.norm(vec)
    n_seg = int(np.ceil(dist / step))
    ans = [start_p + (i / n_seg) * vec for i in range(n_seg + 1)]
    print(f"NUMBER OF WAYPOINTS: {len(ans)}, CENTERED {step} M apart")
    return [start_p + (i / n_seg) * vec for i in range(n_seg + 1)]
wp_offset_err = []

def constant_dist_R_waypoints(start_rotm, goal_rotm, n_seg):
    R_start = R.from_matrix(start_rotm)
    R_goal = R.from_matrix(goal_rotm)
    key_times = [0, 1]
    slerp_fn = R.Slerp(key_times, R.from_matrix([start_rotm, goal_rotm]))
    ts = np.linspace(0, 1, n_seg + 1)
    return [r.as_rotvec() for r in slerp_fn(ts)]

def rotvec_waypoints(start_rotm, goal_rotm, n_seg, eps=1e-8):
    """
    Manual SLERP between two rotation matrices, returning
    n_seg+1 rotation‑vectors (incl. start & goal).

    Works on any SciPy ≥ 1.2 (no R.Slerp needed).
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    q0 = R.from_matrix(start_rotm).as_quat()   # [x y z w]
    q1 = R.from_matrix(goal_rotm).as_quat()

    # Ensure shortest‑path interpolation
    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    def slerp_scalar(t, q0, q1):
        cos_theta = np.dot(q0, q1)
        if 1.0 - cos_theta < eps:
            q = (1 - t) * q0 + t * q1
            return q / np.linalg.norm(q)
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta)
        w0 = np.sin((1 - t) * theta) / sin_theta
        w1 = np.sin(t * theta) / sin_theta
        return w0 * q0 + w1 * q1

    ts = np.linspace(0.0, 1.0, n_seg + 1)
    return [
        R.from_quat(slerp_scalar(t, q0, q1)).as_rotvec()
        for t in ts
    ]


def move_to_target_constant_wp(body_id, steps_per_wp, rotm_basic, height_offset=0, gripper=-1, step_size=0.05, save_dir = "./save_trajs", traj_index = 0):
    image_observations = []
    actions_taken = []
    origin = env.sim.data.body_xpos[eef_id] - pos_base
    destination = get_pos(body_id)
    destination[2] += height_offset
    waypoints = constant_dist_waypoints(origin, destination, step=step_size)
    goal_rotm = get_rot(body_id, rotm_basic)
    print(waypoints[-1], destination)
    rotvec_wps = rotvec_waypoints(
        env.sim.data.body_xmat[eef_id].reshape(3, 3),
        goal_rotm,
        n_seg=len(waypoints) - 1
    )
    obs = env.reset()
    image_observations.append(obs["agentview_image"])
    for wp_idx, (wp, rv_wp) in enumerate(zip(waypoints, rotvec_wps)):
        for step in range(steps_per_wp):
            if has_reached_target(wp, goal_rotm, epsilon_pos=0.0025, epsilon_rot=0.03):
                break
            action[0:3] = wp
            action[3:6] = rv_wp
            action[6] = gripper

            eef_pos = env.sim.data.body_xpos[eef_id] - pos_base
            eef_rot = env.sim.data.body_xmat[eef_id].reshape(3, 3)

            pos_error_vec = eef_pos - destination
            rot_error_vec = R.from_matrix(eef_rot.T @ goal_rotm).magnitude()
            pos_err_history["x"].append(pos_error_vec[0])
            pos_err_history["y"].append(pos_error_vec[1])
            pos_err_history["z"].append(pos_error_vec[2])

            rot_err_history.append(rot_error_vec)
            # print("actual cube A is in ", pos_cubeA)
            # print("now eef is in ", env.sim.data.body_xpos[eef_id])
            # print("actual cube B is in ", pos_cubeB)
            # print("has reached target: ", has_reached_target(wp,goal_rotm))
            # print("action: ", action)

            obs, reward, done, info = env.step(action)
            image_observations.append(obs["agentview_image"])
            actions_taken.append(action.copy())
            # reward = compute_reward(np.concatenate([noisy_pos[0:2], [pos[2]]]), rotm)
            # print(f'------------\n {reward} \n ----------=')
            env.render()
            time.sleep(0.05)
        if has_reached_target(destination, goal_rotm):
            break
        eef_pos = env.sim.data.body_xpos[eef_id] - pos_base
        pos_error_vec = eef_pos - wp
        pos_error_vec_norm = np.linalg.norm(pos_error_vec)
        wp_offset_err.append(pos_error_vec_norm)
        print(f"OFF OF WAYPOINT BY {pos_error_vec_norm:4f}")
    np.savez_compressed(os.path.join(save_dir, f"traj{traj_index}.npz"), observations=np.array(image_observations), actions=np.array(actions_taken))
    print(f"Saved: {save_dir}/traj{traj_index}.npz")
    # plt.figure(figsize=(8, 4))
    # plt.plot(wp_offset_err, marker='o')
    # plt.title("Miss distance at each waypoint")
    # plt.xlabel("Waypoint index")
    # plt.ylabel("Distance from waypoint (m)")
    # plt.grid(True)
    # plt.show()


def move_to_target_linear(body_id, steps_per_wp, rotm_basic, height_offset=0, gripper=-1, n_wp = 10):
    origin = env.sim.data.body_xpos[eef_id] - pos_base
    destination = get_pos(body_id)
    destination[2] += height_offset
    waypoints = np.linspace(origin, destination, n_wp + 1)
    goal_rotm = get_rot(body_id, rotm_basic)

    for wp_idx, wp in enumerate(waypoints):
        for step in range(steps_per_wp):
            if has_reached_target(wp, goal_rotm):
                break
            action[0:3] = wp
            action[3:6] = R.from_matrix(goal_rotm).as_rotvec()
            action[6] = gripper

            eef_pos = env.sim.data.body_xpos[eef_id] - pos_base
            eef_rot = env.sim.data.body_xmat[eef_id].reshape(3, 3)

            pos_error_vec = eef_pos - destination
            rot_error_vec = R.from_matrix(eef_rot.T @ goal_rotm).magnitude()
            pos_err_history["x"].append(pos_error_vec[0])
            pos_err_history["y"].append(pos_error_vec[1])
            pos_err_history["z"].append(pos_error_vec[2])

            rot_err_history.append(rot_error_vec)
            print("actual cube A is in ", pos_cubeA)
            print("now eef is in ", env.sim.data.body_xpos[eef_id])
            print("actual cube B is in ", pos_cubeB)
            print("has reached target: ", has_reached_target(wp,goal_rotm))
            print("action: ", action)

            obs, reward, done, info = env.step(action)
            # reward = compute_reward(np.concatenate([noisy_pos[0:2], [pos[2]]]), rotm)
            # print(f'------------\n {reward} \n ----------=')
            env.render()
            time.sleep(0.05)



def move_to_target_bezier(body_id, steps_per_wp, rotm_basic, height_offset=0, gripper=-1, n_wp = 5, ctrl_offset=np.array([0,0,0.05])):
    p0 = env.sim.data.body_xpos[eef_id] - pos_base
    p3 = get_pos(body_id)

    waypoints = bezier_waypoints(p0, p3, n=n_wp, ctrl_offset=ctrl_offset)
    goal_rotm = get_rot(body_id, rotm_basic)
    for wp_idx, wp in enumerate(waypoints):
        for step in range(steps_per_wp):
            if has_reached_target(wp, goal_rotm):
                break
            action[0:3] = wp
            action[3:6] = R.from_matrix(goal_rotm).as_rotvec()
            action[6] = gripper

            eef_pos = env.sim.data.body_xpos[eef_id] - pos_base
            eef_rot = env.sim.data.body_xmat[eef_id].reshape(3, 3)

            pos_error_vec = eef_pos - p3
            rot_error_vec = R.from_matrix(eef_rot.T @ goal_rotm).magnitude()
            pos_err_history["x"].append(pos_error_vec[0])
            pos_err_history["y"].append(pos_error_vec[1])
            pos_err_history["z"].append(pos_error_vec[2])

            rot_err_history.append(rot_error_vec)
            print("actual cube A is in ", pos_cubeA)
            print("now eef is in ", env.sim.data.body_xpos[eef_id])
            print("actual cube B is in ", pos_cubeB)
            print("has reached target: ", has_reached_target(wp,goal_rotm))
            print("action: ", action)

            obs, reward, done, info = env.step(action)
            # reward = compute_reward(np.concatenate([noisy_pos[0:2], [pos[2]]]), rotm)
            # print(f'------------\n {reward} \n ----------=')
            env.render()
            time.sleep(0.05)




def move_to_target(body_id, steps, rotm_basic, height_offset=0, gripper=-1):
    for i in range(steps):
        pos = get_pos(body_id)
        rotm = get_rot(body_id, rotm_basic)
        pos[2] += height_offset
        print(f"---- STEP {i} ----")
        if not has_reached_target(pos, rotm):
            noisy_pos, _ = add_noise(pos, R.from_matrix(rotm).as_rotvec(), pos_noise=0.01, rot_noise=0)
            
            action[0:3] = pos 
            action[3:6] = R.from_matrix(rotm).as_rotvec()
            # action[0:2] = noisy_pos[0:2]
            # action[2] = pos[2]
            # action[3:6] = R.from_matrix(rotm).as_rotvec()
            action[6] = gripper
            eef_pos = env.sim.data.body_xpos[eef_id] - pos_base
            eef_rot = env.sim.data.body_xmat[eef_id].reshape(3, 3)

            pos_error_vec = eef_pos - pos
            rot_error_vec = R.from_matrix(eef_rot.T @ rotm).magnitude()

            # Log each component
            pos_err_history["x"].append(pos_error_vec[0])
            pos_err_history["y"].append(pos_error_vec[1])
            pos_err_history["z"].append(pos_error_vec[2])

            rot_err_history.append(rot_error_vec)
            print("actual cube A is in ", pos_cubeA)
            print("now eef is in ", env.sim.data.body_xpos[eef_id])
            print("actual cube B is in ", pos_cubeB)
            print("has reached target: ", has_reached_target(pos,rotm))
            print("action: ", action)

            obs, reward, done, info = env.step(action)
            # reward = compute_reward(np.concatenate([noisy_pos[0:2], [pos[2]]]), rotm)
            # print(f'------------\n {reward} \n ----------=')
            env.render()
            time.sleep(0.05)
        else:
            print("has reached target: ", has_reached_target(pos, rotm))
            break

def grasp(body_id, steps, rotm_basic, height_offset=0, gripper=-1):
    for i in range(steps):
        pos = get_pos(body_id)
        rotm = get_rot(body_id, rotm_basic)
        pos[2] += height_offset

        action[0:3] = pos
        action[3:6] = R.from_matrix(rotm).as_rotvec()
        action[6] = gripper

        obs, reward, done, info = env.step(action)
        env.render()

        
        time.sleep(0.05)

def bezier_waypoints(p0, p3, n=20, ctrl_offset=np.array([0, 0, 0.05])):
    p1 = p0 + ctrl_offset
    p2 = p3 + ctrl_offset

    ts = np.linspace(0.0, 1.0, n + 1)
    wp = []
    for t in ts:
        b0 = (1 - t) ** 3
        b1 = 3 * (1 - t) ** 2 * t
        b2 = 3 * (1 - t) * t ** 2
        b3 = t ** 3
        wp.append((b0 * p0) + (b1 * p1) + (b2 * p2) + (b3 * p3))
    return wp

def collect_multiple_image_trajectories(num_trajectories=40, save_dir="./dataset/train"):
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_trajectories):
        print(f"\n=== Collecting Trajectory {i} ===")

        # Reset environment and get rot matrix
        obs = env.reset()
        rotm_basic = get_rotm_basic_4A()

        # Re-initialize cube/body positions if needed
        # (Optional: randomize cube poses here for diversity)

        move_to_target_constant_wp(
            body_id=cubeA_main_id,
            steps_per_wp=200,
            rotm_basic=rotm_basic,
            height_offset=APPROACH_DISTANCE,
            step_size=0.05,
            traj_index=i,
            save_dir=save_dir
        )

# Move and place
rotm_basic = get_rotm_basic_4A()


collect_multiple_image_trajectories(num_trajectories=50)


#Move above cubA
# move_to_target_constant_wp(cubeA_main_id, 200, rotm_basic, height_offset=APPROACH_DISTANCE, step_size = 0.05)

# #move close to cubeA
# move_to_target_constant_wp(cubeA_main_id, 100, rotm_basic)

# # grasp cubeA
# grasp(cubeA_main_id, 30, rotm_basic, gripper = 1)

# # move above cubeA
# move_to_target_constant_wp(cubeA_main_id, 30, rotm_basic, height_offset=APPROACH_DISTANCE, gripper = 1)

# # choose the grapper direction for B
# rotm_basic = get_rotm_basic_4B()

# # move above cubeB
# move_to_target_constant_wp(cubeB_main_id, 200, rotm_basic, APPROACH_DISTANCE, gripper = 1)

# #move close to cubeB
# move_to_target_constant_wp(cubeB_main_id, 100, rotm_basic, GRASP_HEIGHT, gripper = 1)

# #drop cubeA
# grasp(cubeB_main_id, 30, rotm_basic, GRASP_HEIGHT, gripper = -1)

# # move above
# move_to_target_constant_wp(cubeB_main_id, 30, rotm_basic, APPROACH_DISTANCE, gripper = -1)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot position errors
axs[0].plot(pos_err_history["x"], label="Position X")
axs[0].plot(pos_err_history["y"], label="Position Y")
axs[0].plot(pos_err_history["z"], label="Position Z")
axs[0].set_title("EEF Position Error Components")
axs[0].set_ylabel("Position Error (m)")
axs[0].legend()
axs[0].grid(True)

# Plot rotation errors
axs[1].plot(rot_err_history, label="Rotation")

axs[1].set_title("EEF Rotation Error")
axs[1].set_xlabel("Timestep")
axs[1].set_ylabel("Rotation Error (rad)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


input("Press enter to close")