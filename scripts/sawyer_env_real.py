import robosuite as suite
from robosuite.controllers import load_composite_controller_config
# from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.stack import Stack
import numpy as np
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import inspect
root_folder = Path(__file__).parent.parent

robot_name = "Sawyer"
controller_path = "./sawyer_abs_pose.json"
controller_config = load_composite_controller_config(robot=robot_name, controller=str(controller_path))

print(controller_config)
env = suite.make(
    env_name="Stack",
    robots=robot_name,
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    control_freq=20,
    horizon=600,
    render_camera=None,
)

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

def has_reached_target(target_pos, target_rotm, epsilon_pos=0.0028, epsilon_rot=0.01):
    current_pos = env.sim.data.body_xpos[eef_id] - pos_base
    current_rotm = env.sim.data.body_xmat[eef_id].reshape(3, 3)

    pos_err = np.linalg.norm(current_pos - target_pos)
    # rot_err = np.linalg.norm(R.from_matrix(current_rotm).as_rotvec() - R.from_matrix(target_rotm).as_rotvec())
    rot_err = R.from_matrix(current_rotm.T @ target_rotm).magnitude()
    print("pos err: ",pos_err)
    print("rot err: ",rot_err)
    return pos_err < epsilon_pos and rot_err < epsilon_rot

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
            [0, 1, 0],
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

def move_to_target(body_id, steps, rotm_basic, height_offset=0, gripper=-1):
    for i in range(steps):
        pos = get_pos(body_id)
        rotm = get_rot(body_id, rotm_basic)
        pos[2] += height_offset
        print(f"---- STEP {i} ----")
        if not has_reached_target(pos, rotm):
            action[0:3] = pos
            action[3:6] = R.from_matrix(rotm).as_rotvec()
            action[6] = gripper
            print("actual cube A is in ", pos_cubeA)
            print("now eef is in ", env.sim.data.body_xpos[eef_id])
            print("actual cube B is in ", pos_cubeB)
            print("has reached target: ", has_reached_target(pos,rotm))
            print("action: ", action)

            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.05)
        else:
            break

def grasp(body_id, steps, rotm_basic, height_offset=0, gripper=-1):
    for i in range(steps):
        pos = get_pos(body_id)
        rotm = get_rot(body_id, rotm_basic)
        pos[2] += height_offset

        action[0:3] = pos
        action[3:6] = R.from_matrix(rotm).as_rotvec()
        action[12] = gripper

        obs, reward, done, info = env.step(action)
        env.render()

        
        time.sleep(0.05)

# Move and place
rotm_basic = get_rotm_basic_4A()

#Move above cubA
move_to_target(cubeA_main_id, 200, rotm_basic, height_offset=APPROACH_DISTANCE)