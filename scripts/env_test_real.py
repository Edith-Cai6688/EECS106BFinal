import time

import numpy as np
import matplotlib.pyplot as plt

import robosuite
from robosuite.controllers import load_composite_controller_config
from pathlib import Path
root_folder = Path(__file__).parent.parent
from scipy.spatial.transform import Rotation as R
from robosuite_models.robots import indy7_robot


robot_name = "Indy7"

# create controller
controller_path = root_folder/"scripts"/"indy7_absolute_pose.json"
controller_config = load_composite_controller_config(robot = robot_name, controller=str(controller_path))
# print(controller_config)

env = robosuite.make(
    "StackCustom",
    robots=[robot_name],
    controller_configs=controller_config,   # arms controlled via OSC, other parts via JOINT_POSITION/JOINT_VELOCITY
    has_renderer=True,                      # on-screen rendering
    render_camera=None,              # visualize the "frontview" camera
    has_offscreen_renderer=True,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=600,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # no observations needed
    use_camera_obs=True,
    camera_depths=True,
    camera_segmentations="instance",
    camera_names= ["agentview", "robot0_eye_in_hand"],

)
# reset the environment
env.reset()

"""
possible body names:
'world', 'table', 'left_eef_target', 'right_eef_target', 
'robot0_base', 'robot0_link0', 'robot0_link1', 'robot0_link2', 'robot0_link3', 'robot0_link4', 'robot0_link5', 'robot0_link6', 'robot0_link7', 
'robot0_right_hand', 'gripper0_right_right_gripper', 'gripper0_right_eef', 'gripper0_right_leftfinger', 'gripper0_right_finger_joint1_tip', 
'gripper0_right_rightfinger', 'gripper0_right_finger_joint2_tip', 'fixed_mount0_base', 'fixed_mount0_controller_box', 
'fixed_mount0_pedestal_feet', 'fixed_mount0_torso', 'fixed_mount0_pedestal', 'cubeA_main', 'cubeB_main'
"""
# cubeA_main_id = env.sim.model.body_name2id("cubeA_main")
# pos_cubeA = env.sim.data.body_xpos[cubeA_main_id]
# print("Cube A position: ", pos_cubeA)
# rotm_cubeA = env.sim.data.body_xmat[cubeA_main_id].reshape((3,3)) # rotation matrix
# quat_cubeA = env.sim.data.body_xquat[cubeA_main_id] # quaternion in wxyz format
# rotm_from_quat_cubeA = R.from_quat(quat_cubeA, scalar_first = True).as_matrix() # rotation matrix from quaternion

# print("confirm both are the same:", rotm_cubeA, rotm_from_quat_cubeA)

# print(env.action_spec[0].shape)

# === basic body IDs and variances
cubeA_main_id = env.sim.model.body_name2id("cubeA_main")
cubeB_main_id = env.sim.model.body_name2id("cubeB_main")
eef_id = env.sim.model.body_name2id("gripper0_right_eef")
base_id = env.sim.model.body_name2id('robot0_base')

pos_base = env.sim.data.body_xpos[base_id]
rot_base = env.sim.data.body_xmat[base_id].reshape(3, 3)
pos_cubeA = env.sim.data.body_xpos[cubeA_main_id]
pos_cubeB = env.sim.data.body_xpos[cubeB_main_id]
rot_cubeA = env.sim.data.body_xmat[cubeA_main_id].reshape(3, 3)

action = np.zeros(env.action_spec[0].shape[0])
APPROACH_DISTANCE = 0.1
GRASP_HEIGHT = 0.05


# === get position helper function ===
def get_pos(body_id):
    # postion(relative to the base)
    pos = env.sim.data.body_xpos[body_id] - pos_base
    return pos


def get_rot(body_id, rotm_basic):
    # rotation
    rotm = env.sim.data.body_xmat[body_id].reshape((3,3)) @ rotm_basic
    return rotm


# === move to target helper function ===

def has_reached_target(target_pos, target_rotm, epsilon_pos=0.0028, epsilon_rot=0.01):
    current_pos = env.sim.data.body_xpos[eef_id] - pos_base
    current_rotm = env.sim.data.body_xmat[eef_id].reshape(3, 3)

    pos_err = np.linalg.norm(current_pos - target_pos)
    rot_err = np.linalg.norm(R.from_matrix(current_rotm).as_rotvec() - R.from_matrix(target_rotm).as_rotvec())

    print("pos",pos_err)
    print("rot",rot_err)
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


def move_to_target(body_id, steps, rotm_basic, height_offset = 0, gripper = -1):
    for i in range(steps):
        pos = get_pos(body_id)
        rotm = get_rot(body_id, rotm_basic)
        pos[2] += height_offset

        if (has_reached_target(pos,rotm) != 1):
            noisy_pos, noisy_rot = add_noise(pos, R.from_matrix(rotm).as_rotvec())
            action[0:3] = noisy_pos
            action[3:6] = noisy_rot
            # action[0:3] = pos
            # action[3:6] = R.from_matrix(rotm).as_rotvec()
            action[12] = gripper

            obs, reward, done, info = env.step(action)
            env.render()

            # # Show the image, depth, segmentation
            # for cam in env.camera_names:
            #     # plt.figure()
            #     fig, axes = plt.subplots(1, 3)
            #     axes[0].imshow(np.rot90(obs[cam + "_image"], 2))
            #     axes[1].imshow(np.rot90(obs[cam + "_depth"], 2), cmap='gray')
            #     axes[2].imshow(np.rot90(obs[cam + "_segmentation_instance"], 2), cmap='grey')
            #     for ax in axes:
            #         ax.axis('off')
            #     fig.suptitle(cam)
            # plt.show()
            if is_robot_lost_control(noisy_pos, rotm):
                raise Exception("Robot lost control!")

            time.sleep(0.05)
        else:
            break

def grasp(body_id, steps, rotm_basic, height_offset = 0, gripper = -1):
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
    
        
        



# === Move and Place Logical ===

# choose the grapper direction cubeA
rotm_basic = get_rotm_basic_4A()

# move above cubeA
move_to_target(cubeA_main_id, 100, rotm_basic, APPROACH_DISTANCE)

# move close to cubeA
move_to_target(cubeA_main_id, 100, rotm_basic)

# grasp cubeA
grasp(cubeA_main_id, 10, rotm_basic, gripper = 1)

# move above cubeA
move_to_target(cubeA_main_id, 20, rotm_basic, APPROACH_DISTANCE, gripper = 1)

# choose the grapper direction for B
rotm_basic = get_rotm_basic_4B()

# move above cubeB
move_to_target(cubeB_main_id, 100, rotm_basic, APPROACH_DISTANCE, gripper = 1)

# move close to cubeB
move_to_target(cubeB_main_id, 100, rotm_basic, GRASP_HEIGHT, gripper = 1)

#drop cubeA
grasp(cubeB_main_id, 10, rotm_basic, GRASP_HEIGHT, gripper = -1)

# move above
move_to_target(cubeB_main_id, 30, rotm_basic, APPROACH_DISTANCE, gripper = -1)



# for i in range(200):
#     if i < 60:
#         action = np.zeros((env.action_spec[0].shape[0],))
#         action[0:3] = np.array([0.3, 0.1, 0.0]) # Desired position to go in meter
#         action[3:6] = R.from_matrix(rotm_basic).as_rotvec() # Desired orientation to go in rotation vector representation
#         action[6] = -1
#     elif i < 120:
#         action = np.zeros((env.action_spec[0].shape[0],))
#         action[0:3] = np.array([0.5, 0.1, 0.0])
#         action[3:6] = R.from_matrix(rotm_basic).as_rotvec() # Desired orientation to go in rotation vector representation

#     else:
#         action = np.zeros((env.action_spec[0].shape[0],))
#         action[0:3] = np.array([0.7, 0.1, 0.0])
#         action[3:6] = R.from_matrix(rotm_basic).as_rotvec()
#         action[6] = -1

#     print(i, action[0:3])

#     obs, reward, done, info = env.step(action)  # take action in the environment

#     time.sleep(0.05)

#     env.render()  # render on display
#     if i == 0:
#         for key in obs.keys():
#             try:
#                 print(key, obs[key].shape)
#             except:
#                 print(key, obs[key])
#         time.sleep(3)

# eef_quat = obs['robot0_eef_quat'] # quaternion in xyzw format
# # if you want to convert it into the rotation matrix
# eef_rotm = R.from_quat(eef_quat, scalar_first = False).as_matrix()

# print(obs['robot0_eef_quat'])


