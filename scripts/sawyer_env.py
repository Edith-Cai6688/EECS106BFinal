import robosuite as suite
from robosuite.controllers import load_composite_controller_config
# from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.stack import Stack
import numpy as np
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import inspect
import os
import matplotlib.pyplot as plt
root_folder = Path(__file__).parent.parent


# # create controller
robot_name = "Sawyer"
controller_path = "./indy7_absolute_pose.json"
controller_config = load_composite_controller_config(robot = robot_name, controller=str(controller_path))
# # print(controller_config)

print(controller_config)

env = suite.make(
    env_name="Stack",                   # 任务类型：Stacking
    robots="Sawyer",                    # 使用 Sawyer 机械臂
    controller_configs=controller_config,
    has_renderer=True,                  # 实时渲染
    has_offscreen_renderer=True,       # 不保存图像帧
    use_camera_obs=True,               # 暂不使用图像观察
    camera_names=["agentview"],
    camera_heights=256,
    camera_widths=256,
    control_freq=20,                    # 控制频率
    horizon=600,                        # 每条 trajectory 长度
    render_camera=None,          # 渲染视角
)

pos_err_history = {"x": [], "y": [], "z": []}
rot_err_history = []



# 重置环境
# env = GymWrapper(env)
obs = env.reset()
# OBJECTS LIST
#   0: world
#   1: table
#   2: left_eef_target
#   3: right_eef_target
#   4: robot0_base
#   5: robot0_right_arm_base_link
#   6: robot0_right_l0
#   7: robot0_head
#   8: robot0_screen
#   9: robot0_head_camera
#  10: robot0_right_torso_itb
#  11: robot0_right_l1
#  12: robot0_right_l2
#  13: robot0_right_l3
#  14: robot0_right_l4
#  15: robot0_right_arm_itb
#  16: robot0_right_l5
#  17: robot0_right_hand_camera
#  18: robot0_right_wrist
#  19: robot0_right_l6
#  20: robot0_right_hand
#  21: gripper0_right_gripper_base
#  22: gripper0_right_eef
#  23: gripper0_right_l_finger
#  24: gripper0_right_l_finger_tip
#  25: gripper0_right_r_finger
#  26: gripper0_right_r_finger_tip
#  27: robot0_right_l4_2
#  28: robot0_right_l2_2
#  29: robot0_right_l1_2
#  30: fixed_mount0_base
#  31: fixed_mount0_controller_box
#  32: fixed_mount0_pedestal_feet
#  33: fixed_mount0_torso
#  34: fixed_mount0_pedestal
#  35: cubeA_main
#  36: cubeB_main

# === basic body IDs and variances
cubeA_main_id = env.sim.model.body_name2id("cubeA_main")
cubeB_main_id = env.sim.model.body_name2id("cubeB_main")
eef_id = env.sim.model.body_name2id("gripper0_right_eef")
base_id = env.sim.model.body_name2id('robot0_base')

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


# === get position helper function ===
# def get_pos(body_id):
#     # postion(relative to the base)
#     pos = env.sim.data.body_xpos[body_id] - pos_base
#     return pos
PINCH_OFFSET = np.array([0.0, -0.015, -0.035])  # 3.5 cm down from the wrist body

def get_pos(body_id):
    return env.sim.data.body_xpos[body_id] - pos_base



def get_rot(body_id, rotm_basic):
    # rotation
    rotm = env.sim.data.body_xmat[body_id].reshape((3,3)) @ rotm_basic
    return rotm


# === move to target helper function ===

def has_reached_target(target_pos, target_rotm, epsilon_pos=0.0028, epsilon_rot=0.01):
    current_pos = env.sim.data.body_xpos[eef_id] - pos_base
    current_rotm = env.sim.data.body_xmat[eef_id].reshape(3, 3)

    pos_err = np.linalg.norm(current_pos - target_pos)
    # rot_err = np.linalg.norm(R.from_matrix(current_rotm).as_rotvec() - R.from_matrix(target_rotm).as_rotvec())
    rot_err = R.from_matrix(current_rotm.T @ target_rotm).magnitude()


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


def move_to_target(body_id, steps, rotm_basic, image_observations, actions_taken, height_offset = 0, gripper = -1):

    for i in range(steps):
        pos = get_pos(body_id)
        rotm = get_rot(body_id, rotm_basic)
        pos[2] += height_offset

        if (has_reached_target(pos,rotm) != 1):
            # noisy_pos, noisy_rot = add_noise(pos, R.from_matrix(rotm).as_rotvec())
            # action[0:3] = noisy_pos
            # action[3:6] = noisy_rot
            action[0:3] = pos
            action[3:6] = R.from_matrix(rotm).as_rotvec()
            action[6] = gripper
            print("actual cube A is in ", pos_cubeA)
            print("now eef is in ", env.sim.data.body_xpos[eef_id])
            print("actual cube B is in ", pos_cubeB)

            obs, reward, done, info = env.step(action)
            image_observations.append(obs["agentview_image"])
            actions_taken.append(action.copy())
            env.render()
            # eef_pos = env.sim.data.body_xpos[eef_id] - pos_base
            # eef_rot = env.sim.data.body_xmat[eef_id].reshape(3, 3)

            # pos_error_vec = eef_pos - pos
            # rot_error_vec = R.from_matrix(eef_rot.T @ rotm).magnitude()
            # pos_err_history["x"].append(pos_error_vec[0])
            # pos_err_history["y"].append(pos_error_vec[1])
            # pos_err_history["z"].append(pos_error_vec[2])

            # rot_err_history.append(rot_error_vec)

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
            # if is_robot_lost_control(noisy_pos, rotm):
            #     raise Exception("Robot lost control!")

            time.sleep(0.05)
        else:
            break
    return image_observations, actions_taken

def grasp(body_id, steps, rotm_basic, image_observations, actions_taken, height_offset = 0, gripper = -1):
    for i in range(steps):
        pos = get_pos(body_id)
        rotm = get_rot(body_id, rotm_basic)
        pos[2] += height_offset

        action[0:3] = pos
        action[3:6] = R.from_matrix(rotm).as_rotvec()
        action[6] = gripper

        obs, reward, done, info = env.step(action)
        image_observations.append(obs["agentview_image"])
        actions_taken.append(action.copy())
        env.render()

        
        time.sleep(0.05)
    
def collect_multiple_image_trajectories(start_traj=4, end_traj=20, save_dir="./dataset/train"):
    os.makedirs(save_dir, exist_ok=True)


    for i in range(start_traj, end_traj):
        image_observations = []
        actions_taken = []
        print(f"\n=== Collecting Trajectory {i} ===")

        # Reset environment and get rot matrix
        obs = env.reset()


        # choose the grapper direction cubeA
        rotm_basic = get_rotm_basic_4A()

        # move above cubeA
        move_to_target(cubeA_main_id, 200, rotm_basic, image_observations, actions_taken, APPROACH_DISTANCE)

        # move close to cubeA
        move_to_target(cubeA_main_id, 100, rotm_basic, image_observations, actions_taken)

        # grasp cubeA
        grasp(cubeA_main_id, 20, rotm_basic, image_observations, actions_taken, 0.0125,gripper = 1)

        # move above cubeA
        move_to_target(cubeA_main_id, 10, rotm_basic, image_observations, actions_taken, APPROACH_DISTANCE, gripper = 1)

        # choose the grapper direction for B
        rotm_basic = get_rotm_basic_4B()

        # move above cubeB
        move_to_target(cubeB_main_id, 100, rotm_basic, image_observations, actions_taken, APPROACH_DISTANCE, gripper = 1)

        # move close to cubeB
        move_to_target(cubeB_main_id, 100, rotm_basic, image_observations, actions_taken, GRASP_HEIGHT, gripper = 1)

        #drop cubeA
        grasp(cubeB_main_id, 20, rotm_basic, image_observations, actions_taken, GRASP_HEIGHT, gripper = -1)

        # move above
        move_to_target(cubeB_main_id, 30, rotm_basic, image_observations, actions_taken, APPROACH_DISTANCE, gripper = -1)

        np.savez_compressed(os.path.join(save_dir, f"traj{i}.npz"), observations=np.array(image_observations), actions=np.array(actions_taken))
        print(f"Saved: {save_dir}/traj{i}.npz")
                
collect_multiple_image_trajectories()

# image_observations2, actions_taken2 = [], []
# # === Move and Place Logical ===

# # choose the grapper direction cubeA
# rotm_basic = get_rotm_basic_4A()

# # move above cubeA
# move_to_target(cubeA_main_id, 200, rotm_basic, image_observations2, actions_taken2, APPROACH_DISTANCE)

# # move close to cubeA
# move_to_target(cubeA_main_id, 100, rotm_basic)

# # grasp cubeA
# grasp(cubeA_main_id, 20, rotm_basic, 0.02,gripper = 1)

# # move above cubeA
# move_to_target(cubeA_main_id, 20, rotm_basic, APPROACH_DISTANCE, gripper = 1)

# # choose the grapper direction for B
# rotm_basic = get_rotm_basic_4B()

# # move above cubeB
# move_to_target(cubeB_main_id, 100, rotm_basic, APPROACH_DISTANCE, gripper = 1)

# # move close to cubeB
# move_to_target(cubeB_main_id, 100, rotm_basic, GRASP_HEIGHT, gripper = 1)

# #drop cubeA
# grasp(cubeB_main_id, 20, rotm_basic, GRASP_HEIGHT, gripper = -1)

# # move above
# move_to_target(cubeB_main_id, 30, rotm_basic, APPROACH_DISTANCE, gripper = -1)



# fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# # Plot position errors
# axs[0].plot(pos_err_history["x"], label="Position X")
# axs[0].plot(pos_err_history["y"], label="Position Y")
# axs[0].plot(pos_err_history["z"], label="Position Z")
# axs[0].set_title("EEF Position Error Components")
# axs[0].set_ylabel("Position Error (m)")
# axs[0].legend()
# axs[0].grid(True)

# # Plot rotation errors
# axs[1].plot(rot_err_history, label="Rotation")

# axs[1].set_title("EEF Rotation Error")
# axs[1].set_xlabel("Timestep")
# axs[1].set_ylabel("Rotation Error (rad)")
# axs[1].legend()
# axs[1].grid(True)


# plt.tight_layout()
# plt.show()

# input("Press enter to close")
# # # rollout 示例动作
# # for i in range(2000):
# #     action = env.action_space.sample()     # 随机动作，可替换为策略输出
# #     # obs, reward, done, info = env.step(action)
# #     result = env.step(action)
# #     print("Step result:", result)
# #     env.render()