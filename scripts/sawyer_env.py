import robosuite as suite
from robosuite.controllers import load_composite_controller_config
# from robosuite.wrappers import GymWrapper
import numpy as np
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
root_folder = Path(__file__).parent.parent


# # create controller
robot_name = "Sawyer"
controller_path = root_folder/"scripts"/"indy7_absolute_pose.json"
controller_config = load_composite_controller_config(robot = robot_name, controller=str(controller_path))
# # print(controller_config)

# 创建 Sawyer + Stack 环境
env = suite.make(
    env_name="Stack",                   # 任务类型：Stacking
    robots="Sawyer",                    # 使用 Sawyer 机械臂
    controller_configs=controller_config,
    has_renderer=True,                  # 实时渲染
    has_offscreen_renderer=True,       # 不保存图像帧
    use_camera_obs=False,               # 暂不使用图像观察
    control_freq=20,                    # 控制频率
    horizon=600,                        # 每条 trajectory 长度
    render_camera=None,          # 渲染视角
)

# 重置环境
# env = GymWrapper(env)
obs = env.reset()

""" Objects in environment: ('world', 'table', 'left_eef_target', 'right_eef_target', 
                         'robot0_base', 'robot0_right_arm_base_link', 'robot0_right_l0', 
                         'robot0_head', 'robot0_screen', 'robot0_head_camera', 
                         'robot0_right_torso_itb', 'robot0_right_l1', 'robot0_right_l2', 
                         'robot0_right_l3', 'robot0_right_l4', 'robot0_right_arm_itb', 
                         'robot0_right_l5', 'robot0_right_hand_camera', 'robot0_right_wrist', 
                         'robot0_right_l6', 'robot0_right_hand', 'gripper0_right_gripper_base', 
                         'gripper0_right_eef', 'gripper0_right_l_finger',
                           'gripper0_right_l_finger_tip', 'gripper0_right_r_finger', 
                           'gripper0_right_r_finger_tip', 'robot0_right_l4_2', 'robot0_right_l2_2', 
                           'robot0_right_l1_2', 'fixed_mount0_base', 'fixed_mount0_controller_box', 
                           'fixed_mount0_pedestal_feet', 'fixed_mount0_torso', 'fixed_mount0_pedestal', 
                           'cubeA_main', 'cubeB_main') """

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
            # if is_robot_lost_control(noisy_pos, rotm):
            #     raise Exception("Robot lost control!")

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

# # 查看环境中的物体
# print("Objects in environment:", env.sim.model.body_names)


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


# # rollout 示例动作
# for i in range(2000):
#     action = env.action_space.sample()     # 随机动作，可替换为策略输出
#     # obs, reward, done, info = env.step(action)
#     result = env.step(action)
#     print("Step result:", result)
#     env.render()