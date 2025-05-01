import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import CustomPegObject, CustomHoleObject
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

class PegInHole(ManipulationEnv):
    """
    Peg insertion task for a single robot arm.
    The objective is for the robot to pick up a peg and insert it into the designated hole.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.90),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        # Save task settings
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = table_offset

        # These will be our peg and hole objects (created later in _load_model)
        self.peg = None
        self.hole = None

        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Computes the task reward.
        
        Sparse reward:
          - 1.0 if the peg is inserted (i.e. its position lies within tolerance of the hole)
        
        With reward shaping enabled, an additional distance-based bonus is provided.
        """
        success = self._check_success()
        reward = 1.0 if success else 0.0

        if self.reward_shaping:
            # Compute a shaping term: reward increases as the peg gets closer to the hole (in xy plane)
            peg_pos = self.sim.data.body_xpos[self.peg_body_id]
            hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
            dist = np.linalg.norm(peg_pos[:2] - hole_pos[:2])
            reward += (1 - np.tanh(10.0 * dist))
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward

    def _check_success(self):
        """
        Check if the peg is correctly inserted into the hole. 
        NOTE: This is made by ChatGPT, so not made by ME. -- not verified
        """
        # print("Checking Success Not VERIFIED")
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]
        hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
        # Check if peg is near enough to the hole in x and y and below a given height threshold.
        if (
            abs(peg_pos[0] - hole_pos[0]) < 0.03 and 
            abs(peg_pos[1] - hole_pos[1]) < 0.03 and 
            peg_pos[2] < self.table_offset[2] + 0.05
        ):
            return True
        return False

    def _load_model(self):
        """
        Loads the arena, robot, peg, and hole models.
        """
        super()._load_model()

        # Adjust robot base pose relative to the table
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Create the arena.
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        # We assume that the arena XML defines a target hole body named "hole"
        mujoco_arena.set_origin([0, 0, 0])

        # Create a placement initializer if one is not provided.
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            # Sampler for peg with a slight z offset so that it is above the table
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="PegSampler",
                    x_range=[-0.1, 0.1],
                    y_range=[-0.3, -0.1],
                    rotation=None,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.002,
                )
            )
            # Sampler for hole; set z_offset to zero if the hole is meant to be flush with the table.
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="HoleSampler",
                    x_range=[-0.05, 0.05],
                    y_range=[0.05, 0.15],
                    rotation=None,
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.002,
                )
            )
        self.placement_initializer.reset()

        # Instantiate the peg and hole objects.
        self.peg = CustomPegObject(name="peg")
        self.hole = CustomHoleObject(name="hole")
        # Add each object to its respective sampler.
        if isinstance(self.placement_initializer, SequentialCompositeSampler):
            self.placement_initializer.add_objects_to_sampler(
                sampler_name="PegSampler", mujoco_objects=self.peg
            )
            self.placement_initializer.add_objects_to_sampler(
                sampler_name="HoleSampler", mujoco_objects=self.hole
            )
        else:
            self.placement_initializer.add_objects(self.peg)
            self.placement_initializer.add_objects(self.hole)

        # Construct the full task with arena, robot, and both the peg and the hole.
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.peg, self.hole],
        )

    def _setup_references(self):
        """
        Sets up references to simulation elements for efficient access.
        """
        super()._setup_references()
        # Set up references for the peg.
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)
        self.peg_geom_id = [self.sim.model.geom_name2id(g) for g in self.peg.contact_geoms]

        # Define a reference for the target hole.
        # We assume the arena or our object XML defines a body named "hole"
        self.hole_body_id = self.sim.model.body_name2id("hole_main")

    def _setup_observables(self):
        """
        Sets up observables to provide peg position and orientation in the observation.
        """
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.peg_body_id])

            @sensor(modality=modality)
            def peg_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.peg_body_id], to="xyzw")

            observables["peg_pos"] = Observable(
                name="peg_pos",
                sensor=peg_pos,
                sampling_rate=self.control_freq,
                enabled=True,
                active=True,
            )
            observables["peg_quat"] = Observable(
                name="peg_quat",
                sensor=peg_quat,
                sampling_rate=self.control_freq,
                enabled=True,
                active=True,
            )
        return observables

    def _reset_internal(self):
        """
        Resets the internal simulation state.
        """
        super()._reset_internal()
        # Resample the placements for both the peg and the hole
        object_placements = self.placement_initializer.sample()
        for obj_pos, obj_quat, obj in object_placements.values():
            if len(obj.joints) > 0:
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )

            if obj.name == self.hole.name:
                obj_quat_mujoco = T.convert_quat(obj_quat, to ="wxyz")
                # Set the hole position and orientation
                self.sim.model.body_pos[self.hole_body_id] = obj_pos
                self.sim.model.body_quat[self.hole_body_id] = obj_quat