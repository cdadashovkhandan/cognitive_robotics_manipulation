import pybullet as p
from collections import namedtuple
from attrdict import AttrDict
import functools
import os
from datetime import datetime
import numpy as np
import math
from .stereo_depth import StereoDepthSGBM, Intrinsics


def setup_sisbot(p, robotID, gripper_type):
    controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "finger_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
        if info.type == "REVOLUTE":  # set revolute joint to static
            p.setJointMotorControl2(
                robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info

    # explicitly deal with mimic joints
    def controlGripper(robotID, parent, children, mul, **kwargs):
        controlMode = kwargs.pop("controlMode")
        if controlMode == p.POSITION_CONTROL:
            pose = kwargs.pop("targetPosition")
            # move parent joint
            p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                    force=parent.maxForce, maxVelocity=parent.maxVelocity)
            # move child joints
            for name in children:
                child = children[name]
                childPose = pose * mul[child.name]
                p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                        force=child.maxForce, maxVelocity=child.maxVelocity)
        else:
            raise NotImplementedError(
                "controlGripper does not support \"{}\" control mode".format(controlMode))
        # check if there
        if len(kwargs) is not 0:
            raise KeyError("No keys {} in controlGripper".format(
                ", ".join(kwargs.keys())))

    assert gripper_type in ['85', '140']
    mimicParentName = "finger_joint"
    if gripper_type == '85':
        mimicChildren = {"right_outer_knuckle_joint": 1,
                         "left_inner_knuckle_joint": 1,
                         "right_inner_knuckle_joint": 1,
                         "left_inner_finger_joint": -1,
                         "right_inner_finger_joint": -1}
    else:
        mimicChildren = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1}
    parent = joints[mimicParentName]
    children = AttrDict((j, joints[j])
                        for j in joints if j in mimicChildren.keys())
    controlRobotiqC2 = functools.partial(
        controlGripper, robotID, parent, children, mimicChildren)

    return joints, controlRobotiqC2, controlJoints, mimicParentName


def setup_sisbot_force(p, robotID, gripper_type):
    controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "finger_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable", "jointAxis", "parentFramePos", "parentFrameOrn"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        jointAxis = info[13]
        parentFramePos = info[14]
        parentFrameOrn = info[15]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable,
                         jointAxis, parentFramePos, parentFrameOrn)
        if info.type == "REVOLUTE":  # set revolute joint to static
            p.setJointMotorControl2(
                robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    for j in joints:
        print(joints[j])
    # explicitly deal with mimic joints

    def controlGripper(robotID, parent, children, mul, **kwargs):
        controlMode = kwargs.pop("controlMode")
        if controlMode == p.POSITION_CONTROL:
            pose = kwargs.pop("targetPosition")
            # move parent joint
            p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                    force=parent.maxForce, maxVelocity=parent.maxVelocity)
            # p.setJointMotorControl2(robotID, parent.id, p.TORQUE_CONTROL,
            #                         force=10, maxVelocity=parent.maxVelocity)
            return
            # move child joints
            for name in children:
                child = children[name]
                childPose = pose * mul[child.name]
                p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                        force=child.maxForce, maxVelocity=child.maxVelocity)
        else:
            raise NotImplementedError(
                "controlGripper does not support \"{}\" control mode".format(controlMode))
        # check if there
        if len(kwargs) is not 0:
            raise KeyError("No keys {} in controlGripper".format(
                ", ".join(kwargs.keys())))

    assert gripper_type in ['85', '140']
    mimicParentName = "finger_joint"
    if gripper_type == '85':
        mimicChildren = {"right_outer_knuckle_joint": 1,
                         "left_inner_knuckle_joint": 1,
                         "right_inner_knuckle_joint": 1,
                         "left_inner_finger_joint": -1,
                         "right_inner_finger_joint": -1}
    else:
        mimicChildren = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1}
    parent = joints[mimicParentName]
    children = AttrDict((j, joints[j])
                        for j in joints if j in mimicChildren.keys())
    # Create all the gear constraint
    for name in children:
        child = children[name]
        c = p.createConstraint(robotID, parent.id, robotID, child.id, p.JOINT_GEAR, child.jointAxis,
                               # child.parentFramePos, (0, 0, 0), child.parentFrameOrn, (0, 0, 0))
                               (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        p.changeConstraint(c, gearRatio=-mimicChildren[name], maxForce=10000)
    controlRobotiqC2 = functools.partial(
        controlGripper, robotID, parent, children, mimicChildren)

    return joints, controlRobotiqC2, controlJoints, mimicParentName


class Camera:
    def __init__(self, cam_pos, cam_target, near, far, size, fov):
        self.x, self.y, self.z = cam_pos
        self.x_t, self.y_t, self.z_t = cam_target
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 1, 0])

        self.rec_id = None

    def get_cam_img(self):
        """
        Method to get images from camera
        return:
        rgb
        depth
        segmentation mask
        """
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
        rgb = np.array(rgb, dtype=np.uint8).reshape((_h, _w, 4))
        depth = np.array(depth).reshape((_h, _w))
        seg = np.array(seg).reshape((_h, _w))

        return rgb[:, :, 0:3], depth, seg

    def start_recording(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file = f'{save_dir}/{now}.mp4'

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.rec_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file)

    def stop_recording(self):
        p.stopStateLogging(self.rec_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

class StereoCamera:
    """
    Stereo camera wrapper that returns a PyBullet-like depth buffer computed from stereo RGB.

    It constructs two parallel cameras (left/right) by shifting the camera position/target
    along the camera's right axis by +/- baseline/2.

    Output of get_cam_img() matches Camera.get_cam_img():
      - rgb: left RGB
      - depth: non-linear depth buffer in [0,1]
      - seg: left segmentation mask
    """

    def __init__(self,
                 cam_pos, cam_target,
                 near, far, size, fov,
                 baseline_m: float = 0.3,
                 num_disparities: int = 128,
                 block_size: int = 7,
                 invalid_depth_value: float = 0.0):
        self.baseline_m = float(baseline_m)
        self.num_disparities = int(num_disparities)
        self.block_size = int(block_size)
        self.invalid_depth_value = float(invalid_depth_value)

        self.x, self.y, self.z = cam_pos
        self.x_t, self.y_t, self.z_t = cam_target
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        # Compute camera right vector from forward and world up
        cam_pos_np = np.array(cam_pos, dtype=np.float32)
        cam_target_np = np.array(cam_target, dtype=np.float32)

        forward = cam_target_np - cam_pos_np
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)

        half = 0.5 * self.baseline_m
        left_pos = (cam_pos_np - half * right).tolist()
        right_pos = (cam_pos_np + half * right).tolist()
        left_target = cam_target_np.tolist()
        right_target = cam_target_np.tolist()

        print("Stereo left_pos:", left_pos, "right_pos:", right_pos, "baseline:", self.baseline_m)

        self.left_cam = Camera(left_pos, left_target, near, far, size, fov)
        self.right_cam = Camera(right_pos, right_target, near, far, size, fov)

        # Compute fx in pixel units (PyBullet fov is vertical FOV in degrees)
        fov_rad = math.radians(self.fov)
        fy = (self.height / 2.0) / math.tan(fov_rad / 2.0)
        self.fx = fy * (self.width / self.height)

        self.stereo = StereoDepthSGBM(
            num_disparities=self.num_disparities,
            block_size=self.block_size,
            min_disparity=0
        )
        self.K = Intrinsics(fx=self.fx, fy=fy, cx=self.width / 2.0, cy=self.height / 2.0)

    def get_stereo_pair(self):
        left_rgb, _, left_seg = self.left_cam.get_cam_img()
        right_rgb, _, _ = self.right_cam.get_cam_img()

        diff = np.mean(np.abs(left_rgb.astype(np.int16) - right_rgb.astype(np.int16)))
        print("mean abs RGB diff:", diff)
        return left_rgb, right_rgb, left_seg

    def get_cam_img(self):
        left_rgb, right_rgb, left_seg = self.get_stereo_pair()

        depth_buf = self.stereo.estimate_depth_buffer(
            left_rgb, right_rgb,
            K=self.K,
            baseline_m=self.baseline_m,
            near=self.near,
            far=self.far,
            invalid_value=self.invalid_depth_value
        )

        return left_rgb, depth_buf, left_seg
