import pybullet as p
from collections import namedtuple
from attrdict import AttrDict
import functools
import os
from datetime import datetime
import numpy as np


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
    def __init__(self, cam_pos, cam_target, near, far, size, fov, camera_mode):
        self.x, self.y, self.z = cam_pos
        self.x_t, self.y_t, self.z_t = cam_target
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov
        self.camera_mode = camera_mode

        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 1, 0])

        self.rec_id = None

    def match_wrist(self, link_pos, link_orn):
        """
        Reposition camera to be in the same place as the end effector (with offset) if Eye On Hand mode is enabled.
        """
        new_pos = link_pos

        mat = p.getMatrixFromQuaternion(link_orn)
        R = np.array(mat).reshape(3, 3)

        local_forward = np.array([1.0, 0.0, 0.0])
        local_up = np.array([0.0, 0.0, 1.0])

        # Rotate into world frame.
        forward = R.dot(local_forward)
        up = R.dot(local_up)

        target_offset = 0.05    # Point a bit in front of the EE.
        camera_backoff = 0.40   # Distance behind the EE along -forward.
        camera_height = 0.2    # Small offset along up to avoid collision.
        
        # Compute world-space target and camera positions.
        target_pos = (link_pos + forward * target_offset + up * camera_height).tolist()
        new_pos = (link_pos - forward * camera_backoff + up * camera_height)    .tolist()
        
        self.view_matrix = p.computeViewMatrix(new_pos, target_pos, up.tolist())
        self.x, self.y, self.z = new_pos
        self.x_t, self.y_t, self.z_t = target_pos

    def get_cam_img(self, link_pos = None, link_orn = None):
        """
        Method to get images from camera
        return:
        rgb
        depth
        segmentation mask
        """
        
        if link_pos is not None and link_orn is not None:
            self.match_wrist(link_pos, link_orn)

        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
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

# class EyeOnHandCamera(Camera):
#     def __init__(self, cam_pos, cam_target, near, far, size, fov, offset):
#         self.x, self.y, self.z = cam_pos
#         self.x_t, self.y_t, self.z_t = cam_target
#         self.width, self.height = size
#         self.near, self.far = near, far
#         self.fov = fov
#         self.offset = offset
#         aspect = self.width / self.height
#         self.projection_matrix = p.computeProjectionMatrixFOV(
#             fov, aspect, near, far)
#         self.view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 1, 0])

#         self.rec_id = None

#     def __get_camera_pose(self):
#         """
#             Determine the position and orientation of the camera based on the robot and link id
#         """
#         pos, orn = p.getLinkState(self.robot_id, self.eef, computeForwardKinematics=True)
#         # rot_mat = np.array(self.p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        
#         return pos, orn

    
#     def get_cam_img(self):
#         p.computeViewMatrix(link_pos, link_orn, [0, 1, 0])

# TODO: AI-generated code. Don't trust it:

class DualMonoCameras(Camera):
    """
    Two ordinary RGB cameras in fixed world poses (or follow different frames).
    Provides get_cam_imgs() -> ((rgb_l, depth_l, intr_l), (rgb_r, depth_r, intr_r))
    Depth is metric computed from pybullet depth buffer. Optionally produce intrinsics/extrinsics for later stereo rectification/disparity.
    """
    def __init__(self, cam_pos, cam_target, near, far, size, fov):
        # camX_pose = (eye_pos, target_pos, up_vec) each are length-3 iterables in world frame
        super().__init__()


        # fx = fy = 0.5 * self.width / np.tan(np.deg2rad(self.fov) / 2)
        # cx, cy = self.width/2.0, self.height/2.0
        # self.intrinsics = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': self.width, 'height': self.height, 'fov': self.fov}

    def _capture(self, eye, target, up):
        view = self.p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up)
        proj = self.p.computeProjectionMatrixFOV(fov=self.fov, aspect=float(self.width)/self.height, nearVal=self.near, farVal=self.far)
        img = self.p.getCameraImage(self.width, self.height, viewMatrix=view, projectionMatrix=proj,
                                    renderer=self.p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.reshape(img[2], (self.height, self.width, 4))[:, :, :3]
        depth_buffer = np.reshape(img[3], (self.height, self.width))
        depth = self.near * self.far / (self.far - (self.far - self.near) * depth_buffer)
        return rgb[:, :, ::-1].astype(np.uint8), depth.astype(np.float32), self.intrinsics

    def get_cam_img(self):
        out = []
        for (eye, target, up) in self.cam_poses:
            out.append(self._capture(eye, target, up))
        # returns ((rgb_l, depth_l, intr_l), (rgb_r, depth_r, intr_r))
        return tuple(out)

    def get_extrinsics(self):
        # returns relative transform (R, t) from cam0 to cam1 in world frame: used for stereo rectification
        eye0, target0, up0 = self.cam_poses[0]
        eye1, target1, up1 = self.cam_poses[1]
        # for simplicity: use camera origins only as translations; for rotation compute from forward/up
        def pose_from_eye_target(eye, target, up):
            z = np.array(target) - np.array(eye)
            z = z / np.linalg.norm(z)
            x = np.cross(z, np.array(up))
            x = x / np.linalg.norm(x)
            y = np.cross(x, z)
            R = np.stack([x, y, z], axis=1)  # world -> cam
            t = np.array(eye).reshape(3,1)
            return R, t
        R0, t0 = pose_from_eye_target(*self.cam_poses[0])
        R1, t1 = pose_from_eye_target(*self.cam_poses[1])
        # transform from cam0 to cam1: R = R1^T R0? return R_cam0_to_cam1, t
        R_cam0_to_cam1 = R1.T.dot(R0)
        t_cam0_to_cam1 = R1.T.dot(t0 - t1)
        return R_cam0_to_cam1, t_cam0_to_cam1
