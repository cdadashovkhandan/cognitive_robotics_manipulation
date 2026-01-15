import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import save
import torch.utils.data
from PIL import Image
from datetime import datetime

from network.hardware.device import get_device
#from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.visualisation.plot import plot_results
from network.utils.dataset_processing.grasp import detect_grasps
from skimage.filters import gaussian
import os
import cv2

class GraspGenerator:
    IMG_WIDTH = 224
    IMG_ROTATION = -np.pi * 0.5
    
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path, camera, depth_radius, fig, IMG_WIDTH=224, network='GR_ConvNet', device='cpu', use_meter: bool = False):

        if (device=='cpu'):
            self.net = torch.load(net_path, map_location=device)
            self.device = get_device(force_cpu=True)
        else:
            #self.net = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(1))
            #self.device = get_device()
            print ("GPU is not supported yet! :( -- continuing experiment on CPU!" )
            self.net = torch.load(net_path, map_location='cpu')
            self.device = get_device(force_cpu=True)

        # print (self.net)

        self.camera = camera
        self.cam_rotation = np.pi if camera.camera_mode == "wrist" else 0
        self.near = camera.near
        self.far = camera.far
        self.depth_r = depth_radius
        
        self.fig = fig
        self.network = network
        self.use_meter = bool(use_meter)

        self.PIX_CONVERSION = 277 * IMG_WIDTH/224

        self.cam_data = CameraData(width=camera.width,
                                   height=camera.height,
                                   output_size=IMG_WIDTH,
                                   include_depth=True,
                                   include_rgb=True)

        self.IMG_WIDTH = IMG_WIDTH
        print (self.IMG_WIDTH)

        # Get rotation matrix
        img_center = self.IMG_WIDTH / 2 - 0.5
        # self.img_to_cam = self.get_transform_matrix(-img_center/self.PIX_CONVERSION,
        #                                             img_center/self.PIX_CONVERSION,
        #                                             0,
        #                                             self.IMG_ROTATION)
        self.img_to_cam = self.get_transform_matrix(0.0, 0.0, 0.0, self.IMG_ROTATION)
    def cam_to_robot_base(self):
        print("CAM_TO_ROBOT_BASE: Camera position:", self.camera.x, self.camera.y, self.camera.z)
        return self.get_transform_matrix(self.camera.x, self.camera.y, self.camera.z, self.cam_rotation)

    def get_transform_matrix(self, x, y, z, rot):
        #TODO: Cleanup
        wrist_mode = self.camera.camera_mode == "wrist"
        scale_factor_x = 1 #abs(0.8 / x) if wrist_mode else 1
        scale_factor_y = 1 #abs(0.8 / y) if wrist_mode else 1
        print("SCALE FACTOR:", (scale_factor_x, scale_factor_y))
        return np.array([
                        [np.cos(rot) / scale_factor_x,   -np.sin(rot),   0,  x],
                        [np.sin(rot),   np.cos(rot) / scale_factor_y,    0,  y],
                        [0,             0,              1,  z],
                        [0,             0,              0,  1]
                        ])

    @staticmethod
    def _depthbuf_to_meters(d, near, far):
        """Convert OpenGL/PyBullet depth buffer value in [0,1] to metric depth in meters."""
        d = float(np.clip(d, 0.0, 1.0 - 1e-6))  # avoid numerical blow-up near 1.0
        return (far * near) / (far - (far - near) * d)

    def _meters_to_depthbuf(self, z_m):
        """Convert metric depth (meters) to OpenGL/PyBullet-like depth buffer in [0,1]."""
        z = z_m.astype(np.float32)
        z[~np.isfinite(z)] = 0.0
        z = np.clip(z, self.near + 1e-3, self.far - 1e-3)
        d = (self.far / (self.far - self.near)) * (1.0 - (self.near / z))
        return np.clip(d, 0.0, 1.0).astype(np.float32)

    def grasp_to_robot_frame(self, grasp, depth_img):
        """
        return: x, y, z, roll, opening length gripper, object height

        IMPORTANT:
        - If self.use_meter=True, depth_img is METERS (stereo pipeline).
        - If self.use_meter=False, depth_img is OpenGL/PyBullet depth buffer in [0,1].
        """
        print("-----GRASP TO ROBOT FRAME-----")
        print("Grasp:", grasp)

        # Grasp center pixel (x, y in image coordinates)
        x_p, y_p = grasp.center[0], grasp.center[1]

        def _get_valid_vals(radius):
            """
            Collect valid depth values in a square patch of given radius.
            This function is robust to stereo depth holes.
            """
            print("Attempting to get valid values with radius: ", radius)
            x_min = int(np.clip(x_p - radius, 0, self.IMG_WIDTH - 1))
            x_max = int(np.clip(x_p + radius, 0, self.IMG_WIDTH - 1))
            y_min = int(np.clip(y_p - radius, 0, self.IMG_WIDTH - 1))
            y_max = int(np.clip(y_p + radius, 0, self.IMG_WIDTH - 1))

            # numpy indexing: [row(y), col(x)]
            patch = depth_img[y_min:y_max, x_min:x_max]
            vals = patch[np.isfinite(patch)]

            return vals

        # Try the configured (small) radius first
        vals = _get_valid_vals(self.depth_r)

        # Expand the window if stereo holes dominate the small patch
        if vals.size == 0:
            vals = _get_valid_vals(max(self.depth_r, 3))  # ~7x7 window
        if vals.size == 0:
            vals = _get_valid_vals(max(self.depth_r, 6))  # ~13x13 window

        # Still no depth -> give up on this grasp
        if vals.size == 0:
            print("No depth. Giving up on this grasp")
            return None

        # Use a moderate quantile to approximate the closest visible surface without being dominated by rare outliers.
        d_est = float(np.quantile(vals, 0.0005))
        # Alternatively (more conservative):
        # d_est = float(np.median(vals))


        # Convert depth to meters
        if self.use_meter:
            # Stereo: depth already in meters
            z_p = float(np.clip(d_est, 1e-6, 1e6))
        else:
            # Depth buffer: convert via near/far
            z_p = self._depthbuf_to_meters(d_est, self.near, self.far)


        # TODO: messily copy-pasted AI-generated code. It works, but clean it up.
        
        # # Convert pixel coordinates to meters in image plane
        # x_m = x_p / self.PIX_CONVERSION
        # y_m = y_p / self.PIX_CONVERSION

        # # Image space -> camera space
        # img_xyz = np.array([x_m, y_m, -z_p, 1.0])
        # Map grasp pixel (in the network crop) back to full image pixel coordinates
        u = grasp.center[0] + self.cam_data.top_left[0]  # column (x)
        v = grasp.center[1] + self.cam_data.top_left[1]  # row (y)
        # Convert pixel -> camera coordinates using intrinsics and measured depth z_p
        # X_cam = (u - cx) * Z / fx ; Y_cam = (v - cy) * Z / fy
        x_cam = (u - self.camera.K.cx) * z_p / self.camera.K.fx
        y_cam = (v - self.camera.K.cy) * z_p / self.camera.K.fy
        print("X_CAM, Y_CAM: ", x_cam, y_cam)
        # Image space -> camera space (z negative because convention used elsewhere)
        img_xyz = np.array([x_cam, y_cam, -z_p, 1.0])
        cam_space = np.matmul(self.img_to_cam, img_xyz)

        # Camera space -> robot base frame
        robot_frame_ref = np.matmul(self.cam_to_robot_base(), cam_space)

        if self.use_meter:
            z_down = 0.05
            robot_frame_ref[2] -= z_down

        # Compute roll
        roll = grasp.angle * -1 + self.IMG_ROTATION
        if roll < -np.pi / 2:
            roll += np.pi

        # Pixel width -> gripper opening length
        # opening_length = (grasp.length / int(self.MAX_GRASP * self.PIX_CONVERSION)) * self.MAX_GRASP
        # Convert pixel width -> meters using focal length at measured depth:
        # width_pixels = grasp.length (in network crop). Map to full-image pixels first:
        width_pixels = grasp.length * (self.cam_data.output_size / self.IMG_WIDTH)
        # opening_length_m = (width_pixels * z_p) / fx
        opening_length = (width_pixels * z_p) / self.camera.K.fx

        # obj_height = self.DIST_BACKGROUND - z_p
        # Estimate background distance dynamically from valid depth pixels in this frame.
         # If depth is depth-buffer, convert the median to meters; if it's already meters, use it directly.
        valid_mask = np.isfinite(depth_img)
        bg_est = None
        try:
            valid_vals = depth_img[valid_mask]
            if valid_vals.size > 0:
                median_val = float(np.median(valid_vals))
                if self.use_meter:
                    bg_est = median_val
                else:
                    bg_est = float(self._depthbuf_to_meters(median_val, self.near, self.far))
        except Exception:
            bg_est = None
 
        # fallback
        if bg_est is None:
            bg_est = self.DIST_BACKGROUND
         
        obj_height = bg_est - z_p
        print("estimated table distance: ", bg_est)



        print("z_p (meters):", z_p, "robot z:", robot_frame_ref[2])
        print("Robot frame grasp (x,y,z,roll,opening_length,obj_height):",
              robot_frame_ref[0], robot_frame_ref[1], robot_frame_ref[2],
              roll, opening_length, obj_height)

        return robot_frame_ref[0], robot_frame_ref[1], robot_frame_ref[2], roll, opening_length, obj_height

    def post_process_output(self, q_img, cos_img, sin_img, width_img, pixels_max_grasp):
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * pixels_max_grasp

        q_img = gaussian(q_img, 1.0, preserve_range=True)
        ang_img = gaussian(ang_img, 1.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        return q_img, ang_img, width_img

    def predict(self, rgb, depth, n_grasps=1, show_output=False):
        """
        For the network input:
        - If self.use_meter=True, convert meters -> depth buffer [0,1] (keeps network input consistent).
        - Else, use the buffer directly.
        """
        if self.use_meter:
            depth_for_net = self._meters_to_depthbuf(depth)
        else:
            depth_for_net = depth

        # Network depth normalization (kept as original behavior)
        max_val = np.max(depth_for_net)
        if max_val <= 1e-8:
            max_val = 1.0
        depth_n = depth_for_net * (255 / max_val)
        depth_n = np.clip((depth_n - depth_n.mean()) / 175, -1, 1)

        if self.network == 'GR_ConvNet':
            depth_n = np.expand_dims(np.array(depth_n), axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth_n)
        elif self.network == "GG_CNN":
            x = torch.from_numpy(np.expand_dims(depth_n, 0))
        else:
            print("The selected network has not been implemented yet -- please choose another network!")
            exit()

        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.net.predict(xc)
            pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
            q_img, ang_img, width_img = self.post_process_output(
                pred['pos'], pred['cos'], pred['sin'], pred['width'], pixels_max_grasp
            )

        save_name = None
        if show_output:
            im_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            plot = plot_results(self.fig,
                                rgb_img=im_bgr,
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                depth_img=depth_n,
                                no_grasps=3,
                                grasp_width_img=width_img)

            if not os.path.exists('network_output'):
                os.mkdir('network_output')
            time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            save_name = f'network_output/{time}'
            plot.savefig(save_name + '.png')
            plot.clf()

        grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
        return grasps, save_name

    def predict_grasp(self, rgb, depth, n_grasps=1, show_output=False):
        predictions, save_name = self.predict(rgb, depth, n_grasps=n_grasps, show_output=show_output)
        grasps = []
        for grasp in predictions:
            out = self.grasp_to_robot_frame(grasp, depth)
            if out is None:
                continue
            x, y, z, roll, opening_len, obj_height = out
            grasps.append((x, y, z, roll, opening_len, obj_height))

        return grasps, save_name
