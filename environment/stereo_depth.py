import numpy as np
import cv2
from dataclasses import dataclass

def meters_to_depth_buffer(z_m: np.ndarray, near: float, far: float,
                           invalid_value: float = 0.0) -> np.ndarray:
    """
    Convert metric depth (meters) to PyBullet/OpenGL-like non-linear depth buffer in [0,1].

    Formula (inverse of z = (far*near)/(far - (far-near)*d)):
        d = (far/(far-near)) * (1 - near/z)
    """
    z = z_m.astype(np.float32)
    d = np.full_like(z, invalid_value, dtype=np.float32)

    valid = np.isfinite(z) & (z > 1e-6)
    if np.any(valid):
        d_valid = (far / (far - near)) * (1.0 - (near / z[valid]))
        d[valid] = np.clip(d_valid, 0.0, 1.0)

    return d


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

class StereoDepthSGBM:
    def __init__(self,
                 num_disparities: int = 128,
                 block_size: int = 5,
                 min_disparity: int = 0):
        if num_disparities % 16 != 0:
            raise ValueError("num_disparities must be divisible by 16")

        self.num_disparities = num_disparities
        self.block_size = block_size
        self.min_disparity = min_disparity

        # SGBM parameters: a default value (which can be adjusted later).
        P1 = 8 * 1 * block_size * block_size
        P2 = 32 * 1 * block_size * block_size

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def estimate_depth(self,
                       left_rgb: np.ndarray,
                       right_rgb: np.ndarray,
                       K: Intrinsics,
                       baseline_m: float,
                       invalid_value: float = 1.0) -> np.ndarray:
        """
        Returns depth map in meters (float32). Invalid pixels set to NaN by default.
        Assumes images are rectified (epipolar lines horizontal).
        """
        if left_rgb.shape != right_rgb.shape:
            raise ValueError("left/right image shapes do not match")

        # grayscale matching
        left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)

        # disparity: CV_16S, scaled by 16
        disp = self.matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Invalid disparity (<=0) cannot be used to calculate depth.
        valid = disp > 0.5  # can adjust
        depth = np.full(disp.shape, invalid_value, dtype=np.float32)

        # Z = f * B / d
        depth[valid] = (K.fx * baseline_m) / disp[valid]

        print("disp min/max/mean:", float(np.nanmin(disp)), float(np.nanmax(disp)), float(np.nanmean(disp)))
        print("disp valid ratio (>0.5):", float((disp > 0.5).mean()))
        print("depth_m finite ratio:", float(np.isfinite(depth).mean()))
        print("depth_m min/max/std:", float(np.nanmin(depth)), float(np.nanmax(depth)), float(np.nanstd(depth)))

        return depth

    def estimate_depth_buffer(self,
                              left_rgb: np.ndarray,
                              right_rgb: np.ndarray,
                              K: Intrinsics,
                              baseline_m: float,
                              near: float,
                              far: float,
                              invalid_value: float = 1.0) -> np.ndarray:
        """
        Returns a PyBullet/OpenGL-like depth buffer (0..1, non-linear), aligned with Camera.get_cam_img() output.

        - near/far MUST match the projection settings used in your Camera.
        - invalid_value defaults to 0.0 to mimic "no depth" style values, but you can set 1.0 if you prefer.
        """
        depth_m = self.estimate_depth(left_rgb, right_rgb, K, baseline_m, invalid_value=np.nan)
        depth_buf = meters_to_depth_buffer(depth_m, near=near, far=far, invalid_value=invalid_value)

        print("left_rgb", left_rgb.dtype, left_rgb.shape, left_rgb.min(), left_rgb.max())
        print("right_rgb", right_rgb.dtype, right_rgb.shape, right_rgb.min(), right_rgb.max())

        valid_m = np.isfinite(depth_buf)
        print("depth_m valid ratio:", valid_m.mean(), "min/max:", np.nanmin(depth_m), np.nanmax(depth_m))

        print("depth_buf min/max/mean:", float(np.min(depth_buf)), float(np.max(depth_buf)), float(np.mean(depth_buf)))
        return depth_buf
