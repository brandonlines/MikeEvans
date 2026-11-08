"""Direct live webcam oral cavity insertion with contrast-preserved teeth and aperture depth."""
import cv2
import numpy as np

# Inner oral aperture boundary and lateral corners
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
ALIGN_PTS = [61, 291] + INNER_LIPS

class MouthBlender:
    def __init__(self):
        self.reset()

    def reset(self):
        self.smooth_matrix = None

    def render(self, output_image, webcam_frame, current_landmarks, target_landmarks):
        h_out, w_out = output_image.shape[:2]

        top_lip = current_landmarks[13]
        bot_lip = current_landmarks[14]
        opening = float(np.linalg.norm(bot_lip - top_lip))

        # Suppress overlay when mouth is essentially closed
        if opening < 2.0:
            self.smooth_matrix = None
            return output_image

        src_pts = current_landmarks[ALIGN_PTS].astype(np.float32)
        dst_pts = target_landmarks[ALIGN_PTS].astype(np.float32)

        # 2D affine mapping from webcam aperture directly into portrait target coordinates
        matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if matrix is None:
            return output_image

        if self.smooth_matrix is None:
            self.smooth_matrix = matrix
        else:
            self.smooth_matrix = 0.70 * matrix + 0.30 * self.smooth_matrix

        warped_cam = cv2.warpAffine(webcam_frame, self.smooth_matrix, (w_out, h_out), borderMode=cv2.BORDER_REFLECT_101)

        # Build clean aperture mask to replace the stretched 2D mesh texture
        dst_inner = target_landmarks[INNER_LIPS].astype(np.float32)
        mask = np.zeros((h_out, w_out), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(dst_inner).astype(np.int32)], 255)
        
        # Crisp 2px edge feathering so lips cleanly frame the oral cavity
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Smooth ramp from 2.0px to 4.0px opening
        fade_weight = float(np.clip((opening - 2.0) / 2.0, 0.0, 1.0))
        alpha = (mask.astype(np.float32) / 255.0 * fade_weight)[:, :, None]

        # White balance correction based on skin around the lips (avoids sampling the pink stretched seam)
        outer_sample = (mask == 0) & (cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0)
        if outer_sample.sum() > 30:
            cam_skin = np.median(warped_cam[outer_sample], axis=0)
            target_skin = np.median(output_image[outer_sample], axis=0)
            # Gentle color tint adjustment without boosting cavity luminance
            tint = np.clip((target_skin - cam_skin) * 0.45, -20, 20)
            warped_cam = np.clip(warped_cam.astype(np.float32) + tint, 0, 255).astype(np.uint8)

        # Composite live cavity over the stretched mesh
        blended = output_image.astype(np.float32) * (1.0 - alpha) + warped_cam.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)
