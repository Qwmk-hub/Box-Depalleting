import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from inferencer.core.inferencer_base import InferencerBase
from utils import CheckExecTime
from typing import List, Tuple, Dict, Optional

# Helper functions
def _calculate_quaternion(corners_3d: np.ndarray, logger) -> List[float]:
    """Computes a quaternion representing the orientation of a plane defined by 3D corner points."""
    valid_pts = corners_3d[~np.isnan(corners_3d).any(axis=1)]
    if valid_pts.shape[0] < 3:
        logger.warning(f"Not enough valid corners ({valid_pts.shape[0]}) to compute quaternion; using identity.")
        return [0.0, 0.0, 0.0, 1.0]
    p0 = valid_pts[0]
    p1 = None
    for i in range(1, valid_pts.shape[0]):
        if np.linalg.norm(valid_pts[i] - p0) > 1e-6: p1 = valid_pts[i]; break
    if p1 is None:
        logger.warning("All valid points are at the same location; using identity quaternion.")
        return [0.0, 0.0, 0.0, 1.0]
    p2 = None
    for i in range(1, valid_pts.shape[0]):
        if np.linalg.norm(np.cross(p1 - p0, valid_pts[i] - p0)) > 1e-6: p2 = valid_pts[i]; break
    if p2 is None:
        logger.warning("All valid points are collinear; using identity quaternion.")
        return [0.0, 0.0, 0.0, 1.0]
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        logger.warning("Degenerate normal vector; using identity quaternion.")
        return [0.0, 0.0, 0.0, 1.0]
    unit_normal = normal / norm
    try:
        rot, _ = R.align_vectors([[0, 0, 1]], [unit_normal])
        return rot.as_quat().tolist()
    except ValueError as e:
        logger.error(f"Quaternion alignment failed: {e}")
        return [0.0, 0.0, 0.0, 1.0]

def _pixel_to_camera(point_px: Tuple[float, float], depth_map: np.ndarray,
                     intrinsics: Dict[str, float], depth_scale: float, logger) -> Optional[np.ndarray]:
    """Converts pixel coordinates and a depth value to a 3D point in the camera coordinate system.
    Returns coordinates in millimeters."""
    x_px, y_px = np.round(point_px).astype(int)
    h, w = depth_map.shape
    if not (0 <= y_px < h and 0 <= x_px < w):
        logger.debug(f"Pixel coordinates {point_px} are out of depth map bounds ({w}x{h}).")
        return None
    depth_raw = depth_map[y_px, x_px]
    if depth_raw <= 0:
        logger.debug(f"Invalid depth value {depth_raw} at coordinates {point_px}.")
        return None
    z_m = float(depth_raw) / depth_scale
    x_m = (x_px - intrinsics['cx']) * z_m / intrinsics['fx']
    y_m = (y_px - intrinsics['cy']) * z_m / intrinsics['fy']
    return np.array([x_m, y_m, z_m], dtype=np.float64) * 1000.0


class DepalBox(InferencerBase):
    """
    An inferencer class that detects boxes and estimates their 3D pose using a
    YOLO model specifically trained on depth data. All 3D outputs are in millimeters.
    """
    def __init__(self, model_name: str, model_args: dict, logger) -> None:
        """Initializes the DepalBox inferencer."""
        super().__init__(model_name, model_args, logger)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = YOLO(model_args['path_weight'])
        self.model.to(self.device)
        self.imgsz = (640, 640)
        self.min_area_px = 100
        self.depth_scale = 1000.0
        self.box_shrink_ratio = 0.9
        self.conf_thres = 0.7
        self.depth_norm_percentiles = (1, 99)
        self.logger.info("DepalBox inferencer for depth-based detection initialized.")
        self._warmup()

    def _warmup(self):
        """Warms up the YOLO model by performing a dummy inference."""
        self.logger.info("Warming up YOLO model...")
        dummy_img = np.zeros((*self.imgsz, 3), dtype=np.uint8)
        with torch.no_grad(), CheckExecTime() as t:
            self.model.predict(source=dummy_img, verbose=False)
        self.logger.debug(f"Warmup finished in {t.elapsed * 1000:.1f} ms")

    def _depth_normalize(self, depth: np.ndarray) -> np.ndarray:
        """Normalizes the depth map and converts it to a 3-channel image for YOLO input."""
        valid_mask = depth > 0
        if not valid_mask.any():
            return np.zeros((*depth.shape, 3), dtype=np.uint8)
        valid_depth = depth[valid_mask]
        p_min, p_max = self.depth_norm_percentiles
        min_val = np.percentile(valid_depth, p_min)
        max_val = np.percentile(valid_depth, p_max)
        clipped_depth = np.clip(depth, min_val, max_val)
        range_val = max_val - min_val
        if range_val < 1e-6:
            range_val = 1.0
        norm_depth = ((clipped_depth - min_val) / range_val * 255.0).astype(np.uint8)
        yolo_input = cv2.cvtColor(norm_depth, cv2.COLOR_GRAY2BGR)
        yolo_input[~valid_mask] = 0
        return yolo_input

    def _process_single_detection(self, mask: torch.Tensor, orig_shape: Tuple[int, int], 
                                  depth_map: np.ndarray, intrinsics: Dict[str, float],
                                  roi_2d: List[int],
                                  roi_3d_min: Optional[np.ndarray],
                                  roi_3d_max: Optional[np.ndarray]) -> Optional[Dict]:
        """Processes a single detection mask and returns data in the specified output format."""
        h, w = orig_shape
        mask_np = cv2.resize(mask.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) < self.min_area_px: return None
        rect = cv2.minAreaRect(main_contour)
        box_2d = cv2.boxPoints(rect)
        center_2d = box_2d.mean(axis=0)
        shrunk_box_2d = (box_2d - center_2d) * self.box_shrink_ratio + center_2d
        corners_3d_list = [_pixel_to_camera(pt, depth_map, intrinsics, self.depth_scale, self.logger) for pt in shrunk_box_2d]
        corners_3d = np.array([pt if pt is not None else [np.nan]*3 for pt in corners_3d_list])
        center_3d_pos = _pixel_to_camera(center_2d, depth_map, intrinsics, self.depth_scale, self.logger)
        has_valid_center_3d = center_3d_pos is not None
        x, y, cw, ch = cv2.boundingRect(main_contour)
        is_in_roi_2d = (roi_2d[0] <= x and roi_2d[1] <= y and (x + cw) <= roi_2d[2] and (y + ch) <= roi_2d[3])
        is_in_roi_3d = True
        if has_valid_center_3d and roi_3d_min is not None and roi_3d_max is not None:
            is_in_roi_3d = np.all(roi_3d_min <= center_3d_pos) and np.all(center_3d_pos <= roi_3d_max)
        is_valid = has_valid_center_3d and is_in_roi_2d and is_in_roi_3d
        orientation_quat = _calculate_quaternion(corners_3d, self.logger) if has_valid_center_3d else [0.0, 0.0, 0.0, 1.0]
        position_list = center_3d_pos.tolist() if has_valid_center_3d else [np.nan, np.nan, np.nan]
        center_3d_combined = position_list + orientation_quat
        return {
            'corner_2d': shrunk_box_2d.tolist(), 'center_2d': center_2d.tolist(), 'corner_3d': corners_3d.tolist(),
            'center_3d': center_3d_combined, 'valid': bool(is_valid)
        }
    
    def _save_ply(self, detected_objects: List[Dict], filepath: str):
        """Saves 3D points as a wireframe PLY, ensuring high compatibility."""
        valid_color = (0, 0, 255)      # Blue
        invalid_color = (255, 0, 0)    # Red
        center_color = (255, 255, 0)   # Yellow
        axis_color = (255, 255, 255)   # White
        axis_length = 100.0

        vertices, faces, vertex_index_counter = [], [], 0
        
        for item in detected_objects:
            obj_color = valid_color if item['valid'] else invalid_color
            
            # 1. Process Corners and their edges (lines)
            corner_indices = []
            for corner in item['corner_3d']:
                corner_3d = np.array(corner)
                if not np.isnan(corner_3d).any():
                    vertices.append(tuple(corner_3d) + obj_color)
                    corner_indices.append(vertex_index_counter)
                    vertex_index_counter += 1
            
            # if len(corner_indices) == 4:
            #     faces.append([2, corner_indices[0], corner_indices[1]])
            #     faces.append([2, corner_indices[1], corner_indices[2]])
            #     faces.append([2, corner_indices[2], corner_indices[3]])
            #     faces.append([2, corner_indices[3], corner_indices[0]])
            # elif len(corner_indices) == 3:
            #     faces.append([2, corner_indices[0], corner_indices[1]])
            #     faces.append([2, corner_indices[1], corner_indices[2]])
            #     faces.append([2, corner_indices[2], corner_indices[0]])

            # 2. Process Center and Axis Vector
            center_3d_full = item['center_3d']
            center_3d_pos = np.array(center_3d_full[:3])
            quat_xyzw = center_3d_full[3:]
            if not np.isnan(center_3d_pos).any():
                vertices.append(tuple(center_3d_pos) + center_color)
                center_idx = vertex_index_counter
                vertex_index_counter += 1
                
                try:
                    r = R.from_quat(quat_xyzw)
                    z_axis_rotated = r.apply([0, 0, 1])
                    axis_endpoint = center_3d_pos + z_axis_rotated * axis_length
                    
                    vertices.append(tuple(axis_endpoint) + axis_color)
                    axis_end_idx = vertex_index_counter
                    vertex_index_counter += 1
                    
                    # faces.append([2, center_idx, axis_end_idx])
                except ValueError as e:
                    self.logger.warning(f"Could not process quaternion {quat_xyzw} for axis visualization: {e}")

        if not vertices:
            self.logger.warning("No 3D points with valid coordinates found to save to PLY file.")
            return
        try:
            with open(filepath, 'w') as f:
                f.write("ply\n"); f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n"); f.write("property float y\n"); f.write("property float z\n")
                f.write("property uchar red\n"); f.write("property uchar green\n"); f.write("property uchar blue\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                for v in vertices:
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]} {v[4]} {v[5]}\n")
                # for face in faces:
                #     f.write(" ".join(map(str, face)) + "\n")
            # self.logger.info(f"Successfully saved {len(vertices)} points and {len(faces)} faces to {filepath}")
        except IOError as e:
            self.logger.error(f"Failed to save PLY file to {filepath}: {e}")

    def forward(self, input_data: dict) -> dict:
        """Runs the main inference pipeline and returns data in the specified format."""
        rgb = cv2.imread(input_data['rgb_img_path'])
        depth_raw = cv2.imread(input_data['depth_img_path'], cv2.IMREAD_UNCHANGED)
        if rgb is None or depth_raw is None:
            msg = f"Image loading failed: RGB is None: {rgb is None}, Depth is None: {depth_raw is None}"
            self.logger.error(msg)
            return {'error': msg}
        cv2.imwrite("./newdata/rgb_image.png", rgb)
        cv2.imwrite("./newdata/depth_image.png", depth_raw)

        h, w = rgb.shape[:2]
        depth = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)

        depth_clipped = np.clip(depth, 0, 60000)
        yolo_input = self._depth_normalize(depth_clipped)
        
        with torch.no_grad(), CheckExecTime() as t_pred:
            results = self.model.predict(source=yolo_input, iou=0.1, conf=self.conf_thres, imgsz=self.imgsz, retina_masks=True ,verbose=False)
        self.logger.debug(f"YOLO inference took {t_pred.elapsed * 1000:.1f} ms")

        Ki = input_data['color_intrinsic']
        intrinsics = {'fx': float(Ki[0][0]), 'fy': float(Ki[1][1]), 'cx': float(Ki[0][2]), 'cy': float(Ki[1][2])}
        roi_2d = input_data.get('roi_2d', [0, 0, w, h])
        
        roi_3d_min, roi_3d_max = None, None
        roi_3d_raw = input_data.get('roi_3d')
        if roi_3d_raw and isinstance(roi_3d_raw, list) and len(roi_3d_raw) == 24:
            try:
                corners_3d = np.array(roi_3d_raw, dtype=float).reshape(8, 3)
                roi_3d_min = corners_3d.min(axis=0)
                roi_3d_max = corners_3d.max(axis=0)
                self.logger.debug(f"Parsed 3D ROI. Min: {roi_3d_min}, Max: {roi_3d_max}")
            except (ValueError, TypeError) as e:
                self.logger.error(f"Failed to parse 'roi_3d'. It should be a list of 24 numbers. Error: {e}")
        
        output_data = []
        masks = []
        if results[0].masks is not None:
            masks = results[0].masks.data
        
        for mask in masks:
            processed_result = self._process_single_detection(
                mask, (h, w), depth, intrinsics, roi_2d, roi_3d_min, roi_3d_max
            )
            if processed_result:
                output_data.append(processed_result)

        valid_count = sum(1 for item in output_data if item['valid'])
        
        if input_data.get('save_ply', False):
            self._save_ply(output_data, "./ply/detected_objects.ply")
        self.logger.info(f"Detected {len(masks)} masks, {valid_count} valid objects after filtering.")
        
        return {
            'func_name': input_data.get('function_name', ''),
            'detected_count': len(masks),
            'valid_count': valid_count,
            'data': output_data
        }