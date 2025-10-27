import sys
import os
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
import mediapipe as mp

class FaceAnimalFusion:
    def __init__(self):
        # Initialize MediaPipe face mesh detector
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # lower for better adaptability
            min_tracking_confidence=0.3
        )
        
    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Can't load: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def detect_landmarks(self, img):
        # Detect 478 facial landmarks using MediaPipe
        results = self.face_mesh.process(img)
        if not results.multi_face_landmarks:
            raise ValueError("No face detected")
        
        landmarks = results.multi_face_landmarks[0]
        h, w = img.shape[:2]
        
        # Convert normalized coords to pixel coords
        points = []
        for lm in landmarks.landmark:
            points.append([int(lm.x * w), int(lm.y * h)])
        
        return np.array(points)
    
    def get_feature_regions(self, landmarks):
        # Extract bounding boxes for eyes, nose, mouth from landmarks
        regions = {}
        
        # MediaPipe landmark indices for facial features
        left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        left_eye_pts = landmarks[left_eye_indices]
        regions['left_eye'] = {
            'points': left_eye_pts,
            'bbox': self.get_bbox(left_eye_pts, expand=0.3)
        }
        
        right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        right_eye_pts = landmarks[right_eye_indices]
        regions['right_eye'] = {
            'points': right_eye_pts,
            'bbox': self.get_bbox(right_eye_pts, expand=0.3)
        }
        
        nose_indices = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 129, 358, 370, 94, 141, 242, 462]
        nose_pts = landmarks[nose_indices]
        regions['nose'] = {
            'points': nose_pts,
            'bbox': self.get_bbox(nose_pts, expand=0.4)
        }
        
        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        mouth_pts = landmarks[mouth_indices]
        regions['mouth'] = {
            'points': mouth_pts,
            'bbox': self.get_bbox(mouth_pts, expand=0.25)
        }
        
        face_contour = [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454]
        face_pts = landmarks[face_contour]
        regions['face'] = {
            'points': face_pts,
            'bbox': self.get_bbox(landmarks, expand=0.1)
        }
        
        return regions
    
    def get_bbox(self, points, expand=0.0):
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        w = x_max - x_min
        h = y_max - y_min
        
        x_min = int(x_min - w * expand)
        y_min = int(y_min - h * expand)
        x_max = int(x_max + w * expand)
        y_max = int(y_max + h * expand)
        
        return (x_min, y_min, x_max, y_max)
    
    def extract_animal_features(self, animal_img):
        # Estimate animal feature positions
        h, w = animal_img.shape[:2]
        cx, cy = w // 2, h // 2
        
        features = {}
        
        eye_w, eye_h = int(w * 0.12), int(h * 0.08)
        features['left_eye'] = {
            'bbox': (cx + int(w * 0.08), cy - int(h * 0.15),
                    cx + int(w * 0.08) + eye_w, cy - int(h * 0.15) + eye_h)
        }
        
        features['right_eye'] = {
            'bbox': (cx - int(w * 0.08) - eye_w, cy - int(h * 0.15),
                    cx - int(w * 0.08), cy - int(h * 0.15) + eye_h)
        }
        
        nose_w, nose_h = int(w * 0.2), int(h * 0.25)
        features['nose'] = {
            'bbox': (cx - nose_w // 2, cy - int(h * 0.05),
                    cx + nose_w // 2, cy - int(h * 0.05) + nose_h)
        }
        
        mouth_w, mouth_h = int(w * 0.25), int(h * 0.15)
        features['mouth'] = {
            'bbox': (cx - mouth_w // 2, cy + int(h * 0.15),
                    cx + mouth_w // 2, cy + int(h * 0.15) + mouth_h)
        }
        
        features['face'] = {
            'bbox': (int(w * 0.15), int(h * 0.1), int(w * 0.85), int(h * 0.9))
        }
        
        return features
    
    def warp_region(self, src_img, src_bbox, dst_bbox, mask_points=None):
        # Warp region from src to dst with smooth feathering mask
        x1, y1, x2, y2 = src_bbox
        dx1, dy1, dx2, dy2 = dst_bbox
        
        # Clamp to valid image bounds
        x1, x2 = max(0, x1), min(src_img.shape[1], x2)
        y1, y2 = max(0, y1), min(src_img.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        src_region = src_img[y1:y2, x1:x2]
        if src_region.size == 0:
            return None, None
        
        dst_w = dx2 - dx1
        dst_h = dy2 - dy1
        
        if dst_w <= 0 or dst_h <= 0:
            return None, None
        
        warped = cv2.resize(src_region, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)
        
        # Create feathered mask for smooth blending
        mask = np.ones((dst_h, dst_w), dtype=np.float32)
        feather = min(dst_w, dst_h) // 6
        if feather > 0:
            for i in range(feather):
                alpha = (1 - np.cos(i * np.pi / feather)) / 2  # cosine falloff
                mask[i, :] *= alpha
                mask[-i-1, :] *= alpha
                mask[:, i] *= alpha
                mask[:, -i-1] *= alpha
        
        return warped, mask
    
    def match_color_tone(self, src, tgt):
        # Match color distribution using LAB color space
        src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
        tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Transfer mean and std for each channel
        result = src_lab.copy()
        for i in range(3):
            s_mean, s_std = src_lab[:,:,i].mean(), src_lab[:,:,i].std()
            t_mean, t_std = tgt_lab[:,:,i].mean(), tgt_lab[:,:,i].std()
            
            if s_std > 0:
                result[:,:,i] = ((src_lab[:,:,i] - s_mean) * (t_std / s_std)) + t_mean
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    
    def blend_feature(self, base, feature, mask, bbox, strength=0.8):
        # Blend feature into base image with color matching
        dx1, dy1, dx2, dy2 = bbox
        
        dx1 = max(0, dx1)
        dy1 = max(0, dy1)
        dx2 = min(base.shape[1], dx2)
        dy2 = min(base.shape[0], dy2)
        
        if dx2 <= dx1 or dy2 <= dy1:
            return base
        
        w = dx2 - dx1
        h = dy2 - dy1
        
        if feature.shape[1] != w or feature.shape[0] != h:
            feature = cv2.resize(feature, (w, h))
            mask = cv2.resize(mask, (w, h))
        
        base_region = base[dy1:dy2, dx1:dx2]
        feature = self.match_color_tone(feature, base_region)
        
        mask_3ch = np.stack([mask * strength] * 3, axis=-1)
        blended = (base_region * (1 - mask_3ch) + feature * mask_3ch).astype(np.uint8)
        
        result = base.copy()
        result[dy1:dy2, dx1:dx2] = blended
        
        return result
    
    def add_texture_overlay(self, base, animal, mask, strength=0.2):
        # Add subtle fur texture overlay
        h, w = base.shape[:2]
        animal = cv2.resize(animal, (w, h))
        
        # Extract high-frequency texture details
        gray = cv2.cvtColor(animal, cv2.COLOR_RGB2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        texture = cv2.subtract(gray, blur)
        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)
        
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
        result = base.copy().astype(np.float32)
        result += (texture.astype(np.float32) - 128) * strength * mask_3ch * 0.8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def create_fusion(self, face, animal, landmarks, strength=0.7):
        # Main fusion: map animal features onto human face
        h, w = face.shape[:2]
        result = face.copy()
        
        animal = cv2.resize(animal, (w, h), interpolation=cv2.INTER_CUBIC)
        
        human_feat = self.get_feature_regions(landmarks)
        animal_feat = self.extract_animal_features(animal)
        
        # Create face mask
        hull = cv2.convexHull(landmarks)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        mask = cv2.GaussianBlur(mask, (51, 51), 20)
        
        # Add texture first
        result = self.add_texture_overlay(result, animal, mask, strength * 0.25)
        
        # Map each feature with adaptive strength
        for feat in ['left_eye', 'right_eye', 'nose', 'mouth']:
            if feat in animal_feat and feat in human_feat:
                a_bbox = animal_feat[feat]['bbox']
                h_bbox = human_feat[feat]['bbox']
                
                warped, feat_mask = self.warp_region(animal, a_bbox, h_bbox)
                
                if warped is not None and feat_mask is not None:
                    # Different strength per feature type
                    if feat in ['left_eye', 'right_eye']:
                        s = strength * 0.85
                    elif feat == 'nose':
                        s = strength * 0.80
                    else:
                        s = strength * 0.65
                    
                    result = self.blend_feature(result, warped, feat_mask, h_bbox, s)
        
        return result
    
    def create_visualization(self, face, animal, landmarks):
        h, w = face.shape[:2]
        animal = cv2.resize(animal, (w, h))
        
        face_vis = face.copy()
        human_feat = self.get_feature_regions(landmarks)
        colors = {
            'left_eye': (255, 0, 0),
            'right_eye': (255, 0, 0),
            'nose': (0, 255, 0),
            'mouth': (0, 0, 255)
        }
        
        for name, region in human_feat.items():
            if name in colors:
                x1, y1, x2, y2 = region['bbox']
                cv2.rectangle(face_vis, (x1, y1), (x2, y2), colors[name], 2)
                cv2.putText(face_vis, name.replace('_', ' '), (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1)
        
        animal_vis = animal.copy()
        animal_feat = self.extract_animal_features(animal)
        
        for name, region in animal_feat.items():
            if name in colors:
                x1, y1, x2, y2 = region['bbox']
                cv2.rectangle(animal_vis, (x1, y1), (x2, y2), colors[name], 2)
                cv2.putText(animal_vis, name.replace('_', ' '), (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1)
        
        return face_vis, animal_vis
    
    def create_comparison_grid(self, face, animal, light, mod, strong, landmarks):
        h, w = face.shape[:2]
        
        face_feat, animal_feat = self.create_visualization(face, animal, landmarks)
        
        row1 = np.hstack([face, face_feat, animal_feat])
        row2 = np.hstack([light, mod, strong])
        grid = np.vstack([row1, row2])
        
        labels = [
            (15, 30, "Original Human"),
            (w + 15, 30, "Human Features"),
            (2*w + 15, 30, "Animal Features"),
            (15, h + 30, "Light (50%)"),
            (w + 15, h + 30, "Moderate (70%)"),
            (2*w + 15, h + 30, "Strong (85%)")
        ]
        
        for x, y, text in labels:
            cv2.putText(grid, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3)
            cv2.putText(grid, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        
        return grid
    
    def process(self, face_path, animal_path, output='result'):
        # Process images and generate fusion variants
        face = self.load_image(face_path)
        animal = self.load_image(animal_path)
        landmarks = self.detect_landmarks(face)
        
        # Generate three strength levels
        light = self.create_fusion(face, animal, landmarks, 0.5)
        mod = self.create_fusion(face, animal, landmarks, 0.7)
        strong = self.create_fusion(face, animal, landmarks, 0.85)
        
        cv2.imwrite(f'{output}_light.jpg', cv2.cvtColor(light, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{output}_moderate.jpg', cv2.cvtColor(mod, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{output}_strong.jpg', cv2.cvtColor(strong, cv2.COLOR_RGB2BGR))
        
        grid = self.create_comparison_grid(face, animal, light, mod, strong, landmarks)
        cv2.imwrite(f'{output}_comparison.jpg', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        
        return mod


if __name__ == "__main__":
    fusion = FaceAnimalFusion()
    fusion.process('David-Beckham.jpg', 'wolf.png', 'beckham_wolf')
    fusion.process('C.jpg', 'wolf.png', 'ronaldo_wolf')
