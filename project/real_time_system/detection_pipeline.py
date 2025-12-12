"""
Real-time Detection Pipeline for AI Security System
Combines YOLO, Mask R-CNN, and CNN for comprehensive surveillance
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Optional imports - detectron2 may not be available
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except (ImportError, Exception):
    # pygame may fail to load due to pkg_resources issues
    PYGAME_AVAILABLE = False

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import time
from collections import deque
import sqlite3
from datetime import datetime
import logging
from utils.config import Config
from utils.logger import setup_logger, log_alert

class DetectionPipeline:
    """Main detection pipeline combining all models"""

    def __init__(self, config_path='config.yaml'):
        self.config = Config(config_path)
        self.logger = setup_logger('detection_pipeline')

        # Initialize models
        self.yolo_model = None
        self.mask_rcnn_predictor = None
        self.cnn_model = None
        self.tracker = None

        self._load_models()

        # Alert system
        self.alert_cooldown = self.config.alert_config['alert_cooldown']
        self.last_alert_time = 0

        # FPS tracking
        self.fps_buffer = deque(maxlen=30)
        self.frame_count = 0

        # Person tracking
        self.person_tracks = {}

        # Heatmap
        self.heatmap = np.zeros((self.config.detection_config['heatmap_resolution'][1],
                               self.config.detection_config['heatmap_resolution'][0]), dtype=np.float32)

        # Latest frame for streaming
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Audio system
        if PYGAME_AVAILABLE and self.config.alert_config['audio_enabled']:
            pygame.mixer.init()

        # Database
        if self.config.alert_config['log_to_db']:
            self._init_database()

    def _load_models(self):
        """Load all ML models"""
        try:
            # YOLO model
            try:
                if Path(self.config.model_paths['yolo_model_path']).exists():
                    self.yolo_model = YOLO(self.config.model_paths['yolo_model_path'])
                    self.logger.info("✓ YOLO model loaded")
                else:
                    self.logger.warning("YOLO model not found, using default YOLOv8n")
                    self.yolo_model = YOLO('yolov8n.pt')  # Use smaller model
                    self.logger.info("✓ YOLO model loaded (default)")
            except Exception as e:
                self.logger.warning(f"YOLO loading failed: {e}, using placeholder")
                self.yolo_model = self._create_placeholder_yolo_model()

            # Mask R-CNN model
            if DETECTRON2_AVAILABLE and Path(self.config.model_paths['mask_rcnn_model_path']).exists():
                try:
                    cfg = get_cfg()
                    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
                    cfg.MODEL.WEIGHTS = str(self.config.model_paths['mask_rcnn_model_path'])
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
                    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
                    self.mask_rcnn_predictor = DefaultPredictor(cfg)
                    self.logger.info("✓ Mask R-CNN model loaded")
                except Exception as e:
                    self.logger.warning(f"Mask R-CNN load failed: {e}")
                    self._create_placeholder_mask_rcnn_model()
            else:
                if not DETECTRON2_AVAILABLE:
                    self.logger.warning("⚠ Detectron2 not installed - creating placeholder Mask R-CNN")
                else:
                    self.logger.warning("⚠ Mask R-CNN model file not found - using placeholder")
                self._create_placeholder_mask_rcnn_model()

            # CNN model
            if Path(self.config.model_paths['cnn_model_path']).exists():
                try:
                    self.cnn_model = torch.load(self.config.model_paths['cnn_model_path'], map_location='cpu')
                    self.cnn_model.eval()
                    self.logger.info("✓ CNN model loaded")
                except Exception as e:
                    self.logger.warning(f"CNN model load failed: {e}")
                    self._create_placeholder_cnn_model()
            else:
                self.logger.warning("⚠ CNN model file not found - using placeholder model")
                self._create_placeholder_cnn_model()

            # DeepSort tracker - now with placeholder
            try:
                self.tracker = self._create_placeholder_deepsort_tracker()
                self.logger.info("✓ DeepSort tracker initialized")
            except Exception as e:
                self.logger.warning(f"DeepSort initialization failed: {e}, using placeholder")
                self.tracker = self._create_placeholder_deepsort_tracker()

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def _create_placeholder_yolo_model(self):
        """Create a simple placeholder YOLO-like model for testing"""
        self.logger.info("Creating placeholder YOLO model for testing")
        
        class PlaceholderYOLO:
            def __init__(self):
                self.names = {0: 'person', 1: 'gun', 2: 'knife'}
            
            def __call__(self, frame, conf=0.5):
                # Return empty detections
                class Result:
                    def __init__(self):
                        self.boxes = []
                        self.names = {0: 'person', 1: 'gun', 2: 'knife'}
                return [Result()]
        
        return PlaceholderYOLO()

    def _create_placeholder_cnn_model(self):
        """Create a simple placeholder CNN model for action classification"""
        self.logger.info("Creating placeholder CNN model for testing")
        
        class SimpleCNN(torch.nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                )
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(128 * 28 * 28, 256),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(256, 5),  # 5 action classes
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        self.cnn_model = SimpleCNN()
        self.cnn_model.eval()
        self.logger.info("✓ Placeholder CNN model created for testing")

    def _create_placeholder_mask_rcnn_model(self):
        """Create a simple placeholder Mask R-CNN model for segmentation"""
        self.logger.info("Creating placeholder Mask R-CNN model for testing")
        
        class PlaceholderMaskRCNN:
            def __init__(self):
                self.names = {0: 'person'}
            
            def __call__(self, frame):
                # Return empty masks/instances
                class Instances:
                    def __init__(self):
                        self.pred_masks = torch.tensor([], dtype=torch.bool)
                        self.pred_boxes = torch.tensor([], dtype=torch.float32)
                        self.scores = torch.tensor([], dtype=torch.float32)
                
                class Output:
                    def __init__(self):
                        self.instances = Instances()
                
                return Output()
        
        self.mask_rcnn_predictor = PlaceholderMaskRCNN()
        self.logger.info("✓ Placeholder Mask R-CNN model created for testing")

    def _create_placeholder_deepsort_tracker(self):
        """Create a simple placeholder DeepSort tracker for person tracking"""
        self.logger.info("Creating placeholder DeepSort tracker for testing")
        
        class PlaceholderDeepSort:
            def __init__(self):
                self.next_id = 0
                self.tracks = {}
            
            def update_tracks(self, detections):
                """Simple tracking using centroid distance"""
                current_tracks = {}
                
                for det in detections:
                    bbox = det.get('bbox', [0, 0, 10, 10])
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Find nearest track
                    best_id = None
                    best_dist = float('inf')
                    
                    for track_id, (prev_cx, prev_cy) in self.tracks.items():
                        dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                        if dist < 50 and dist < best_dist:  # 50 pixel threshold
                            best_dist = dist
                            best_id = track_id
                    
                    # Assign ID
                    if best_id is None:
                        best_id = self.next_id
                        self.next_id += 1
                    
                    current_tracks[best_id] = (cx, cy)
                    det['track_id'] = best_id
                
                self.tracks = current_tracks
                return detections
        
        self.tracker = PlaceholderDeepSort()
        self.logger.info("✓ Placeholder DeepSort tracker created for testing")
        return self.tracker

    def _init_database(self):
        """Initialize SQLite database for logging"""
        db_path = Path(self.config.alert_config['db_path'])
        db_path.parent.mkdir(exist_ok=True)

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    frame_number INTEGER,
                    person_id INTEGER,
                    action TEXT,
                    confidence REAL,
                    bbox TEXT,
                    alert_triggered INTEGER DEFAULT 0
                )
            ''')
            # Subscribers table for alert recipients
            conn.execute('''
                CREATE TABLE IF NOT EXISTS subscribers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def _preprocess_frame(self, frame):
        """Preprocess frame for detection"""
        # Resize if needed
        height, width = frame.shape[:2]
        if width > 1920:  # Max width
            ratio = 1920 / width
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (new_width, new_height))

        return frame

    def _run_yolo_detection(self, frame):
        """Run YOLO detection"""
        if self.yolo_model is None:
            return []

        results = self.yolo_model(frame, conf=self.config.detection_config['confidence_threshold'])
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': cls,
                    'class_name': result.names[cls]
                })

        return detections

    def _run_mask_rcnn_segmentation(self, frame, person_bboxes):
        """Run Mask R-CNN segmentation on person detections"""
        if self.mask_rcnn_predictor is None:
            return []

        masks = []
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            if person_crop.size == 0:
                continue

            try:
                # Run prediction
                outputs = self.mask_rcnn_predictor(person_crop)

                # Check if outputs has instances (real model) or is placeholder
                if hasattr(outputs, 'instances'):
                    # Real Mask R-CNN model
                    pred_masks = outputs.instances.pred_masks.cpu().numpy()
                    pred_boxes = outputs.instances.pred_boxes.tensor.cpu().numpy()
                    pred_classes = outputs.instances.pred_classes.cpu().numpy()

                    for mask, box, cls in zip(pred_masks, pred_boxes, pred_classes):
                        if cls == 0:  # Person class
                            # Adjust mask coordinates to original frame
                            mask_resized = cv2.resize(mask.astype(np.uint8), (int(x2-x1), int(y2-y1)))
                            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            full_mask[int(y1):int(y2), int(x1):int(x2)] = mask_resized

                            masks.append({
                                'mask': full_mask,
                                'bbox': [x1, y1, x2, y2]
                            })
                else:
                    # Placeholder model - skip
                    pass
            except Exception as e:
                self.logger.debug(f"Mask R-CNN segmentation error: {e}")
                pass

        return masks

    def _run_cnn_classification(self, frame, person_bboxes):
        """Run CNN action classification on person crops"""
        if self.cnn_model is None:
            return []

        actions = []
        class_names = ['fighting', 'falling', 'running', 'stealing', 'normal_walking']
        
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            if person_crop.size == 0:
                continue

            try:
                # Preprocess for CNN
                person_crop_resized = cv2.resize(person_crop, (224, 224))
                person_crop_norm = person_crop_resized.astype(np.float32) / 255.0
                person_crop_tensor = np.transpose(person_crop_norm, (2, 0, 1))
                person_crop_tensor = torch.from_numpy(person_crop_tensor).unsqueeze(0)

                # Run prediction
                with torch.no_grad():
                    outputs = self.cnn_model(person_crop_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_class].item()

                actions.append({
                    'action': class_names[pred_class],
                    'confidence': confidence,
                    'bbox': bbox
                })
            except Exception as e:
                self.logger.debug(f"CNN classification error: {e}")
                continue

        return actions

    def _update_tracking(self, detections, frame):
        """Update DeepSort tracking"""
        if self.tracker is None:
            return []

        # For placeholder tracker, just return simple tracked objects
        tracked_objects = []
        for i, det in enumerate(detections):
            if det['class_name'] == 'person':
                bbox = det['bbox']
                tracked_objects.append({
                    'id': i,
                    'bbox': bbox,
                    'centroid': [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                })

        return tracked_objects

    def _check_suspicious_activity(self, detections, actions, tracks):
        """Check for suspicious activity using rule engine"""
        alerts = []

        # Rule 1: Fighting detection
        for action in actions:
            if action['action'] == 'fighting' and action['confidence'] > 0.7:
                alerts.append({
                    'type': 'fighting',
                    'confidence': action['confidence'],
                    'bbox': action['bbox'],
                    'message': f'Fighting detected with {action["confidence"]:.2f} confidence'
                })

        # Rule 2: Weapon detection
        for det in detections:
            if det['class_name'] in ['gun', 'knife'] and det['confidence'] > 0.6:
                alerts.append({
                    'type': 'weapon',
                    'confidence': det['confidence'],
                    'bbox': det['bbox'],
                    'message': f'{det["class_name"].title()} detected with {det["confidence"]:.2f} confidence'
                })

        # Rule 3: Running in restricted areas
        for action in actions:
            if action['action'] == 'running' and action['confidence'] > 0.8:
                # Check if in restricted area (implement area logic)
                alerts.append({
                    'type': 'running',
                    'confidence': action['confidence'],
                    'bbox': action['bbox'],
                    'message': f'Running detected with {action["confidence"]:.2f} confidence'
                })

        # Rule 4: Abandoned object detection (simplified)
        # Implement based on track persistence

        return alerts

    def _trigger_alert(self, alert):
        """Trigger alert system"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return

        self.last_alert_time = current_time

        # Log alert
        log_alert(self.logger, alert['type'], alert['message'],
                 alert['confidence'], alert['bbox'])

        # Audio alert
        if self.config.alert_config['audio_enabled']:
            threading.Thread(target=self._play_alert_sound).start()

        # Email alert
        if self.config.alert_config['email_enabled']:
            threading.Thread(target=self._send_email_alert, args=(alert,)).start()

        # Popup alert
        if self.config.alert_config['popup_enabled']:
            threading.Thread(target=self._show_popup_alert, args=(alert,)).start()

        # Screenshot
        threading.Thread(target=self._save_screenshot, args=(alert,)).start()

    def _play_alert_sound(self):
        """Play alert sound"""
        if not PYGAME_AVAILABLE:
            self.logger.debug("Pygame not available, skipping audio alert")
            return
        try:
            pygame.mixer.music.load('assets/alert.wav')
            pygame.mixer.music.play()
        except Exception as e:
            self.logger.error(f"Audio alert failed: {e}")

    def _send_email_alert(self, alert):
        """Send email alert"""
        try:
            smtp_server = self.config.alert_config.get('smtp_server')
            smtp_port = self.config.alert_config.get('smtp_port')
            email_user = self.config.alert_config.get('email_user')
            email_password = self.config.alert_config.get('email_password')
            
            # Validate configuration
            if not all([smtp_server, smtp_port, email_user, email_password]):
                self.logger.warning("Email configuration incomplete - check config.yaml")
                return
            
            # Check for placeholder values
            if 'your_email' in email_user or 'your_' in email_password:
                self.logger.warning("Email configuration not set up - using placeholder values")
                return
            
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = ', '.join(self.config.alert_config.get('alert_recipients', []))
            msg['Subject'] = f'Security Alert: {alert["type"].title()}'

            body = f"""
Security Alert Detected!

Type: {alert['type'].title()}
Message: {alert['message']}
Confidence: {alert['confidence']:.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the security system immediately.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent successfully for {alert['type']}")

        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"SMTP Auth failed: {e}")
            self.logger.warning("Check email/password in config.yaml. For Gmail use App Password: https://myaccount.google.com/apppasswords")
        except smtplib.SMTPException as e:
            self.logger.error(f"SMTP error: {e}")
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")

    def _show_popup_alert(self, alert):
        """Show popup alert"""
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("Security Alert", alert['message'])
            root.destroy()

        except Exception as e:
            self.logger.error(f"Popup alert failed: {e}")

    def _save_screenshot(self, alert):
        """Save screenshot with alert"""
        try:
            # This would be called with current frame
            # Implementation depends on frame capture
            pass
        except Exception as e:
            self.logger.error(f"Screenshot save failed: {e}")

    def _update_heatmap(self, tracks):
        """Update activity heatmap"""
        for track in tracks:
            centroid = track['centroid']
            x, y = int(centroid[0]), int(centroid[1])

            if 0 <= x < self.heatmap.shape[1] and 0 <= y < self.heatmap.shape[0]:
                self.heatmap[y, x] += 1

    def _draw_overlay(self, frame, detections, masks, actions, tracks, alerts):
        """Draw detection overlay on frame"""
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['class_name']}: {det['confidence']:.2f}",
                       (int(bbox[0]), int(bbox[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw masks
        for mask_info in masks:
            mask = mask_info['mask']
            frame[mask > 0] = frame[mask > 0] * 0.7 + np.array([0, 0, 255]) * 0.3

        # Draw tracks
        for track in tracks:
            bbox = track['bbox']
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track['id']}",
                       (int(bbox[0]), int(bbox[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw alerts
        for alert in alerts:
            bbox = alert['bbox']
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
            cv2.putText(frame, f"ALERT: {alert['type'].upper()}",
                       (int(bbox[0]), int(bbox[1])-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw FPS
        if self.fps_buffer:
            fps = len(self.fps_buffer) / sum(self.fps_buffer)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def process_frame(self, frame):
        """Process a single frame through the entire pipeline"""
        start_time = time.time()

        # Preprocess
        frame = self._preprocess_frame(frame)

        # YOLO detection
        detections = self._run_yolo_detection(frame)

        # Extract person bboxes
        person_bboxes = [det['bbox'] for det in detections if det['class_name'] == 'person']

        # Mask R-CNN segmentation
        masks = self._run_mask_rcnn_segmentation(frame, person_bboxes)

        # CNN action classification
        actions = self._run_cnn_classification(frame, person_bboxes)

        # Update tracking
        tracks = self._update_tracking(detections, frame)

        # Check for suspicious activity
        alerts = self._check_suspicious_activity(detections, actions, tracks)

        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert)

        # Update heatmap
        self._update_heatmap(tracks)

        # Draw overlay
        frame = self._draw_overlay(frame, detections, masks, actions, tracks, alerts)

        # Update FPS
        processing_time = time.time() - start_time
        self.fps_buffer.append(processing_time)

        # Save latest frame as JPEG bytes for dashboard live view
        try:
            _, imbuf = cv2.imencode('.jpg', frame)
            self.latest_frame_jpg = imbuf.tobytes()
        except Exception:
            self.latest_frame_jpg = None

        # Save quick live stats
        try:
            self.latest_stats = {
                'detections': len(detections),
                'alerts': len(alerts),
                'tracks': len(tracks) if tracks else 0,
                'fps': (len(self.fps_buffer) / sum(self.fps_buffer)) if self.fps_buffer and sum(self.fps_buffer) > 0 else 0.0
            }
        except Exception:
            self.latest_stats = {'detections': 0, 'alerts': 0, 'tracks': 0, 'fps': 0.0}

        return frame, detections, masks, actions, tracks, alerts

    def run(self, source=0):
        """Run the detection pipeline"""
        self.logger.info(f"Starting detection pipeline with source: {source}")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.error(f"Could not open video source: {source}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame, detections, masks, actions, tracks, alerts = self.process_frame(frame)

                # Display
                cv2.imshow('AI Security System', processed_frame)

                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")

        except Exception as e:
            self.logger.error(f"Detection pipeline error: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Detection pipeline stopped")
