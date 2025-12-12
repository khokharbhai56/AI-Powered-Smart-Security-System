"""
Streamlit Dashboard for AI Security System Monitoring
"""

import sys
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import threading
import time
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import Config
from utils.logger import setup_logger
from real_time_system.detection_pipeline import DetectionPipeline

class SecurityDashboard:
    """Streamlit dashboard for security monitoring"""
    
    # Class-level variables shared across all instances
    _instance = None
    _monitoring_active = False
    _pipeline = None
    _latest_frame = None
    _frame_lock = threading.Lock()
    _monitoring_thread = None
    _alert_history = deque(maxlen=1000)
    _detection_stats = {
        'total_detections': 0,
        'alerts_today': 0,
        'active_tracks': 0,
        'fps': 0
    }

    def __new__(cls):
        """Singleton pattern to maintain state across Streamlit reruns"""
        if cls._instance is None:
            cls._instance = super(SecurityDashboard, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.config = Config()
        self.logger = setup_logger('dashboard')
        self.db_path = Path(self.config.alert_config['db_path'])
        
        # Auto-initialize pipeline
        self.init_pipeline()
    
    @property
    def is_monitoring(self):
        return SecurityDashboard._monitoring_active
    
    @is_monitoring.setter
    def is_monitoring(self, value):
        SecurityDashboard._monitoring_active = value
    
    @property
    def pipeline(self):
        return SecurityDashboard._pipeline
    
    @pipeline.setter
    def pipeline(self, value):
        SecurityDashboard._pipeline = value
    
    @property
    def latest_frame(self):
        return SecurityDashboard._latest_frame
    
    @latest_frame.setter
    def latest_frame(self, value):
        SecurityDashboard._latest_frame = value
    
    @property
    def frame_lock(self):
        return SecurityDashboard._frame_lock
    
    @property
    def monitoring_thread(self):
        return SecurityDashboard._monitoring_thread
    
    @monitoring_thread.setter
    def monitoring_thread(self, value):
        SecurityDashboard._monitoring_thread = value
    
    @property
    def alert_history(self):
        return SecurityDashboard._alert_history
    
    @property
    def detection_stats(self):
        return SecurityDashboard._detection_stats

    def init_pipeline(self):
        """Initialize detection pipeline"""
        try:
            self.pipeline = DetectionPipeline()
            self.logger.info("Detection pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            # Don't raise, allow dashboard to still work

    def load_alert_history(self, days=7):
        """Load alert history from database"""
        if not self.db_path.exists():
            return pd.DataFrame()

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                query = f"""
                SELECT timestamp, action, confidence, alert_triggered
                FROM detections
                WHERE timestamp >= datetime('now', '-{days} days')
                AND alert_triggered = 1
                ORDER BY timestamp DESC
                """
                df = pd.read_sql_query(query, conn)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            self.logger.error(f"Failed to load alert history: {e}")
            return pd.DataFrame()

    def get_detection_stats(self):
        """Get current detection statistics"""
        if not self.db_path.exists():
            return self.detection_stats

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Total detections today
                today = datetime.now().date()
                query_today = f"""
                SELECT COUNT(*) as count FROM detections
                WHERE date(timestamp) = '{today}'
                """
                self.detection_stats['total_detections'] = pd.read_sql_query(query_today, conn)['count'].iloc[0]

                # Alerts today
                query_alerts = f"""
                SELECT COUNT(*) as count FROM detections
                WHERE date(timestamp) = '{today}' AND alert_triggered = 1
                """
                self.detection_stats['alerts_today'] = pd.read_sql_query(query_alerts, conn)['count'].iloc[0]

        except Exception as e:
            self.logger.error(f"Failed to get detection stats: {e}")

        return self.detection_stats

    def add_email_subscriber(self, email):
        """Add email subscriber for alerts"""
        if not self.db_path.exists():
            return False, "Database not initialized"

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO subscribers (email) VALUES (?)
                ''', (email,))
                conn.commit()
            return True, f"✅ Email {email} subscribed successfully!"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"

    def remove_email_subscriber(self, email):
        """Remove email subscriber"""
        if not self.db_path.exists():
            return False, "Database not initialized"

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('DELETE FROM subscribers WHERE email = ?', (email,))
                conn.commit()
            return True, f"✅ Email {email} unsubscribed"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"

    def get_email_subscribers(self):
        """Get all email subscribers"""
        if not self.db_path.exists():
            return []

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                df = pd.read_sql_query('SELECT email FROM subscribers ORDER BY created_at DESC', conn)
                return df['email'].tolist() if not df.empty else []
        except Exception as e:
            self.logger.error(f"Failed to get subscribers: {e}")
            return []

    def create_alert_timeline(self, df):
        """Create alert timeline chart"""
        if df.empty:
            return go.Figure()

        # Group by hour
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_alerts = df.groupby('hour').size().reset_index(name='count')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_alerts['hour'],
            y=hourly_alerts['count'],
            mode='lines+markers',
            name='Alerts per Hour',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title='Alert Timeline (Last 7 Days)',
            xaxis_title='Time',
            yaxis_title='Number of Alerts',
            height=400
        )

        return fig

    def create_alert_distribution(self, df):
        """Create alert type distribution chart"""
        if df.empty:
            return go.Figure()

        action_counts = df['action'].value_counts()

        fig = go.Figure(data=[go.Pie(
            labels=action_counts.index,
            values=action_counts.values,
            title='Alert Types Distribution'
        )])

        fig.update_layout(height=400)
        return fig

    def create_confidence_histogram(self, df):
        """Create confidence score histogram"""
        if df.empty:
            return go.Figure()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['confidence'],
            nbinsx=20,
            name='Confidence Scores',
            marker_color='lightblue'
        ))

        fig.update_layout(
            title='Alert Confidence Distribution',
            xaxis_title='Confidence Score',
            yaxis_title='Frequency',
            height=400
        )

        return fig

    def monitoring_worker(self, source):
        """Background monitoring worker - optimized for non-blocking performance"""
        self.logger.info(f"Starting monitoring with source: {source}")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.error(f"Could not open video source: {source}")
            self.is_monitoring = False
            return

        # Optimize video capture settings for faster frame retrieval
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to get latest frames
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus for stability

        try:
            frame_count = 0
            start_time = time.time()
            last_process_time = time.time()
            process_interval = 0.05  # Process frames every 50ms (20fps max), display at capture rate
            consecutive_failures = 0
            last_reopen_attempt = 0
            reopen_backoff = 2.0  # seconds between reopen attempts

            base_dir = Path(__file__).resolve().parents[2]
            log_path = base_dir / 'logs' / 'monitoring_errors.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)

            def append_log(msg: str):
                try:
                    with open(log_path, 'a', encoding='utf-8') as lf:
                        lf.write(f"{datetime.now().isoformat()} - {msg}\n")
                except Exception:
                    pass

            while self.is_monitoring:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    msg = f"Failed to read frame (attempt {consecutive_failures})"
                    self.logger.warning(msg)
                    append_log(msg)

                    now = time.time()
                    # Try to reopen the capture after a few consecutive failures
                    if consecutive_failures > 5 and (now - last_reopen_attempt) > reopen_backoff:
                        last_reopen_attempt = now
                        self.logger.info("Attempting to reopen video source...")
                        append_log("Attempting to reopen video source")
                        try:
                            cap.release()
                        except Exception:
                            pass
                        time.sleep(0.2)
                        cap = cv2.VideoCapture(source)
                        if cap.isOpened():
                            msg = "Reopened video source successfully"
                            self.logger.info(msg)
                            append_log(msg)
                            consecutive_failures = 0
                            start_time = time.time()
                            frame_count = 0
                            continue
                        else:
                            msg = f"Could not reopen video source: {source}"
                            self.logger.warning(msg)
                            append_log(msg)
                            # If cannot reopen quickly, keep trying but do not stop monitoring
                            time.sleep(0.5)
                            continue
                    else:
                        # brief backoff before next read attempt
                        time.sleep(0.05)
                        continue

                # Reset failure counter on successful read
                consecutive_failures = 0
                frame_count += 1
                current_time = time.time()

                # ALWAYS save latest frame for LIVE display (no wait)
                with self.frame_lock:
                    self.latest_frame = frame.copy()

                # Process frame at controlled interval (non-blocking)
                if self.pipeline and (current_time - last_process_time) > process_interval:
                    try:
                        processed_frame, detections, masks, actions, tracks, alerts = self.pipeline.process_frame(frame)

                        # Update stats - increment total detections
                        self.detection_stats['total_detections'] += len(detections)
                        self.detection_stats['active_tracks'] = len(tracks) if tracks else 0

                        # Calculate FPS (display frames)
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            self.detection_stats['fps'] = frame_count / elapsed

                        # Store alerts
                        for alert in alerts:
                            self.detection_stats['alerts_today'] += 1
                            self.alert_history.append({
                                'timestamp': datetime.now(),
                                'type': alert['type'],
                                'message': alert['message'],
                                'confidence': alert['confidence']
                            })

                        last_process_time = current_time
                    except Exception as e:
                        self.logger.error(f"Frame processing error: {e}", exc_info=True)
                        append_log(f"Frame processing error: {e}")
                        # Continue capturing frames even if processing fails
                        continue

                # Minimal sleep - frames arrive as fast as camera provides them
                time.sleep(0.001)

        except Exception as e:
            self.logger.error(f"Monitoring error: {e}", exc_info=True)
            self.is_monitoring = False

        finally:
            try:
                cap.release()
            except:
                pass
            self.logger.info(f"Monitoring stopped - processed {frame_count} frames")

    def start_monitoring(self, source):
        """Start monitoring"""
        if self.is_monitoring:
            self.logger.info("Monitoring already active")
            return

        self.logger.info(f"Starting monitoring with source: {source}")
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self.monitoring_worker, args=(source,), daemon=True)
        self.monitoring_thread.start()
        self.logger.info(f"Monitoring thread started. is_monitoring={self.is_monitoring}")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.logger.info("Stopping monitoring")
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        self.logger.info(f"Monitoring stopped. is_monitoring={self.is_monitoring}")

    def render_dashboard(self):
        """Render the main dashboard"""
        st.set_page_config(
            page_title="AI Security System Dashboard",
            page_icon="🔒",
            layout="wide"
        )
        
        # Initialize session state for monitoring control
        if 'monitoring_active' not in st.session_state:
            st.session_state['monitoring_active'] = False
        if 'last_frame_time' not in st.session_state:
            st.session_state['last_frame_time'] = 0

        st.title("🔒 AI-Powered Smart Security System Dashboard")

        # Sidebar
        with st.sidebar:
            st.header("Control Panel")

            # Initialize pipeline button
            if st.button("Initialize Detection Pipeline"):
                self.init_pipeline()
                if self.pipeline:
                    st.success("Pipeline initialized successfully!")
                else:
                    st.error("Failed to initialize pipeline")

            # Subscriber / Alert settings
            st.subheader("Alert Recipients")
            email_input = st.text_input("Add recipient email", value="")
            if st.button("Add Subscriber"):
                if not email_input:
                    st.warning("Enter an email address first")
                else:
                    try:
                        with sqlite3.connect(str(self.db_path)) as conn:
                            conn.execute("INSERT OR IGNORE INTO subscribers (email) VALUES (?)", (email_input,))
                        st.success(f"Added subscriber: {email_input}")
                    except Exception as e:
                        st.error(f"Failed to add subscriber: {e}")

            # List subscribers
            try:
                subs = []
                with sqlite3.connect(str(self.db_path)) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT id, email, created_at FROM subscribers ORDER BY created_at DESC")
                    subs = cur.fetchall()
                if subs:
                    for sid, semail, screated in subs:
                        col_a, col_b = st.columns([3,1])
                        col_a.write(f"{semail}")
                        if col_b.button("Remove", key=f"remove_{sid}"):
                            try:
                                with sqlite3.connect(str(self.db_path)) as conn:
                                    conn.execute("DELETE FROM subscribers WHERE id=?", (sid,))
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Failed to remove subscriber: {e}")
                else:
                    st.info("No subscribers configured")
            except Exception:
                st.info("No subscribers configured")

            # Send test alert
            if st.button("Send Test Alert"):
                try:
                    alert_cfg = self.config.alert_config
                    smtp_server = alert_cfg.get('smtp_server')
                    smtp_port = alert_cfg.get('smtp_port')
                    email_user = alert_cfg.get('email_user')
                    email_password = alert_cfg.get('email_password')

                    if not (smtp_server and smtp_port and email_user and email_password):
                        st.error("SMTP configuration missing in config.yaml or environment")
                    else:
                        # Get recipients
                        with sqlite3.connect(str(self.db_path)) as conn:
                            rows = conn.execute("SELECT email FROM subscribers").fetchall()
                        recipients = [r[0] for r in rows]
                        if not recipients:
                            st.warning("No recipients to send to")
                        else:
                            import smtplib
                            from email.mime.text import MIMEText
                            msg = MIMEText("This is a test alert from your AI Security Dashboard.")
                            msg['Subject'] = 'Test Alert'
                            msg['From'] = email_user
                            msg['To'] = ', '.join(recipients)

                            server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
                            server.starttls()
                            server.login(email_user, email_password)
                            server.sendmail(email_user, recipients, msg.as_string())
                            server.quit()
                            st.success(f"Test alert sent to {len(recipients)} recipient(s)")
                except Exception as e:
                    st.error(f"Failed to send test alert: {e}")

            # Monitoring controls
            st.subheader("Monitoring Controls")

            source_options = {
                "Webcam": 0,
                "IP Camera": "rtsp://your_ip_camera_url",
                "Video File": "path/to/video.mp4"
            }

            selected_source = st.selectbox("Video Source", list(source_options.keys()))
            source = source_options[selected_source]

            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ Start Monitoring", width='stretch', key="start_btn"):
                    if self.pipeline is not None:
                        if not self.is_monitoring:
                            self.start_monitoring(source)
                            st.success("✅ Monitoring started! Fetching video stream...")
                        else:
                            st.warning("⚠️ Already monitoring")
                    else:
                        st.error("❌ Pipeline not initialized")
            
            with col2:
                if st.button("⏹️ Stop Monitoring", width='stretch', key="stop_btn"):
                    if self.is_monitoring:
                        self.stop_monitoring()
                        st.success("✅ Monitoring stopped!")
                    else:
                        st.warning("⚠️ Not currently monitoring")

            # Monitoring status indicator
            st.divider()
            col_status = st.columns(3)
            if self.is_monitoring:
                with col_status[0]:
                    st.metric("Status", "🟢 ACTIVE", delta="Live")
            else:
                with col_status[0]:
                    st.metric("Status", "⚫ IDLE", delta="Ready")

        # Main content
        st.markdown("---")
        st.header("🎯 Real-Time Monitoring Dashboard")
        
        # Model Status Panel
        st.subheader("🤖 Model Status")
        col_yolo, col_cnn, col_mask, col_deeport = st.columns(4)
        
        with col_yolo:
            if self.pipeline and self.pipeline.yolo_model:
                st.success("✅ YOLO\nObject Detection")
            else:
                st.error("❌ YOLO\nNot Available")
        
        with col_cnn:
            if self.pipeline and self.pipeline.cnn_model:
                st.success("✅ CNN\nAction Classification")
            else:
                st.error("❌ CNN\nNot Available")
        
        with col_mask:
            if self.pipeline and self.pipeline.mask_rcnn_predictor:
                st.success("✅ Mask R-CNN\nInstance Segmentation")
            else:
                st.error("❌ Mask R-CNN\nNot Available")
        
        with col_deeport:
            if self.pipeline and self.pipeline.tracker:
                st.success("✅ DeepSort\nObject Tracking")
            else:
                st.error("❌ DeepSort\nNot Available")
        
        st.divider()
        
        # Live Metrics
        st.subheader("📊 Live Metrics")
        col1, col2, col3, col4 = st.columns(4)

        # Get current stats
        stats = self.get_detection_stats()
        
        # Also include live stats from monitoring
        live_total = stats['total_detections']
        live_alerts = stats['alerts_today']
        live_tracks = stats['active_tracks']
        live_fps = stats['fps']

        with col1:
            st.metric("📍 Total Detections", live_total, delta=None)

        with col2:
            st.metric("🚨 Alerts Today", live_alerts, delta=None)

        with col3:
            st.metric("👥 Active Tracks", live_tracks, delta=None)

        with col4:
            st.metric("⚡ FPS", f"{live_fps:.1f}", delta=None)

        st.divider()

        # Live View with enhanced layout
        st.subheader("📹 Live Video Feed")
        
        # Create columns for live video and stats
        live_col1, live_col2 = st.columns([2.5, 1])

        with live_col1:
            # Video display section
            video_container = st.container(border=True)
            
            with video_container:
                # Show video feed if monitoring
                if self.is_monitoring:
                    # Check if frame is available
                    if self.latest_frame is not None:
                        with self.frame_lock:
                            frame = self.latest_frame.copy()
                        
                        # Convert BGR to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="📹 Live Stream (Real-time)", width='stretch')
                        
                        # Store current frame info
                        st.session_state['last_frame_time'] = time.time()
                    else:
                        # Waiting for first frame
                        st.info("⏳ Initializing camera feed... Please wait a moment.")
                else:
                    # Not monitoring
                    st.info("📸 Live video feed not available. \n\n**Start monitoring to begin streaming.**")

        with live_col2:
            st.subheader("📊 Live Stats")
            
            # Create containers for metrics
            metric_box = st.container(border=True)
            
            with metric_box:
                # Get current stats
                det_count = self.detection_stats['total_detections']
                alert_count = self.detection_stats['alerts_today']
                track_count = self.detection_stats['active_tracks']
                fps_val = self.detection_stats['fps']
                
                # Display metrics
                st.metric("🎯 Detections", det_count)
                st.metric("🚨 Alerts", alert_count)
                st.metric("👥 Tracks", track_count)
                st.metric("⚡ FPS", f"{fps_val:.1f}")
            
            # Status indicator
            st.markdown("---")
            if self.is_monitoring:
                st.success("🟢 Live Streaming Active")
                st.caption("📊 Stats updating in real-time")
            else:
                st.warning("🔴 Monitoring Inactive")
                st.caption("⏸️ Click 'Start Monitoring' to begin")
        
        # Controlled auto-refresh: refresh the UI at a short interval while
        # monitoring so the displayed frame and live stats update, but
        # avoid rapid restarts that interrupt background threads.
        if self.is_monitoring:
            now = time.time()
            last = st.session_state.get('last_refresh', 0)
            # refresh approximately every 150ms
            if now - last > 0.15:
                st.session_state['last_refresh'] = now
                time.sleep(0.05)
                try:
                    st.experimental_rerun()
                except Exception:
                    # If rerun fails for any reason, continue without crashing
                    pass

        st.divider()

        # Recent alerts
        st.subheader("🚨 Recent Alerts")
        if self.alert_history:
            recent_alerts = list(self.alert_history)[-10:]  # Last 10 alerts
            alert_cols = st.columns([1, 3, 2])
            alert_cols[0].write("**Time**")
            alert_cols[1].write("**Message**")
            alert_cols[2].write("**Confidence**")
            st.divider()
            for alert in reversed(recent_alerts):
                alert_cols = st.columns([1, 3, 2])
                alert_cols[0].write(alert['timestamp'].strftime('%H:%M:%S'))
                alert_cols[1].write(f"🚨 {alert['message']}")
                alert_cols[2].write(f"{alert.get('confidence', 0):.2f}")
        else:
            st.info("✅ No recent alerts - system running normally")

        st.divider()

        # Analytics section
        st.subheader("📊 Analytics & History")
        
        # Load historical data
        alert_df = self.load_alert_history()

        if not alert_df.empty:
            # Charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.plotly_chart(self.create_alert_timeline(alert_df), width='stretch')

            with chart_col2:
                st.plotly_chart(self.create_alert_distribution(alert_df), width='stretch')

            # Confidence histogram
            st.plotly_chart(self.create_confidence_histogram(alert_df), width='stretch')

            # Detailed table
            st.subheader("📋 Detailed Alert Log")
            st.dataframe(alert_df, width='stretch', height=300)

            # Data table
            st.header("Alert History")
            st.dataframe(alert_df)
        else:
            st.info("No alert history available")

        # System status
        st.header("System Status")

        status_col1, status_col2 = st.columns(2)

        with status_col1:
            st.subheader("Model Status")
            model_status = {
                "YOLO Model": "✅ Loaded" if self.pipeline and hasattr(self.pipeline, 'yolo_model') and self.pipeline.yolo_model else "❌ Not Loaded",
                "Mask R-CNN": "✅ Loaded" if self.pipeline and hasattr(self.pipeline, 'mask_rcnn_predictor') and self.pipeline.mask_rcnn_predictor else "❌ Not Loaded",
                "CNN Model": "✅ Loaded" if self.pipeline and hasattr(self.pipeline, 'cnn_model') and self.pipeline.cnn_model else "❌ Not Loaded",
                "DeepSort Tracker": "✅ Loaded" if self.pipeline and hasattr(self.pipeline, 'tracker') and self.pipeline.tracker else "❌ Not Loaded"
            }

            for model, status in model_status.items():
                st.write(f"{model}: {status}")

        with status_col2:
            st.subheader("Alert System Status")
            alert_status = {
                "Email Alerts": "✅ Enabled" if self.config.alert_config['email_enabled'] else "❌ Disabled",
                "Audio Alerts": "✅ Enabled" if self.config.alert_config['audio_enabled'] else "❌ Disabled",
                "Popup Alerts": "✅ Enabled" if self.config.alert_config['popup_enabled'] else "❌ Disabled",
                "Database Logging": "✅ Enabled" if self.config.alert_config['log_to_db'] else "❌ Disabled"
            }

            for alert_type, status in alert_status.items():
                st.write(f"{alert_type}: {status}")

def main():
    """Main dashboard function"""
    # Set page configuration
    st.set_page_config(
        page_title="AI Security System",
        page_icon="🔒",
        layout="wide"
    )
    
    # Auto-refresh when monitoring is active
    dashboard = SecurityDashboard()
    
    # Initialize session state for auto-refresh
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Note: Removed forced auto-refresh loop here to avoid restarting the
    # app while the background monitoring thread is running. UI updates
    # will occur on user interactions (buttons) and normal Streamlit reruns.
    
    dashboard.render_dashboard()

if __name__ == '__main__':
    main()
