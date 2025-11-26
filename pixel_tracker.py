import cv2
import numpy as np
import collections
from collections import deque
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter, detrend, filtfilt
import time
import collections


class PulseRateEstimator:
    """
    Schätzt Pulsfrequenz (BPM) aus einem fortlaufenden PPG-ähnlichen Signal.
    - Füge pro Frame einen Wert (ppg_value) hinzu
    - Nach einigen Sekunden gibt update() eine BPM-Schätzung zurück (oder None)
    """

    def __init__(self, buffer_seconds=10, min_seconds=5):
        self.buffer_seconds = buffer_seconds
        self.min_seconds = min_seconds
        self.values = collections.deque(maxlen=10000)
        self.times = collections.deque(maxlen=10000)

    def _bandpass(self, signal, fs, low_hz=0.7, high_hz=3.0, order=3):
        nyq = 0.5 * fs
        low = low_hz / nyq
        high = high_hz / nyq
        b, a = butter(order, [low, high], btype="band")
        # filtfilt: zero-phase filtering
        return filtfilt(b, a, signal)

    def update(self, ppg_value, t=None):
        """
        ppg_value: Skalar aus deinem ROI (z.B. mean green)
        t: optional Zeitstempel (Sekunden). Wenn None -> time.time()
        return: geschätzte BPM oder None, wenn noch zu wenig Daten
        """
        if t is None:
            t = time.time()

        self.values.append(float(ppg_value))
        self.times.append(float(t))

        # Noch zu wenige Samples?
        if len(self.values) < 10:
            return None

        # Nur letzten buffer_seconds betrachten
        t_arr = np.array(self.times)
        v_arr = np.array(self.values)

        t_max = t_arr[-1]
        t_min = max(t_arr[0], t_max - self.buffer_seconds)

        # Daten im Zeitfenster auswählen
        mask = t_arr >= t_min
        t_arr = t_arr[mask]
        v_arr = v_arr[mask]

        duration = t_arr[-1] - t_arr[0]
        if duration < self.min_seconds:
            # noch nicht genug Historie
            return None

        # Ungleichmäßige Abtastung -> auf gleichmäßige Zeitachse resamplen
        n_samples = len(t_arr)
        fs_est = n_samples / duration  # geschätzte Samplingrate

        t_uniform = np.linspace(t_arr[0], t_arr[-1], n_samples)
        v_uniform = np.interp(t_uniform, t_arr, v_arr)

        # Trend entfernen & bandpass filter
        v_detr = detrend(v_uniform, type="linear")
        v_filt = self._bandpass(v_detr, fs_est, low_hz=0.7, high_hz=3.0)

        # FFT
        freqs = np.fft.rfftfreq(len(v_filt), d=1.0 / fs_est)
        spectrum = np.abs(np.fft.rfft(v_filt))

        # Nur plausible Herzfrequenzen
        mask = (freqs >= 0.7) & (freqs <= 3.0)
        if not np.any(mask):
            return None

        dom_freq = freqs[mask][np.argmax(spectrum[mask])]
        bpm = dom_freq * 60.0
        return bpm

class PulseAmplifier:
    """
    Einfacher rPPG / Eulerian-ähnlicher Verstärker für Puls:
    - extrahiert aus einem ROI den mittleren Grünkanal
    - bandpasst das 1D-Signal im Herzfrequenzbereich
    - verstärkt die gefilterte Amplitude und schreibt sie zurück in den Grünkanal
    """

    def __init__(self, fs=30.0, lowcut=0.7, highcut=3.0, order=3,
                 buffer_size=150, alpha=30.0):
        """
        fs         : Framerate (Hz)
        lowcut     : untere Grenzfrequenz (z.B. 0.7 Hz ~ 42 bpm)
        highcut    : obere Grenzfrequenz (z.B. 3.0 Hz ~ 180 bpm)
        order      : Ordnung des Butterworth-Filters
        buffer_size: Anzahl der Frames im Ringpuffer (Temporal Window)
        alpha      : Verstärkungsfaktor für die Visualisierung
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.buffer_size = buffer_size
        self.alpha = alpha

        self._b, self._a = self._butter_bandpass(lowcut, highcut, fs, order)
        self._signal_buffer = []  # 1D: mittlerer Grünwert pro Frame

    def _butter_bandpass(self, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def process(self, frame, x1, y1, x2, y2):
        """
        frame : BGR-Frame (np.ndarray)
        (x1,y1,x2,y2): ROI-Koordinaten in Frame-Pixeln

        Rückgabe:
            amplified_roi : verstärktes ROI (BGR, uint8) oder None,
            ppg_value     : letzter gefilterter Wert (float) oder None
        """
        # ROI extrahieren & bounds absichern
        h, w = frame.shape[:2]
        x1_cl = max(0, min(w-1, x1))
        x2_cl = max(0, min(w,   x2))
        y1_cl = max(0, min(h-1, y1))
        y2_cl = max(0, min(h,   y2))

        if x2_cl <= x1_cl or y2_cl <= y1_cl:
            # leeres / invalider ROI
            return None, None

        roi = frame[y1_cl:y2_cl, x1_cl:x2_cl]

        # in float umwandeln und auf 0..1 normieren
        roi_f = roi.astype(np.float32) / 255.0

        # Grünkanal extrahieren
        g = roi_f[:, :, 1]

        # 1D-Signal: Mittelwert über ROI
        g_mean = float(g.mean())
        self._signal_buffer.append(g_mean)

        if len(self._signal_buffer) < self.buffer_size:
            # noch nicht genug Historie für sinnvollen Filter
            return roi.copy(), None

        # Puffer auf feste Länge halten
        if len(self._signal_buffer) > self.buffer_size:
            self._signal_buffer.pop(0)

        signal = np.array(self._signal_buffer, dtype=np.float32)

        # Zeitliches Bandpass-Filtering
        filtered = lfilter(self._b, self._a, signal)
        latest_filtered = filtered[-1]

        # Visualisierung: gefilterte Komponente verstärkt auf Grünkanal addieren
        amplified_roi = roi_f.copy()
        # gleicher Zusatz auf alle Pixel im Grünkanal
        amplified_roi[:, :, 1] += self.alpha * latest_filtered

        # Clipping & zurück zu uint8
        amplified_roi = np.clip(amplified_roi, 0.0, 1.0)
        amplified_roi_u8 = (amplified_roi * 255.0).astype(np.uint8)

        return amplified_roi_u8, latest_filtered

class PixelTracker:
    def __init__(self, roi_box, n_points=10, history_length=100, use_bg_subtraction=False, track_shoulders=False, fps=30):
        """
        Initialize the pixel tracker.
        
        Args:
            roi_box: tuple (x1, y1, x2, y2) defining the ROI
            n_points: number of distinctive points to track
            history_length: how many frames to keep in history
            use_bg_subtraction: whether to use background subtraction
            track_shoulders: whether to track shoulder movement and subtract it as noise
            fps: frames per second of the video (for filter design)
        """
        self.x1, self.y1, self.x2, self.y2 = roi_box
        self.n_points = n_points
        self.history_length = history_length
        self.tracked_points = None
        self.initial_positions = None
        self.vertical_displacement_history = deque(maxlen=history_length)
        self.relative_displacement_history = deque(maxlen=history_length)  # Relative to shoulders
        self.shoulder_displacement_history = deque(maxlen=history_length)
        self.frame_count = 0
        self.use_bg_subtraction = use_bg_subtraction
        self.track_shoulders = track_shoulders
        self.fps = fps
        
        # Filtered signal storage
        self.filtered_signal = None
        self.breathing_signal = None
        self.breathing_rate = None
        
        # Shoulder tracking variables
        self.shoulder_midpoint_initial = None
        self.shoulder_midpoint_current = None
        
        # Initialize background subtractor if enabled
        if use_bg_subtraction:
            # MOG2 is adaptive and works well for moving people
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, 
                varThreshold=16, 
                detectShadows=False
            )
    
    def update_shoulder_position(self, left_shoulder, right_shoulder):
        """
        Update the current shoulder midpoint position.
        
        Args:
            left_shoulder: (x, y) coordinates of left shoulder
            right_shoulder: (x, y) coordinates of right shoulder
        """
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2
        
        if self.shoulder_midpoint_initial is None:
            self.shoulder_midpoint_initial = shoulder_midpoint
        
        self.shoulder_midpoint_current = shoulder_midpoint
    
    def update_roi(self, new_roi_box):
        """
        Update the ROI coordinates (useful when tracking a moving object).
        
        Args:
            new_roi_box: tuple (x1, y1, x2, y2) defining the new ROI
        """
        self.x1, self.y1, self.x2, self.y2 = new_roi_box
        
    def find_distinctive_points(self, frame):
        """
        Find the most distinctive (high contrast) points in the ROI.
        Uses Shi-Tomasi corner detection which finds good features to track.
        """
        # Extract ROI
        roi = frame[self.y1:self.y2, self.x1:self.x2]
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Apply background subtraction if enabled
        mask = None
        if self.use_bg_subtraction:
            # Apply background subtractor to the ROI
            fg_mask = self.bg_subtractor.apply(roi)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Dilate slightly to include edges
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
            
            mask = fg_mask
        
        # Use Shi-Tomasi corner detection to find distinctive points
        corners = cv2.goodFeaturesToTrack(
            roi_gray,
            maxCorners=self.n_points,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
            mask=mask
        )
        
        if corners is not None:
            # Convert coordinates back to full frame coordinates
            corners = corners.reshape(-1, 2)
            corners[:, 0] += self.x1
            corners[:, 1] += self.y1
            self.tracked_points = corners.astype(np.float32)
            self.initial_positions = self.tracked_points.copy()
            return True
        return False
    
    def track_points(self, prev_frame, curr_frame, min_points_threshold=5, replenish=True):
        """
        Track points from previous frame to current frame using optical flow.
        
        Args:
            prev_frame: previous frame
            curr_frame: current frame
            min_points_threshold: if points drop below this, find new ones
            replenish: whether to automatically add new points when some are lost
        """
        if self.tracked_points is None or len(self.tracked_points) == 0:
            return False
        
        # Convert frames to grayscale
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
        
        # Calculate optical flow (Lucas-Kanade method)
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            self.tracked_points,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        if new_points is not None:
            # Keep only successfully tracked points
            good_new = new_points[status.flatten() == 1]
            good_old = self.tracked_points[status.flatten() == 1]
            good_initial = self.initial_positions[status.flatten() == 1]
            
            # Filter out points that are outside the ROI
            inside_roi_mask = (
                (good_new[:, 0] >= self.x1) & 
                (good_new[:, 0] <= self.x2) & 
                (good_new[:, 1] >= self.y1) & 
                (good_new[:, 1] <= self.y2)
            )
            
            points_outside = np.sum(~inside_roi_mask)
            if points_outside > 0:
                print(f"Removed {points_outside} points outside ROI")
            
            good_new = good_new[inside_roi_mask]
            good_old = good_old[inside_roi_mask]
            good_initial = good_initial[inside_roi_mask]
            
            if len(good_new) > 0:
                self.tracked_points = good_new
                self.initial_positions = good_initial
                
                # Replenish lost points if enabled and below threshold
                if replenish and len(self.tracked_points) < min_points_threshold:
                    self._replenish_points(curr_frame, min_points_threshold)
                
                # Calculate vertical displacement from initial positions
                vertical_displacements = self.tracked_points[:, 1] - self.initial_positions[:, 1]
                mean_vertical_displacement = np.mean(vertical_displacements)
                
                # Calculate shoulder displacement if tracking shoulders
                shoulder_displacement = 0.0
                if self.track_shoulders and self.shoulder_midpoint_initial is not None and self.shoulder_midpoint_current is not None:
                    shoulder_displacement = self.shoulder_midpoint_current[1] - self.shoulder_midpoint_initial[1]
                    self.shoulder_displacement_history.append(shoulder_displacement)
                    
                    # Calculate relative displacement (thorax movement - shoulder movement)
                    relative_displacement = mean_vertical_displacement - shoulder_displacement
                    self.relative_displacement_history.append(relative_displacement)
                else:
                    self.relative_displacement_history.append(mean_vertical_displacement)
                
                self.vertical_displacement_history.append(mean_vertical_displacement)
                self.frame_count += 1
                return True
        
        return False
    
    def _replenish_points(self, frame, target_count):
        """
        Add new tracking points to replace lost ones.
        """
        points_needed = self.n_points - len(self.tracked_points)
        if points_needed <= 0:
            return
        
        # Extract ROI
        roi = frame[self.y1:self.y2, self.x1:self.x2]
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Create a mask to avoid existing points
        mask = np.ones_like(roi_gray) * 255
        for point in self.tracked_points:
            x, y = point - np.array([self.x1, self.y1])
            cv2.circle(mask, (int(x), int(y)), 15, 0, -1)
        
        # Apply background subtraction if enabled
        if self.use_bg_subtraction:
            fg_mask = self.bg_subtractor.apply(roi)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
            
            # Combine with existing point mask
            mask = cv2.bitwise_and(mask, fg_mask)
        
        # Find new distinctive points
        new_corners = cv2.goodFeaturesToTrack(
            roi_gray,
            maxCorners=points_needed,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
            mask=mask
        )
        
        if new_corners is not None:
            # Convert coordinates back to full frame coordinates
            new_corners = new_corners.reshape(-1, 2)
            new_corners[:, 0] += self.x1
            new_corners[:, 1] += self.y1
            new_corners = new_corners.astype(np.float32)
            
            # Add to tracked points (these new points start at current position)
            self.tracked_points = np.vstack([self.tracked_points, new_corners])
            self.initial_positions = np.vstack([self.initial_positions, new_corners])
            
            print(f"Replenished {len(new_corners)} points (now tracking {len(self.tracked_points)} points)")
    
    def apply_filters(self, use_relative=True):
        """
        Apply two-stage filtering to extract breathing sine wave.
        Stage 1: Low-pass filter to remove high-frequency noise
        Stage 2: Bandpass filter to extract breathing frequency
        
        Args:
            use_relative: Use relative displacement (with shoulder compensation) if True
        
        Returns:
            dict with filtered signals and breathing rate
        """
        # Choose which signal to filter
        if use_relative and len(self.relative_displacement_history) > 0:
            signal_data = np.array(list(self.relative_displacement_history))
        else:
            signal_data = np.array(list(self.vertical_displacement_history))
        
        if len(signal_data) < 30:  # Need minimum samples
            print("Not enough data for filtering (need at least 30 frames)")
            return None
        
        # Stage 1: Low-pass filter (remove high-frequency noise from walking/vibrations)
        # Cutoff at 1 Hz to keep breathing range but remove walking artifacts
        nyquist = self.fps / 2
        lowpass_cutoff = 1 / nyquist
        
        # Design 4th order Butterworth low-pass filter
        b_low, a_low = signal.butter(4, lowpass_cutoff, btype='low')
        
        # Apply filter (use filtfilt for zero phase delay)
        stage1_filtered = signal.filtfilt(b_low, a_low, signal_data)
        
        # Stage 2: Bandpass filter for breathing (0.08 - 0.8 Hz)
        # Typical breathing: 5-48 breaths/min = 0.08-0.8 Hz
        breathing_low = 0.08 / nyquist
        breathing_high = 0.8 / nyquist
        
        # Design 4th order Butterworth bandpass filter
        b_band, a_band = signal.butter(4, [breathing_low, breathing_high], btype='band')
        
        # Apply bandpass filter to extract breathing component
        #breathing_filtered = signal.filtfilt(b_band, a_band, stage1_filtered)
        breathing_filtered = stage1_filtered

        
        # Estimate breathing rate using FFT
        breathing_rate = self._estimate_breathing_rate(breathing_filtered)
        
        # Store results
        self.filtered_signal = stage1_filtered
        self.breathing_signal = breathing_filtered
        self.breathing_rate = breathing_rate
        
        return {
            'lowpass_filtered': stage1_filtered,
            'breathing_signal': breathing_filtered,
            'breathing_rate': breathing_rate,
            'original': signal_data
        }
    
    def _estimate_breathing_rate(self, breathing_signal):
        """
        Estimate breathing rate from filtered signal using FFT.
        
        Returns:
            Breathing rate in breaths per minute
        """
        # Perform FFT
        n = len(breathing_signal)
        yf = fft(breathing_signal)
        xf = fftfreq(n, 1/self.fps)
        
        # Only look at positive frequencies in breathing range (0.15 - 0.8 Hz)
        mask = (xf >= 0.15) & (xf <= 0.8)
        xf_breathing = xf[mask]
        yf_breathing = np.abs(yf[mask])
        
        if len(yf_breathing) == 0:
            return None
        
        # Find dominant frequency
        peak_idx = np.argmax(yf_breathing)
        dominant_freq = xf_breathing[peak_idx]
        
        # Convert to breaths per minute
        breathing_rate = dominant_freq * 60
        
        return breathing_rate
    
    def draw_tracked_points(self, frame):
        """
        Draw the tracked points and their trajectories on the frame.
        """
        if self.tracked_points is None:
            return frame
        
        # Draw ROI
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)
        
        # Draw tracked points
        for i, (point, initial) in enumerate(zip(self.tracked_points, self.initial_positions)):
            x, y = point.ravel()
            x_init, y_init = initial.ravel()
            
            # Draw current position
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Draw line from initial to current position
            cv2.line(frame, (int(x_init), int(y_init)), (int(x), int(y)), (255, 0, 0), 1)
            
            # Draw initial position
            cv2.circle(frame, (int(x_init), int(y_init)), 3, (255, 255, 0), -1)
        
        # Display displacement information
        y_offset = 30
        if len(self.vertical_displacement_history) > 0:
            mean_disp = self.vertical_displacement_history[-1]
            cv2.putText(frame, f"Absolute Disp: {mean_disp:.2f}px", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        if self.track_shoulders and len(self.shoulder_displacement_history) > 0:
            shoulder_disp = self.shoulder_displacement_history[-1]
            cv2.putText(frame, f"Shoulder Disp: {shoulder_disp:.2f}px", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y_offset += 25
            
            if len(self.relative_displacement_history) > 0:
                relative_disp = self.relative_displacement_history[-1]
                cv2.putText(frame, f"Relative Disp: {relative_disp:.2f}px", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def plot_vertical_movement(self, save_path=None, apply_filter=True):
        """
        Plot the mean vertical movement over time.
        Shows absolute displacement, shoulder displacement, relative displacement,
        and filtered breathing signal.
        
        Args:
            save_path: Path to save the plot
            apply_filter: Whether to apply filtering and show breathing signal
        """
        if len(self.vertical_displacement_history) == 0:
            print("No data to plot yet.")
            return
        
        frames = list(range(len(self.vertical_displacement_history)))
        displacements = list(self.vertical_displacement_history)
        
        # Apply filters if requested
        filter_results = None
        if apply_filter:
            filter_results = self.apply_filters(use_relative=self.track_shoulders)
        
        if self.track_shoulders and len(self.shoulder_displacement_history) > 0:
            # Plot with filtering
            if filter_results is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Top plot: All displacements
            ax1.plot(frames, displacements, 'b-', linewidth=1.5, alpha=0.7, label='Absolute (Thorax pixels)')
            ax1.plot(frames, list(self.shoulder_displacement_history), 'g-', linewidth=1.5, alpha=0.7, label='Shoulder movement (noise)')
            ax1.plot(frames, list(self.relative_displacement_history), 'r-', linewidth=2, label='Relative (Thorax - Shoulder)')
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax1.set_xlabel('Frame Number', fontsize=12)
            ax1.set_ylabel('Vertical Displacement (pixels)', fontsize=12)
            ax1.set_title('Thorax Movement: Absolute vs Relative to Shoulders', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Middle plot: Original vs Low-pass filtered
            if filter_results is not None:
                signal_to_plot = list(self.relative_displacement_history)
                ax2.plot(frames, signal_to_plot, 'gray', linewidth=1, alpha=0.5, label='Raw signal')
                ax2.plot(frames, filter_results['lowpass_filtered'], 'b-', linewidth=2, label='Low-pass filtered (< 2.5 Hz)')
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax2.set_xlabel('Frame Number', fontsize=12)
                ax2.set_ylabel('Vertical Displacement (pixels)', fontsize=12)
                ax2.set_title('Stage 1: Low-pass Filter (Noise Removal)', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Bottom plot: Breathing signal (sine wave)
                ax3.plot(frames, filter_results['breathing_signal'], 'r-', linewidth=2.5, label='Breathing signal (0.15-0.8 Hz)')
                ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax3.set_xlabel('Frame Number', fontsize=12)
                ax3.set_ylabel('Vertical Displacement (pixels)', fontsize=12)
                
                title = 'Stage 2: Breathing Sine Wave'
                if filter_results['breathing_rate'] is not None:
                    title += f" (Rate: {filter_results['breathing_rate']:.1f} breaths/min)"
                ax3.set_title(title, fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            else:
                # Just relative displacement if no filtering
                ax2.plot(frames, list(self.relative_displacement_history), 'r-', linewidth=2, label='Relative Displacement')
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax2.set_xlabel('Frame Number', fontsize=12)
                ax2.set_ylabel('Vertical Displacement (pixels)', fontsize=12)
                ax2.set_title('Relative Thorax Movement (Shoulder Noise Removed)', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            plt.tight_layout()
        else:
            # Plot without shoulder tracking
            if filter_results is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
                
                # Original signal
                ax1.plot(frames, displacements, 'b-', linewidth=2, label='Original Displacement')
                ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax1.set_xlabel('Frame Number', fontsize=12)
                ax1.set_ylabel('Vertical Displacement (pixels)', fontsize=12)
                ax1.set_title('Original Vertical Movement', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Low-pass filtered
                ax2.plot(frames, displacements, 'gray', linewidth=1, alpha=0.5, label='Raw signal')
                ax2.plot(frames, filter_results['lowpass_filtered'], 'b-', linewidth=2, label='Low-pass filtered')
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax2.set_xlabel('Frame Number', fontsize=12)
                ax2.set_ylabel('Vertical Displacement (pixels)', fontsize=12)
                ax2.set_title('Stage 1: Low-pass Filter', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Breathing signal
                ax3.plot(frames, filter_results['breathing_signal'], 'r-', linewidth=2.5, label='Breathing signal')
                ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax3.set_xlabel('Frame Number', fontsize=12)
                ax3.set_ylabel('Vertical Displacement (pixels)', fontsize=12)
                
                title = 'Stage 2: Breathing Sine Wave'
                if filter_results['breathing_rate'] is not None:
                    title += f" (Rate: {filter_results['breathing_rate']:.1f} breaths/min)"
                ax3.set_title(title, fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                
                plt.tight_layout()
            else:
                # Simple plot
                plt.figure(figsize=(12, 6))
                plt.plot(frames, displacements, 'b-', linewidth=2, label='Mean Vertical Displacement')
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero displacement')
                plt.xlabel('Frame Number', fontsize=12)
                plt.ylabel('Vertical Displacement (pixels)', fontsize=12)
                plt.title('Mean Vertical Movement of Tracked Points', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
        # Print summary
        if filter_results is not None and filter_results['breathing_rate'] is not None:
            print(f"\n=== Breathing Analysis ===")
            print(f"Breathing Rate: {filter_results['breathing_rate']:.1f} breaths/min")
            print(f"Breathing Amplitude: {np.max(np.abs(filter_results['breathing_signal'])):.2f} pixels")


def main():
    """
    Example usage with video stream or file.
    """
    # Open video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)  # Change to video file path if needed
    
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return
    
    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Define ROI (modify these coordinates as needed)
    height, width = first_frame.shape[:2]
    x1, y1 = width // 4, height // 4
    x2, y2 = 3 * width // 4, 3 * height // 4
    roi_box = (x1, y1, x2, y2)
    
    # Initialize tracker
    tracker = PixelTracker(roi_box, n_points=10, history_length=200)
    
    # Find initial distinctive points
    if not tracker.find_distinctive_points(first_frame):
        print("Error: Could not find distinctive points in ROI")
        return
    
    print(f"Tracking {len(tracker.tracked_points)} points")
    print("Press 'q' to quit, 'r' to reset tracking, 'p' to plot and save")
    
    prev_frame = first_frame.copy()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Track points
            tracker.track_points(prev_frame, frame)
            
            # Draw visualization
            display_frame = frame.copy()
            display_frame = tracker.draw_tracked_points(display_frame)
            
            # Show frame
            cv2.imshow('Pixel Tracker', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset tracking
                tracker.find_distinctive_points(frame)
                print("Tracking reset")
            elif key == ord('p'):
                # Plot and save
                tracker.plot_vertical_movement('/home/claude/vertical_movement.png')
            
            prev_frame = frame.copy()
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final plot
        print(f"\nTracked for {tracker.frame_count} frames")
        tracker.plot_vertical_movement('/home/claude/vertical_movement_final.png')


if __name__ == "__main__":
    main()