import argparse
import cv2
import json
import math
import os
import sys
import time
from datetime import datetime

import mediapipe as mp
import numpy as np

# --- Constants (MediaPipe FaceMesh indices) ---
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
LEFT_PUPIL_LANDMARK = 473
RIGHT_PUPIL_LANDMARK = 468


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def get_bbox_from_landmarks(landmarks, frame_w, frame_h):
    if not landmarks:
        return None
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    if not xs or not ys:
        return None
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)
    x_min = clamp(x_min, 0, frame_w - 1)
    y_min = clamp(y_min, 0, frame_h - 1)
    w = clamp(w, 1, frame_w - x_min)
    h = clamp(h, 1, frame_h - y_min)
    return x_min, y_min, w, h


def process_eye(roi_gray, eye_rect, fallback_thresh=50):
    """Return ((left_perc, right_perc), (cx, cy), threshold_mode, used_thresh) for one eye.
    eye_rect is (ex, ey, ew, eh) relative to roi_gray.
    """
    ex, ey, ew, eh = eye_rect
    roi_h, roi_w = roi_gray.shape
    ex = clamp(ex, 0, roi_w - 1)
    ey = clamp(ey, 0, roi_h - 1)
    ew = clamp(ew, 1, roi_w - ex)
    eh = clamp(eh, 1, roi_h - ey)

    eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
    if eye_roi.size == 0:
        return (50.0, 50.0), (ew // 2, eh // 2), "none", float(fallback_thresh)

    # Histogram equalization can improve contrast
    try:
        eye_eq = cv2.equalizeHist(eye_roi)
    except cv2.error:
        eye_eq = eye_roi

    # Try Otsu first
    used_thresh = None
    threshold_mode = "otsu"
    try:
        _, thr = cv2.threshold(eye_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # The threshold value from Otsu is encoded in the first return of cv2.threshold, but
        # as we used 0 there, it's fine to report a dummy here.
        used_thresh = -1.0
    except cv2.error:
        # Fallback to fixed threshold
        threshold_mode = "fixed"
        used_thresh = float(fallback_thresh)
        _, thr = cv2.threshold(eye_eq, fallback_thresh, 255, cv2.THRESH_BINARY_INV)

    mid_x = ew // 2
    left_half = thr[:, :mid_x]
    right_half = thr[:, mid_x:]

    left_pixels = cv2.countNonZero(left_half)
    right_pixels = cv2.countNonZero(right_half)
    total = left_pixels + right_pixels
    if total == 0:
        left_perc = 50.0
        right_perc = 50.0
    else:
        left_perc = (left_pixels / float(total)) * 100.0
        right_perc = (right_pixels / float(total)) * 100.0

    m = cv2.moments(thr)
    cx, cy = ew // 2, eh // 2
    if m["m00"] != 0:
        cx = int(m["m10"] / m["m00"]) 
        cy = int(m["m01"] / m["m00"]) 

    return (left_perc, right_perc), (cx, cy), threshold_mode, used_thresh


def main():
    parser = argparse.ArgumentParser(description="Gaze (percentile) webcam test - horizontal only")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    parser.add_argument("--show", action="store_true", help="Show visualization window")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video to output folder")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory; defaults to backend/data/temp/gaze_live_<ts>")
    parser.add_argument("--fallback-thresh", type=int, default=50, help="Fallback pupil threshold if Otsu fails")
    parser.add_argument("--ema-alpha", type=float, default=0.5, help="EMA alpha for signal smoothing (0..1)")
    parser.add_argument("--enter-diff", type=float, default=10.0, help="Enter LEFT/RIGHT when |L-R| >= this percent")
    parser.add_argument("--exit-diff", type=float, default=5.0, help="Return to CENTER when |L-R| <= this percent")
    parser.add_argument("--calib-frames", type=int, default=30, help="Calibration frames for baseline signal")
    args = parser.parse_args()

    # Prepare output directory
    if args.outdir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join("backend", "data", "temp", f"gaze_live_{ts}")
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    per_frame_path = os.path.join(outdir, "gaze_frames.jsonl")
    video_path = os.path.join(outdir, "gaze_annotated.mp4")

    # Video capture
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("ERROR: Could not open webcam device", file=sys.stderr)
        sys.exit(1)

    # Try to get FPS/size; provide defaults if unknown
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    horiz_counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
    total_processed = 0

    # Calibration and smoothing state
    calib_vals = []  # signal s = (avg_right - avg_left)
    baseline_s = 0.0
    baseline_locked = False
    ema_s = None
    state = "CENTER"

    start_time = time.time()
    end_time = start_time + args.duration

    with open(per_frame_path, "w", encoding="utf-8") as pf:
        frame_index = 0
        while time.time() < end_time:
            ok, frame = cap.read()
            if not ok:
                break

            ts_ms = int((time.time() - start_time) * 1000)
            frame_bgr = frame
            frame_h, frame_w = frame_bgr.shape[:2]
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # FaceMesh detection
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            face_detected = False
            eyes_detected = False
            label = "N/A"
            confidence = 0.0
            avg_left = 50.0
            avg_right = 50.0
            left_eye_abs = None
            right_eye_abs = None
            left_pupil_abs = None
            right_pupil_abs = None
            threshold_mode_agg = "none"
            used_thresh_agg = -1.0
            ema_signal_out = None

            if res.multi_face_landmarks:
                face_detected = True
                lms = res.multi_face_landmarks[0]
                all_lms = [lm for lm in lms.landmark]
                face_box = get_bbox_from_landmarks(all_lms, frame_w, frame_h)

                if face_box:
                    fx, fy, fw, fh = face_box
                    face_roi_gray = gray[fy:fy + fh, fx:fx + fw]

                    # Eye boxes (absolute)
                    left_eye_lm = [lms.landmark[i] for i in LEFT_EYE_LANDMARKS]
                    right_eye_lm = [lms.landmark[i] for i in RIGHT_EYE_LANDMARKS]
                    left_eye_box_abs = get_bbox_from_landmarks(left_eye_lm, frame_w, frame_h)
                    right_eye_box_abs = get_bbox_from_landmarks(right_eye_lm, frame_w, frame_h)

                    # Pupil landmarks (absolute)
                    try:
                        lp_lm = lms.landmark[LEFT_PUPIL_LANDMARK]
                        rp_lm = lms.landmark[RIGHT_PUPIL_LANDMARK]
                        left_pupil_abs = [int(lp_lm.x * frame_w), int(lp_lm.y * frame_h)]
                        right_pupil_abs = [int(rp_lm.x * frame_w), int(rp_lm.y * frame_h)]
                    except Exception:
                        pass

                    if left_eye_box_abs and right_eye_box_abs:
                        # Convert to face-ROI relative coords
                        lx_abs, ly_abs, lw, lh = left_eye_box_abs
                        rx_abs, ry_abs, rw, rh = right_eye_box_abs
                        lx_rel, ly_rel = lx_abs - fx, ly_abs - fy
                        rx_rel, ry_rel = rx_abs - fx, ry_abs - fy

                        # Clamp to face ROI
                        lx_rel = clamp(lx_rel, 0, max(0, fw - 1))
                        ly_rel = clamp(ly_rel, 0, max(0, fh - 1))
                        rx_rel = clamp(rx_rel, 0, max(0, fw - 1))
                        ry_rel = clamp(ry_rel, 0, max(0, fh - 1))
                        lw = clamp(lw, 1, max(1, fw - lx_rel))
                        lh = clamp(lh, 1, max(1, fh - ly_rel))
                        rw = clamp(rw, 1, max(1, fw - rx_rel))
                        rh = clamp(rh, 1, max(1, fh - ry_rel))

                        left_perc, left_centroid, mode_l, used_l = process_eye(
                            face_roi_gray, (lx_rel, ly_rel, lw, lh), args.fallback_thresh
                        )
                        right_perc, right_centroid, mode_r, used_r = process_eye(
                            face_roi_gray, (rx_rel, ry_rel, rw, rh), args.fallback_thresh
                        )

                        eyes_detected = True
                        threshold_mode_agg = "otsu" if (mode_l == "otsu" or mode_r == "otsu") else "fixed"
                        used_thresh_agg = max(float(used_l), float(used_r))

                        # Averages across both eyes
                        avg_left = (left_perc[0] + right_perc[0]) / 2.0
                        avg_right = (left_perc[1] + right_perc[1]) / 2.0

                        # Signal: positive => RIGHT, negative => LEFT
                        s = (avg_right - avg_left)

                        # Calibration
                        if not baseline_locked:
                            calib_vals.append(s)
                            if len(calib_vals) >= args.calib_frames:
                                baseline_s = float(np.mean(calib_vals))
                                baseline_locked = True

                        adj_s = s - (baseline_s if baseline_locked else 0.0)

                        # EMA smoothing
                        ema_s = adj_s if ema_s is None else (args.ema_alpha * adj_s + (1 - args.ema_alpha) * ema_s)
                        ema_signal_out = float(ema_s)

                        # Hysteresis classification
                        if state == "CENTER":
                            if ema_s <= -args.enter_diff:
                                state = "LEFT"
                            elif ema_s >= args.enter_diff:
                                state = "RIGHT"
                        elif state == "LEFT":
                            if -args.exit_diff <= ema_s <= args.exit_diff:
                                state = "CENTER"
                            elif ema_s >= args.enter_diff:
                                state = "RIGHT"
                        elif state == "RIGHT":
                            if -args.exit_diff <= ema_s <= args.exit_diff:
                                state = "CENTER"
                            elif ema_s <= -args.enter_diff:
                                state = "LEFT"

                        label = state
                        confidence = float(abs(adj_s))  # percent difference scale

                        # --- Visualization ---
                        if args.show or writer is not None:
                            # Draw face, eyes, midlines, centroids
                            cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 1)
                            # Left eye abs
                            cv2.rectangle(frame_bgr, (lx_abs, ly_abs), (lx_abs + lw, ly_abs + lh), (0, 255, 0), 1)
                            # Right eye abs
                            cv2.rectangle(frame_bgr, (rx_abs, ry_abs), (rx_abs + rw, ry_abs + rh), (0, 255, 0), 1)
                            # Midlines
                            cv2.line(frame_bgr, (lx_abs + lw // 2, ly_abs), (lx_abs + lw // 2, ly_abs + lh), (255, 255, 0), 1)
                            cv2.line(frame_bgr, (rx_abs + rw // 2, ry_abs), (rx_abs + rw // 2, ry_abs + rh), (255, 255, 0), 1)
                            # Pupil centroids (relative -> absolute)
                            lcx, lcy = left_centroid
                            rcx, rcy = right_centroid
                            cv2.circle(frame_bgr, (lx_abs + int(lcx), ly_abs + int(lcy)), 2, (0, 0, 255), -1)
                            cv2.circle(frame_bgr, (rx_abs + int(rcx), ry_abs + int(rcy)), 2, (0, 0, 255), -1)
                            # Label text
                            label_text = f"{label}  L={avg_left:.1f}% R={avg_right:.1f}% | s={adj_s:.1f} ~{ema_s:.1f} | mode={threshold_mode_agg}"
                            cv2.putText(frame_bgr, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            # Per-frame JSONL
            rec = {
                "frame_index": frame_index,
                "ts_ms": ts_ms,
                "face_detected": bool(face_detected),
                "eyes_detected": bool(eyes_detected),
                "label": label,
                "confidence": round(float(confidence), 3),
                "avg_left_perc": round(float(avg_left), 2),
                "avg_right_perc": round(float(avg_right), 2),
                "left_pupil_abs": left_pupil_abs,
                "right_pupil_abs": right_pupil_abs,
                "debug": {
                    "threshold_mode": threshold_mode_agg,
                    "used_threshold": used_thresh_agg,
                    "baseline_locked": baseline_locked,
                    "baseline_s": round(float(baseline_s), 3),
                    "ema_signal": round(float(ema_signal_out), 3) if ema_signal_out is not None else None,
                },
            }
            pf.write(json.dumps(rec) + "\n")

            # Update counts
            if label in ("LEFT", "CENTER", "RIGHT"):
                horiz_counts[label] += 1
                total_processed += 1

            # Show or save
            if writer is not None:
                writer.write(frame_bgr)
            if args.show:
                cv2.imshow("Gaze Percentile (Horizontal Only)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_index += 1

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    # Final summary
    total = max(1, total_processed)
    pct = {k: (v / float(total)) * 100.0 for k, v in horiz_counts.items()}
    summary = {
        "frames_processed": frame_index,
        "frames_with_valid_label": total_processed,
        "horiz_counts": horiz_counts,
        "horiz_pct": {k: round(v, 2) for k, v in pct.items()},
        "per_frame_path": per_frame_path,
        "video_path": video_path if os.path.exists(video_path) else None,
        "thresholds": {
            "enter_diff": args.enter_diff,
            "exit_diff": args.exit_diff,
            "fallback_thresh": args.fallback_thresh,
            "ema_alpha": args.ema_alpha,
            "calib_frames": args.calib_frames,
        },
    }
    print("Summary:", json.dumps(summary))


if __name__ == "__main__":
    main()
