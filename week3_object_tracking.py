"""
Week 3 Assignment: Object Tracking with Feature Matching and Optical Flow

Two complementary techniques are demonstrated:

  1. ORB + Brute Force Matching — compares two specific frames by extracting
     binary descriptors and matching them globally. This is a feature MATCHING
     technique: it answers "which patch in frame A corresponds to which patch
     in frame B?" without knowing anything about the frames in between.

  2. Lucas-Kanade Optical Flow — continuously tracks a fixed set of corner
     points from one frame to the very next frame. This is a motion TRACKING
     technique: it answers "where did these exact pixels move to?" across the
     full video in real time.

Together they satisfy the assignment's requirement for both keypoint matching
(ORB/BFMatcher) and continuous motion tracking (LK Optical Flow).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\shweiss\Downloads\IMG_3260.mov"
OUTPUT_VIDEO = "tracked_output.mp4"
SCREENSHOT_FIRST = "first_frame_keypoints.png"
SCREENSHOT_MID = "tracking_midpoint.png"
SCREENSHOT_FINAL = "tracking_final.png"
TRAJECTORY_PLOT = "trajectory_plot.png"
ORB_MATCHES_IMG = "orb_matches.png"

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize=(21, 21),       # Size of the search window at each pyramid level
    maxLevel=3,             # Number of pyramid levels
    criteria=(              # Stop iterating when accuracy or iteration count is met
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,    # max iterations
        0.01   # epsilon
    )
)

# Shi-Tomasi corner detection parameters (used inside the ROI for LK flow)
FEATURE_PARAMS = dict(
    maxCorners=150,         # Maximum number of keypoints to detect
    qualityLevel=0.01,      # Minimum quality of a corner (0–1)
    minDistance=7,          # Minimum pixel distance between returned corners
    blockSize=7             # Size of the neighborhood for corner detection
)

# ─────────────────────────────────────────────
# STEP 1: LOAD VIDEO
# ─────────────────────────────────────────────
print("Loading video...")
video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    raise FileNotFoundError(
        f"Could not open video at: {VIDEO_PATH}\n"
        "Check that the file path is correct and the file exists."
    )

fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video loaded successfully: {os.path.basename(VIDEO_PATH)}")
print(f"  Resolution  : {frame_width} x {frame_height}")
print(f"  FPS         : {fps:.2f}")
print(f"  Total frames: {total_frames}")

# ─────────────────────────────────────────────
# STEP 2: READ FIRST FRAME AND SELECT ROI
# ─────────────────────────────────────────────
ret, first_frame = video.read()
if not ret:
    video.release()
    raise RuntimeError("Failed to read the first frame from the video.")

# Resize display if frame is very large (keeps the ROI window manageable on screen)
display_scale = 1.0
MAX_DISPLAY_DIM = 900
if max(frame_width, frame_height) > MAX_DISPLAY_DIM:
    display_scale = MAX_DISPLAY_DIM / max(frame_width, frame_height)
    display_frame = cv2.resize(first_frame, (0, 0), fx=display_scale, fy=display_scale)
else:
    display_frame = first_frame.copy()

print("\nA window will open showing the first frame.")
print("Draw a rectangle around the CEREAL BOX and press ENTER (or SPACE) to confirm.")
print("Press 'c' to cancel.\n")

# cv2.selectROI returns (x, y, w, h) in the coordinate space of display_frame
roi_display = cv2.selectROI("Select ROI — draw around cereal box, press ENTER", display_frame, False)
cv2.destroyAllWindows()

if roi_display == (0, 0, 0, 0):
    video.release()
    raise RuntimeError("No ROI selected. Please re-run the script and draw a rectangle.")

# Scale ROI coordinates back to the original frame resolution
x = int(roi_display[0] / display_scale)
y = int(roi_display[1] / display_scale)
w = int(roi_display[2] / display_scale)
h = int(roi_display[3] / display_scale)

print(f"ROI selected (original resolution): x={x}, y={y}, w={w}, h={h}")

# ─────────────────────────────────────────────
# STEP 3: DETECT KEYPOINTS INSIDE THE ROI (Shi-Tomasi for LK flow)
# ─────────────────────────────────────────────
# Convert the first frame to grayscale — optical flow requires single-channel images
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Crop the grayscale ROI region to detect features only within the object boundary
roi_gray = first_gray[y:y + h, x:x + w]

# Detect strong corners (Shi-Tomasi) inside the ROI
keypoints_roi = cv2.goodFeaturesToTrack(roi_gray, mask=None, **FEATURE_PARAMS)

if keypoints_roi is None or len(keypoints_roi) == 0:
    video.release()
    raise RuntimeError(
        "No keypoints detected in the selected ROI. "
        "Try selecting a region with more texture or adjusting FEATURE_PARAMS."
    )

# Translate keypoint coordinates from ROI-space to full-frame space
keypoints_roi[:, 0, 0] += x
keypoints_roi[:, 0, 1] += y

print(f"Initial keypoints detected (Shi-Tomasi for LK flow): {len(keypoints_roi)}")

# Visualize keypoints on the first frame and save as a screenshot
first_frame_vis = first_frame.copy()
cv2.rectangle(first_frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)  # ROI box in green
for kp in keypoints_roi:
    cx, cy = int(kp[0][0]), int(kp[0][1])
    cv2.circle(first_frame_vis, (cx, cy), 4, (0, 0, 255), -1)  # Red dots = keypoints

cv2.imwrite(SCREENSHOT_FIRST, first_frame_vis)
print(f"Saved: {SCREENSHOT_FIRST}")

# ─────────────────────────────────────────────
# STEP 4: ORB FEATURE MATCHING BETWEEN TWO FRAMES
#
# ORB (Oriented FAST and Rotated BRIEF) detects keypoints and computes a
# compact binary descriptor for each one. Brute Force Matching then compares
# every descriptor in frame A against every descriptor in frame B using the
# Hamming distance (appropriate for binary descriptors). The result is a visual
# that shows which features in the early frame correspond to features in a later
# frame — demonstrating that the cereal box is recognizable across time even
# after significant movement.
#
# This is distinct from Lucas-Kanade Optical Flow: ORB+BFMatcher works on any
# two independent frames with no knowledge of the frames in between, while LK
# flow incrementally tracks specific pixel neighborhoods frame-by-frame.
# ─────────────────────────────────────────────
print("\n--- ORB Feature Matching ---")

# Choose an early frame (frame 10) and a later frame (~75% through the video)
# so that the cereal box has clearly moved between the two samples.
EARLY_FRAME_IDX = 10
LATE_FRAME_IDX = int(total_frames * 0.75)

def read_frame_at(cap, index):
    """Seek to a specific frame index and return the frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {index} from video.")
    return frame

# Read the two comparison frames
frame_early = read_frame_at(video, EARLY_FRAME_IDX)
frame_late = read_frame_at(video, LATE_FRAME_IDX)

# Convert both frames to grayscale — ORB operates on single-channel images
gray_early = cv2.cvtColor(frame_early, cv2.COLOR_BGR2GRAY)
gray_late = cv2.cvtColor(frame_late, cv2.COLOR_BGR2GRAY)

# Create the ORB detector.
# nfeatures caps the number of keypoints returned per frame.
orb = cv2.ORB_create(nfeatures=500)

# detectAndCompute returns:
#   kp  — list of cv2.KeyPoint objects (location, scale, orientation)
#   des — numpy array of binary descriptors, one row per keypoint
kp_early, des_early = orb.detectAndCompute(gray_early, None)
kp_late, des_late = orb.detectAndCompute(gray_late, None)

print(f"  ORB keypoints in early frame  (frame {EARLY_FRAME_IDX}): {len(kp_early)}")
print(f"  ORB keypoints in late frame   (frame {LATE_FRAME_IDX}): {len(kp_late)}")

if des_early is None or des_late is None or len(des_early) == 0 or len(des_late) == 0:
    print("  WARNING: Not enough ORB descriptors found — skipping ORB match image.")
else:
    # Brute Force Matcher with Hamming distance (correct for ORB binary descriptors).
    # crossCheck=True enforces mutual best-match: a pair is accepted only if
    # descriptor A's best match is B AND B's best match is A. This reduces false matches.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_early, des_late)

    # Sort matches by distance — lower Hamming distance means more similar descriptors
    matches = sorted(matches, key=lambda m: m.distance)

    # Keep only the top 50 matches for a clean, readable visualization
    top_matches = matches[:50]
    print(f"  Total matches found          : {len(matches)}")
    print(f"  Displaying top              : {len(top_matches)}")

    # Draw the matched keypoint pairs side-by-side with connecting lines
    # flags=2 draws only the matched keypoints, not all detected ones
    orb_match_img = cv2.drawMatches(
        frame_early, kp_early,
        frame_late, kp_late,
        top_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(ORB_MATCHES_IMG, orb_match_img)
    print(f"  Saved: {ORB_MATCHES_IMG}")

# Reset video to the beginning for the optical flow tracking pass below
video.set(cv2.CAP_PROP_POS_FRAMES, 1)  # Frame 0 was already read; start at frame 1

# ─────────────────────────────────────────────
# STEP 5: SET UP VIDEO WRITER AND TRACKING STATE
# ─────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

# prev_gray and prev_points are updated each frame to "previous" state
prev_gray = first_gray.copy()
prev_points = keypoints_roi.copy()

# trajectory_history stores the path of each keypoint as a list of (x, y) tuples
# Key: point index (int), Value: list of (x, y) positions
trajectory_history: dict[int, list[tuple[int, int]]] = {
    i: [(int(kp[0][0]), int(kp[0][1]))] for i, kp in enumerate(keypoints_roi)
}

# Assign a unique color to each keypoint trajectory for the overlay
np.random.seed(42)
colors = np.random.randint(0, 255, (len(keypoints_roi), 3)).tolist()

# Persistent canvas that accumulates trajectory lines across frames
trajectory_canvas = np.zeros_like(first_frame)

frame_count = 1  # We already read frame 0 as first_frame
midpoint_frame = total_frames // 2
mid_saved = False

# Write the annotated first frame to the output video
out.write(first_frame_vis)

# ─────────────────────────────────────────────
# STEP 6: TRACK KEYPOINTS USING LUCAS-KANADE OPTICAL FLOW
#
# Unlike ORB matching (which compares two arbitrary frames in isolation),
# Lucas-Kanade Optical Flow incrementally tracks the same corner points
# from one frame to the very next frame throughout the entire video.
# It assumes the pixel neighborhood around each point does not change much
# between adjacent frames, which is valid at 30 fps. The result is smooth
# motion trajectories that show the cereal box's path across time.
# ─────────────────────────────────────────────
print("\n--- Lucas-Kanade Optical Flow Tracking ---")
print("Tracking keypoints across frames...")

frame_vis = first_frame_vis.copy()  # Fallback in case the loop never runs

while True:
    ret, frame = video.read()
    if not ret:
        break  # End of video

    frame_count += 1
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow: find where prev_points moved in curr_gray.
    # next_points: estimated new positions; status: 1 if tracked, 0 if lost.
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **LK_PARAMS
    )

    # Keep only points that were successfully tracked (status == 1)
    if next_points is not None:
        good_prev = prev_points[status == 1]
        good_next = next_points[status == 1]
    else:
        good_prev = np.array([])
        good_next = np.array([])

    # Draw motion vectors (lines from old to new position) and keypoint dots
    frame_vis = frame.copy()

    for i, (new, old) in enumerate(zip(good_next, good_prev)):
        nx, ny = int(new[0]), int(new[1])
        ox, oy = int(old[0]), int(old[1])
        color = colors[i % len(colors)]

        # Draw the trajectory line on the persistent canvas so trails accumulate
        cv2.line(trajectory_canvas, (ox, oy), (nx, ny), color, 2)

        # Draw a filled circle at the current keypoint position
        cv2.circle(frame_vis, (nx, ny), 4, color, -1)

        # Update trajectory history for the Matplotlib plot
        if i in trajectory_history:
            trajectory_history[i].append((nx, ny))

    # Blend the persistent trajectory canvas onto the current frame
    frame_vis = cv2.add(frame_vis, trajectory_canvas)

    # Draw the original ROI bounding box and a frame counter as HUD text
    cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame_vis,
        f"Frame: {frame_count}/{total_frames}  Tracking: {len(good_next)} pts",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    out.write(frame_vis)

    # Save midpoint screenshot
    if not mid_saved and frame_count >= midpoint_frame:
        cv2.imwrite(SCREENSHOT_MID, frame_vis)
        print(f"  Saved midpoint screenshot at frame {frame_count}: {SCREENSHOT_MID}")
        mid_saved = True

    # Update previous state for next iteration
    prev_gray = curr_gray.copy()
    if len(good_next) > 0:
        prev_points = good_next.reshape(-1, 1, 2)
    else:
        # All points lost — attempt to re-detect features in the original ROI region
        print(f"  All keypoints lost at frame {frame_count}. Re-detecting...")
        roi_gray_curr = curr_gray[y:y + h, x:x + w]
        new_kps = cv2.goodFeaturesToTrack(roi_gray_curr, mask=None, **FEATURE_PARAMS)
        if new_kps is not None:
            new_kps[:, 0, 0] += x
            new_kps[:, 0, 1] += y
            prev_points = new_kps
        else:
            print("  Could not re-detect keypoints. Stopping early.")
            break

# Save the final frame screenshot
cv2.imwrite(SCREENSHOT_FINAL, frame_vis)
print(f"  Saved final screenshot: {SCREENSHOT_FINAL}")

# Release all video resources
video.release()
out.release()

print(f"\nFrames processed: {frame_count}")
print(f"Output video saved: {OUTPUT_VIDEO}")

# ─────────────────────────────────────────────
# STEP 7: MATPLOTLIB TRAJECTORY PLOT
# ─────────────────────────────────────────────
print("\nGenerating trajectory plot...")

fig, ax = plt.subplots(figsize=(10, 6))

for idx, path in trajectory_history.items():
    if len(path) > 5:  # Only plot points tracked for more than a few frames
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        color = [c / 255 for c in colors[idx % len(colors)]]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.75)
        ax.scatter(xs[0], ys[0], color=color, s=30, zorder=5)          # Start dot
        ax.scatter(xs[-1], ys[-1], color=color, s=60, marker="x", zorder=5)  # End marker

ax.set_title("Keypoint Motion Trajectories — Cereal Box Tracking", fontsize=14)
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")

# Invert Y axis so the plot matches image coordinate space (origin at top-left)
ax.invert_yaxis()
ax.set_xlim(0, frame_width)
ax.set_ylim(frame_height, 0)

# Draw the ROI boundary on the plot for spatial reference
roi_rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none", label="Initial ROI")
ax.add_patch(roi_rect)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(TRAJECTORY_PLOT, dpi=150)
plt.close()
print(f"Saved: {TRAJECTORY_PLOT}")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print("\n=== Processing Complete — Output Files ===")
print(f"  {SCREENSHOT_FIRST}  — first frame with detected keypoints")
print(f"  {SCREENSHOT_MID}    — mid-video tracking state")
print(f"  {SCREENSHOT_FINAL}       — final frame tracking state")
print(f"  {ORB_MATCHES_IMG}       — ORB Brute Force feature matches")
print(f"  {TRAJECTORY_PLOT}    — Matplotlib motion trajectory plot")
print(f"  {OUTPUT_VIDEO}     — full processed video with overlays")
