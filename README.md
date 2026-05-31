# Week 3: Object Tracking with Feature Matching and Optical Flow

Graduate Computer Vision Assignment — Artificial Intelligence in Computer Vision

## Demo Video

[![Object Tracking Demo](https://img.youtube.com/vi/gj0mQvpTVE4/0.jpg)](https://youtube.com/shorts/gj0mQvpTVE4)

> Click the thumbnail above to watch the tracked output video on YouTube.

## Overview

This project implements a prototype object-tracking system that processes a video of a cereal box moving through the frame. It demonstrates two complementary computer vision techniques:

- **ORB + Brute Force Matching** — compares two specific frames to show the object is recognizable across time using binary feature descriptors
- **Lucas-Kanade Optical Flow** — continuously tracks keypoints frame-by-frame across the full video to produce smooth motion trajectories

## Requirements

```
opencv-python
numpy
matplotlib
```

Install with:

```bash
pip install opencv-python numpy matplotlib
```

## Input Video

- **File:** `IMG_3260.mov`
- **Content:** Cereal box moving through the frame
- **Resolution:** 1080 × 1920 (portrait)
- **Duration:** ~12.4 seconds, 371 frames at 30 fps

## How to Run

1. Place `IMG_3260.mov` in the same folder as the script (or update `VIDEO_PATH` at the top of the script).
2. Run the script:

```bash
python week3_object_tracking.py
```

3. A window will open showing the first frame. **Draw a rectangle around the cereal box** and press **Enter** to confirm the Region of Interest (ROI).
4. The script will process all frames automatically and save the output files.

## Output Files

| File | Description |
|------|-------------|
| `first_frame_keypoints.png` | First frame with detected Shi-Tomasi keypoints shown as red dots inside the ROI |
| `orb_matches.png` | Top 50 ORB feature matches between frame 10 and frame 278 |
| `tracking_midpoint.png` | Frame 185 showing accumulated trajectory lines and current keypoint positions |
| `tracking_final.png` | Final frame with full trajectory overlay |
| `trajectory_plot.png` | Matplotlib plot of all keypoint motion paths in pixel-coordinate space |
| `tracked_output.mp4` | Full processed video with keypoint dots and trajectory lines overlaid — [view on YouTube](https://youtube.com/shorts/gj0mQvpTVE4) |

## Key Results

- **150** Shi-Tomasi keypoints detected inside the ROI on frame 1
- **500** ORB keypoints detected in each comparison frame
- **173** mutual Brute Force matches found (top 50 displayed)
- **371** frames processed to completion
- Auto-recovery triggered at frame 354 when all keypoints were lost — the script re-detected features inside the original ROI and resumed tracking

## Techniques Used

### Keypoint Detection — `cv2.goodFeaturesToTrack`
Shi-Tomasi corner detection identifies strong, trackable features inside the selected bounding box. Detection is restricted to the ROI to avoid latching onto background clutter.

### Feature Matching — `cv2.ORB_create` + `cv2.BFMatcher`
ORB detects keypoints and computes binary descriptors in two frames separated by time. The Brute Force Matcher with Hamming distance and `crossCheck=True` finds mutual best-match pairs, which are then sorted by match quality.

### Continuous Tracking — `cv2.calcOpticalFlowPyrLK`
Lucas-Kanade Optical Flow tracks the Shi-Tomasi keypoints incrementally across every consecutive frame pair. A 3-level image pyramid (`maxLevel=3`) and a 21×21 pixel search window allow the tracker to handle larger inter-frame displacements.

## Project Structure

```
Week3/
├── week3_object_tracking.py       # Main tracking script
├── IMG_3260.mov                   # Input video
├── first_frame_keypoints.png      # Output screenshot
├── orb_matches.png                # Output screenshot
├── tracking_midpoint.png          # Output screenshot
├── tracking_final.png             # Output screenshot
├── trajectory_plot.png            # Output plot
├── tracked_output.mp4             # Output video
└── README.md                      # This file
```

## References

- GeeksforGeeks. (2023a, January 3). *Converting color video to grayscale using OpenCV in Python.* https://www.geeksforgeeks.org/converting-color-video-to-grayscale-using-opencv-in-python/
- GeeksforGeeks. (2024p, September 5). *Python OpenCV: Capture video from camera.* https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
- Siromer. (2024a, October 8). *Detecting and tracking objects with ORB algorithm using OpenCV.* Medium. https://medium.com/thedeephub/detecting-and-tracking-objects-with-orb-using-opencv-d228f4c9054e
- Stats Wire. (2023, December 4). *OpenCV 26: Brute force matching with ORB descriptors* [Video]. YouTube. https://www.youtube.com/watch?v=lr1Sr0HJOoM
