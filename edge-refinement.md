# Edge-Based Refinement for YOLO Stabilization

## Overview

Edge-based refinement is a sub-pixel precision technique that fine-tunes YOLO's sprocket hole detection to reduce vertical jitter in stabilized 8mm film frames. It works as a second-stage refinement after YOLO's initial coarse detection.

## The Problem

YOLO object detection correctly identifies sprocket hole regions, but the **bounding box edges have random 20-50 pixel offsets** from the actual physical edges of the sprocket holes. This is because:

- YOLO bounding boxes are loose approximations of object regions
- The box edges don't tightly fit the physical hole boundaries
- Frame-to-frame, these box edges vary randomly relative to the true physical edge
- This variation translates to 5-10 pixels of vertical jitter in stabilized output

## The Solution: Multi-Method Edge Detection

Instead of using YOLO's bounding box top edge directly, we detect the **true physical top edge** of the sprocket hole using computer vision edge detection within the YOLO bounding box.

### Algorithm Steps

#### Step 1: Extract Region of Interest (ROI)

```
YOLO Bounding Box (loose fit):
    ┌─────────────────┐
    │  margin (20px)  │
    │ ┌─────────────┐ │
    │ │  Sprocket   │ │ <- True physical edge somewhere in here
    │ │    Hole     │ │
    │ └─────────────┘ │
    └─────────────────┘
```

- Expand YOLO box by 20px vertically, 10px horizontally for context
- Extract this ROI from the frame
- Search the top 75% of the bounding box for the edge

#### Step 2: Apply Multiple Edge Detection Methods

**Method 1: Sobel Y Gradient (60% weight)**
- Computes vertical intensity gradient using Sobel operator
- Detects **dark-to-bright transitions** (sprocket hole → film)
- Most reliable for finding horizontal edges
- Formula: `sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1)`

**Method 2: Canny Edge Detection (25% weight)**
- Uses adaptive thresholds based on local image intensity
- Lower threshold: `median_intensity × 0.33`
- Upper threshold: `median_intensity × 0.66`
- Adapts to varying contrast and exposure

**Method 3: Morphological Gradient (15% weight)**
- Alternative edge detection using morphological operations
- Formula: `dilate(image) - erode(image)`
- Robust to noise and film grain

#### Step 3: Create Horizontal Line Profile

For each edge detection method:

```python
# Sum edge intensity along each horizontal line
edge_profile = np.sum(edges[search_start:search_end, :], axis=1)
```

This creates a **1D profile** where:
- Each Y position has a score = "total edge intensity at this horizontal line"
- Strong horizontal edges (like the top of a sprocket hole) appear as **peaks**
- Noise and grain average out across the width

**Visual representation:**
```
Image Y positions:          Edge Profile:
─────────────── 100         ▁▁▁▁ (weak)
═══════════════ 105         ████ (strong peak!) ← Top edge found here
╔═════════════╗ 110         ▂▂▂▂ (medium)
║   HOLE      ║ 115         ▁▁▁▁ (weak)
║   (dark)    ║ 120         ▁▁▁▁ (weak)
```

#### Step 4: Combine Methods with Weighted Voting

```python
# Normalize each profile to 0-1 range
sobel_profile = sobel_profile / max(sobel_profile)
canny_profile = canny_profile / max(canny_profile)
morph_profile = morph_profile / max(morph_profile)

# Weighted combination
combined = 0.60 × sobel_profile + 0.25 × canny_profile + 0.15 × morph_profile
```

This creates a consensus view where all three methods "vote" on edge locations.

#### Step 5: Find Edge Candidates (Peaks)

```python
threshold = max(combined_profile) × 0.4

for each Y position:
    if profile[y] > threshold AND
       profile[y] >= profile[y-1] AND  # Local maximum
       profile[y] >= profile[y+1]:
        add Y to peak_indices
```

Identifies all strong horizontal edges in the search region.

#### Step 6: Select Best Edge

Uses a scoring function that balances:
1. **Edge strength**: How strong is the detected edge?
2. **Proximity to expected position**: How close to YOLO's box edge?

```python
score = edge_strength / (1.0 + distance_penalty)

where:
    distance_penalty = distance_from_expected / (box_height × 0.3)
```

This prefers strong edges that are reasonably close to where YOLO detected the hole.

#### Step 7: Sub-pixel Refinement

Uses quadratic interpolation on the peak:

```python
# Get three points around the peak
y_prev = profile[peak - 1]
y_curr = profile[peak]
y_next = profile[peak + 1]

# Parabolic interpolation
sub_pixel_offset = (y_prev - y_next) / (2 × (2×y_curr - y_prev - y_next))
refined_y = peak + sub_pixel_offset  # e.g., 105.347 instead of 105
```

This provides fractional pixel precision (e.g., 1814.356 instead of 1814.000).

#### Step 8: Sanity Check

```python
if abs(refined_y - yolo_y) > 60:
    reject refinement, use original YOLO position
```

Rejects refinements that are too far from YOLO's estimate (likely errors).

## Why It Works

### Robustness Through Averaging
- Summing horizontally across the width of the sprocket hole averages out:
  - Film grain (random noise)
  - Small scratches or dirt particles
  - Minor variations in edge sharpness
  - JPEG compression artifacts

### Multiple Method Consensus
- Different edge detectors have different strengths and weaknesses
- Weighted voting reduces false positives
- Sobel is most reliable, others provide confirmation

### Adaptive to Image Conditions
- Canny thresholds adapt to local brightness and contrast
- Works with varying film exposure, grain, and quality
- Morphological gradient robust to uneven illumination

## Configuration Options

### In AfterScan.py

**Enable/Disable:**
```python
yolo_subpixel_refinement = True  # Default: enabled
```

**UI Control:**
- Checkbox: "Sub-pixel Y" in stabilization settings
- Only active when YOLO stabilization is selected

**Search Region:**
```python
search_end = margin_y + int(box_height × 0.75)  # Top 75% of box
```

**Deviation Tolerance:**
```python
max_deviation = 60  # pixels - YOLO boxes can be loose
```

## Optional Temporal Smoothing

After edge refinement, optional exponential smoothing can be applied:

```python
yolo_filter_method = "none"         # Pure edge detection (default)
yolo_filter_method = "exponential"  # Add temporal smoothing
```

**Exponential smoothing:**
- Blends current detection with previous frame
- Adaptive alpha: responds faster to large motions
- Formula: `smoothed = α × current + (1-α) × previous`

## Performance Characteristics

**Computational Cost:**
- Adds ~1-2ms per frame on modern hardware
- Three edge detection passes + profiling
- Negligible compared to YOLO inference (~50-100ms)

**Accuracy:**
- Sub-pixel precision: ~0.1-0.5 pixel accuracy
- Typical adjustment: 20-50 pixels from YOLO box edge
- Expected jitter reduction: 60-80% vs. raw YOLO boxes

## Expected Results

### Before Edge Refinement
```
Frame N:   detected_y = 1814.000  (YOLO box edge)
Frame N+1: detected_y = 1822.000  (jumped 8 pixels)
Frame N+2: detected_y = 1831.000  (jumped 9 pixels)
Frame N+3: detected_y = 1800.000  (jumped -31 pixels!)
```
**Result:** 5-10 pixel vertical jitter in stabilized output

### After Edge Refinement
```
Frame N:   detected_y = 1795.347  (true edge + sub-pixel)
Frame N+1: detected_y = 1795.821  (jumped 0.474 pixels)
Frame N+2: detected_y = 1796.103  (jumped 0.282 pixels)
Frame N+3: detected_y = 1795.556  (jumped -0.547 pixels)
```
**Result:** Sub-pixel stability, smooth stabilized output

## Debugging and Diagnostics

### Log Output

When enabled, detailed logs show:
```
[INFO] Sub-pixel refinement: Y 1814 -> 1795.347 (adjusted -18.653px, score: 0.847)
```

**Warnings to watch for:**
```
[WARNING] Sub-pixel refinement: No edges detected (profile max: 0.023)
[WARNING] Sub-pixel refinement: Excessive deviation (85.23px), rejecting
[WARNING] Sub-pixel refinement: Invalid search range (30 to 28, roi height: 120)
```

These indicate edge detection failures (falls back to YOLO box edge).

### Success Indicators
- ✅ Detected Y values have decimal places (e.g., 1814.356)
- ✅ Frame-to-frame Y variation < 1 pixel
- ✅ Adjustments typically 20-50 pixels from YOLO box edge
- ✅ Edge detection scores > 0.5

### Failure Indicators
- ❌ Detected Y values are integers (e.g., 1814.000)
- ❌ Many "No edges detected" warnings
- ❌ Excessive deviation warnings
- ❌ Frame-to-frame jumps still 5+ pixels

## Technical References

**Edge Detection Methods:**
- Sobel operator: Computes image gradient using convolution
- Canny edge detection: Multi-stage algorithm with hysteresis
- Morphological gradient: Dilation minus erosion

**Sub-pixel Techniques:**
- Quadratic interpolation: Fits parabola to 3-point peak
- Common in computer vision for sub-pixel feature localization
- Provides 0.1-0.5 pixel accuracy when peak is well-defined

**Related Techniques:**
- Phase correlation for sub-pixel image registration
- Lucas-Kanade optical flow for motion estimation
- Template matching with sub-pixel refinement

## Limitations

1. **Requires visible edges:** Doesn't work if sprocket holes are:
   - Extremely out of focus
   - Completely obscured by dirt/damage
   - Outside the frame

2. **Assumes dark holes on bright film:** Looks for dark-to-bright transitions
   - Inverted negatives might need adjustment
   - Very low contrast film may fail

3. **Single reference point:** Only refines Y position of top edge
   - Doesn't correct for rotation or scale
   - Assumes holes are reasonably rectangular

4. **Temporal independence:** Each frame analyzed independently
   - No motion prediction (unlike Kalman filtering)
   - Can't handle complete occlusions

## Future Enhancements

Potential improvements:
- Detect multiple edges (top and bottom) for averaging
- Rotation correction based on hole shape
- Machine learning for edge scoring
- Multi-frame consensus for difficult frames
- Adaptive search region sizing

## Conclusion

Edge-based refinement transforms YOLO's coarse bounding box detection into precise, sub-pixel accurate sprocket hole localization. By combining multiple edge detection methods and horizontal line profiling, it achieves consistent frame-to-frame stability, reducing jitter from 5-10 pixels down to sub-pixel levels.

The key insight is that **YOLO finds the right region, but edge detection finds the exact position** within that region.
