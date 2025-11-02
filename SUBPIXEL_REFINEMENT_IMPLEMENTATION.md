# Sub-Pixel Y-Axis Refinement for YOLO Stabilization

## Overview

This feature adds sub-pixel accurate Y-axis refinement to YOLO-based sprocket hole detection to reduce vertical jitter in frame stabilization. The jitter is caused by random variations in film scanner mechanics stopping position, making temporal averaging ineffective.

## Implementation Details

### 1. Core Algorithm

The refinement uses **edge detection** to precisely locate the top edge of the detected sprocket hole:

1. **ROI Extraction**: Expands the YOLO bounding box by 15px vertically to capture edges better
2. **Preprocessing**: Applies Gaussian blur to reduce noise
3. **Edge Detection**: Uses Canny edge detection to find strong edges
4. **Profile Analysis**: Sums edge intensity along each horizontal line in the top half of the sprocket hole
5. **Edge Localization**: Identifies the first strong edge from the top
6. **Sub-pixel Interpolation**: Uses parabolic interpolation on edge profile for sub-pixel accuracy
7. **Sanity Check**: Rejects refinements that deviate more than 10px from original detection

### 2. Key Function

**`refine_yolo_sprocket_y_position(img_ref, sprocket_box)`** - [AfterScan.py:4734-4833](AfterScan.py#L4734-L4833)

- Input: BGR image and YOLO bounding box [x1, y1, x2, y2]
- Output: Refined Y position (float) with sub-pixel accuracy
- Falls back to original Y position if refinement fails

### 3. Integration Points

#### Global Configuration
- Line 383: `yolo_subpixel_refinement = False` - Feature toggle

#### Detection Integration
- Lines 4931-4939: Applied in `detect_yolo_sprocket()` after YOLO detection
- Only refines Y-axis position (detected_y), X-axis unchanged

#### UI Controls
- Lines 7303-7325: "Sub-pixel Y" checkbox under YOLO settings
- Enabled/disabled with other YOLO controls when stabilization method changes
- Lines 7130, 7138: Added to enable/disable logic

#### Configuration Persistence
- Line 849: Save to project config
- Lines 1156-1159: Load from project config

## Usage

1. **Enable YOLO Stabilization**: Select "YOLO" as stabilization method
2. **Enable Sub-pixel Refinement**: Check the "Sub-pixel Y" checkbox
3. **Process Frames**: The refinement applies automatically during stabilization

## Benefits

- **Reduces Vertical Jitter**: Sub-pixel accuracy eliminates frame-to-frame Y-axis variations
- **No Temporal Averaging**: Works independently for each frame, suitable for random mechanical variations
- **Minimal Performance Impact**: Edge detection only on small ROI around detected sprocket
- **Robust Fallback**: Returns original position if refinement fails or deviates too much

## Parameters (Tunable)

In `refine_yolo_sprocket_y_position()`:

- `margin_y = 15`: Vertical expansion for ROI (larger = more context, slower)
- `margin_x = 5`: Horizontal expansion for ROI
- `GaussianBlur kernel = (5, 5)`: Noise reduction (larger = smoother but less precise)
- `Canny thresholds = (30, 100)`: Edge detection sensitivity
- `search_end = margin_y + int(box_height * 0.5)`: Search in top 50% of box
- `threshold = np.max(edge_profile) * 0.5`: Edge strength threshold (50% of max)
- `sanity_check = 10px`: Maximum allowed deviation from original position

## Debugging

Enable debug logging to see refinement details:

```python
logging.debug(f"Sub-pixel refinement: Y adjusted by {refined_y - y1:.2f} pixels")
```

Check logs for:
- Refinement deltas (should be small, typically < 3 pixels)
- Fallback cases (ROI empty, no edges, large deviation)
- Edge detection success rates

## Testing Recommendations

1. Compare output with/without refinement enabled
2. Check frame alignment stability in video playback
3. Monitor logs for sanity check rejections (indicates parameter tuning needed)
4. Test on various film qualities (scratched, low contrast, damaged perforations)

## Future Enhancements

1. Make edge detection parameters configurable via UI
2. Add visualization of detected edges (debug mode)
3. Consider X-axis refinement if horizontal jitter is also problematic
4. Adaptive parameters based on YOLO confidence
