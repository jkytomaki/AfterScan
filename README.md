# AfterScan - R8/S8 film post-scan utility 

This utility is intended to handle the basic post-processing after film scanning is completed, mainly stabilizing and cropping.

Originally created for the T-Scann 8 project (Torulf Holmström, http://tscann8.torulf.com/index.html), but probably can be used for other frame-by-frame film scanners as well.

Actions performed by this tool on the captured frames are:
- Stabilization (taking the film sprocket hole as reference)
- Cropping
- Video generation

And that's about it. Other post-processing jobs (film cleanup, color correction) should be done with other tools to be found elsewhere.

This tool relies in the following open source projects to achieve its objectives:
* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)
* [FFmpeg](https://ffmpeg.org/)
* [Pillow](https://python-pillow.org/)

## How it works:
Sprocket holes in 8mm films are expected to be in a very precise location for each film type (S8/R8). What this tool does is to detect, thanks to OpenCV, the hole(s) in each frame, and then shift the frame as required so that the holes fall in the expected position. This process is individual for each frame, so the length of the scanned film should have no effect on the result. A few points to highlight:
- Make sure that the input files provided to the tool contain each a full frame, with the sprocket hole(s) 50% visible at least. Otherwise the results will not be good
- When dealing with overexposed frames, where the sprocket hole is not visible to the naked eye, the tool will not be able to stabilize the frame properly. However, as such frames do not have any useful content, and there should be very few of them, should not be a big deal

## Basic instructions:
1) Select the source and target folders. They have to be different folders; frames will be taken from the source folder and, after processing, saved in the target folder, to be used at the video generation step, or by a third party tool to perform further processing
2) Using the current frame slider , search for a frame that is fully visible, and click on the 'Image crop area' to define the part of the image you want in the output files
3) Select the 'Stabilize' and 'Crop' checkboxes to enable both processes
4) Select the film type (S8/R8). Might not be necessary since when loading the source frames, the tool should detect the film type, and propose a change if the setting is incorrect
5) Finally, if you want the tool to generate the video, you can select the relevant checkbox. The checkbox 'Skip frame regeneration' is there to allow generating again all the stabilized/cropped frames in case they are already there, and go directly to the video generation step. Options available when video is generated:
   - Filename: Name of the file where the video will be written. It will be stored in the target folder, together with the stabilized/cropped frames. If no name is supplied, the tool will automatically create one with a timestamp
   - FPS: Frames per second. For Super 8 this should be 18 and, I think, 16 in the case of R8
   - Quality/speed choice: Three options available, from high quality (slow) to Fast (low quality)
   
## Additional information
You can find a description of the UI eolements in the [wiki](https://github.com/jareff-g/AfterScan/wiki/AfterScan-UI-description).



