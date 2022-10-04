# AfterScan - R8/S8 film post-scan utility 

This utility is intended to handle the basic post-processing after film scanning is completed.
Originally created for the T-Scann 8 project (Torulf Holmström, http://tscann8.torulf.com/index.html), but probably can be used for other film scanners as well.
Licensed under a MIT LICENSE.

Actions performed by this tool on the captured frames are:
- Stabilization (taking the film sprocket hole as reference)
- Cropping
- Video generation

This tool relies in the following open source projects to achieve its objectives:
* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)
* [FFmpeg](https://ffmpeg.org/)
* [Pillow](https://python-pillow.org/)


Instructions:
1) Select the source and target folders. They have to be different folders; frames will be taken from the source folder and, after processing, saved in the target folder to be used at the video generation step, or by a third party tool to perform further processing.
2) Using the frame selection buttons, look for a frame that is well centered, and where the sprocket hole is clearly visible, preferably with a good contrast against its surroundings
3) Click on the 'Perforation search area' button, to define the area where the tool will search for the sprocket hole.
   - A new window will be displayed with the current frame; You can then draw a rectangle with the area you think will fetch the perforation in each and every frame. If you are not happy with the rectangle just drawn, just try again (rectangle not editable). Once you are happy with the definition, press enter to confirm and exit. If you just want to cancel the selection, pressing escape will do it and get out of the window.  
   - This step is __very important__ because it will reduce the chances of having false positives while searching for the hole, securing a good stabilization of all the frames. The area has to be big enough to cover the hole plus some of the surroundings, up and down and to the right. You can browse through the captured frames to have an idea about how much the hole might move, in order to identify the best size. Making it too big (covering the full frame would be the extreme case) will cause false positives, and the stabilization will not work.
4) Once the perforation area is defined, the checkbox to enable stabilization will be enabled: You can select it now.  
5) Nest step would be to confirm the rectangle just defined is good. Without selecting any other checkbox, click on the 'Start' button and the stabilization process will start. Since the perforations are still visible, this run is a very good chance to appreciate if the perforation area selection was good; If you see that the perforation hole is steady and doesn't move, the it means it was a good selection, you can keep it. Otherwise you can try again to select a different area.
   - Warning: The stabilization process targets __only__ to stabilize frames against each other so that the perforation appears at the same location for all of them, avoiding the slight (S8/R8 frames are small!) discrepancies during the capture. Stabilization within the film itself is of course out of scope; if the person doing the original filming had a shaky hand on the day of the filming there's nothing this tool can do.
6) Form here things are easier. Click on the 'Image crop area' to define the part of the image you want to appear in the images generated in the target folder (and the video if you later choose to generate it as well). Rectangle is defined in the same was as for the stabilization area (only that this time the area you select will be different of course).
7) Finally, if you want the tool to generate the video, you can select the relevant checkbox. The checkbox 'Skip frame regeneration' is there to allow generating again all the stabilized/cropped frames in case they are already there, and go directly to the video generation step. Options available when video is generated:
   - Filename: Name of the file where the video will be written. It will be stored in the target folder, together with the stabilized/cropped frames. If no name is supplied, the tool will automatically create one with a timestamp.
   - FPS: Frames per second. For Super 8 and Regular 8 this should be 18 most of the time (that's why it is the default value)
   - Quality/speed choice: Three options available, from high quality (slow) to Fast (low quality). The default is medium. Your choice.



