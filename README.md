## Advanced Lane Finding Project

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_output2.png "Undistorted"
[image3]: ./examples/binary_combo_example.png "Binary Example"
[image4]: ./examples/warped_straight_lines.png "Warp Example"
[image5]: ./examples/color_fit_lines.png "Fit Visual"
[image6]: ./examples/plot_down.png "Output"
[image7]: ./examples/video_output.gif "Output"
[video1]: ./project_video.mp4 "Video"

## Project Dependencies

This project requires **Python 3.5** with the following libraries/dependencies installed:

- [Numpy](http://www.numpy.org/)
- [Matplotlib](http://matplotlib.org/)
- [OpenCV](http://opencv.org/)
- [MoviePy](http://zulko.github.io/moviepy/)


## Support Files

- `Advanced-Lane-Finding.ipynb` - The main notebook of the project.
- `helper.py` - The script contained required helper functions.
- `lineinfo.py` - The script define Python class to track line information.

---



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cell 2 of the IPython notebook located in "./Advanced-Lane-Finding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I applied a distortion correction using the camera calibration matrix and distortion coefficients. This logic is wrapped in helper function undistort_image() locate in "helper.py" file.
I applied this function test images and received one of the following result:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are warpped in function def `apply_schanel_and_gradient()` in `helper.py`).  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 22 through 40 in the file `helper.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warp_image()` function takes an input image (`img`).  I chose to hardcode the source and destination points in the following manner:

```python
src_points = np.float32([[700,450],[1150,img_height],[190,img_height],[590,450]])
dst_points = np.float32([[990,0],[940,img_height ], [300,img_height],[300,0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 700, 450      | 990, 0        | 
| 1150, 720     | 940, 720      |
| 190, 720      | 300, 720      |
| 590, 450      | 300, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I implemented the sliding window search technique to scan line pixels and fit my lane lines with a 2nd order polynomial. 

Here is the result:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Curvature calclation is wrapped in helper function `calc_curvature_and_center()` between lines 255 through 292 in my code in `helper.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 295 through 325 in my code in `helper.py` in the function `paint_lanes_interest_region()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

![alt text][image7]

Here's a [link to my video result](./videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The current lane detection algorithm  work well for the project video "project_video.mp4", in which the lanes are clealy visible and mark, but the algorithm fail when it run against videos with heavy shade/shadow on the roads and sharp curves.

The algorithm can be improved by expiermenting with other color spaces such as LAB and LUV to reduce shadows and return high lanes contrast. Base on my researches, I learn that B channel of LAB might work well in picking up yellow lanes, and L channel from LUV might works well to pick up white lanes, and both without the shadow noise that comes with S channel of HSV.
