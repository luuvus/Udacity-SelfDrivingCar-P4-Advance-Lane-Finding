import numpy as np
import cv2
import matplotlib.pyplot as plt
import lineinfo

def calibrate_undistore(img, objpoints, imgpoints):
    
    img_width_height = (img.shape[1],img.shape[0])
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_width_height, None, None)
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist

def undistort_image(img, mtx, dist):
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist

def warp_image(img):
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    # define four source points cordinates
    src_points = np.float32([[700,450],[1150,img_height],[190,img_height],[590,450]])
    #src_points = np.float32([[730,470],[1150,img_height],[190,img_height],[600,470]])
    
    #define four desired/destination points/coordinates
    
    dst_points = np.float32([[990,0],[940,img_height ], [300,img_height],[300,0]])
    #dst_points = np.float32([[900,0],[940,img_height ], [300,img_height],[400,0]])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inverse_matrix = cv2.getPerspectiveTransform( dst_points,src_points)
    
    warped_image = cv2.warpPerspective(img, matrix, (img_width, img_height), flags = cv2.INTER_LINEAR)
    
    return warped_image, matrix, inverse_matrix

def warp_image2(img):
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    # define four source points cordinates
    src_points = np.float32([[730,470],[1150,img_height],[190,img_height],[600,470]])
    #src_points = np.float32([[700,450],[1150,img_height],[190,img_height],[600,450]])
    #src_points = np.float32([[700,500],[1150,img_height],[190,img_height],[600,500]])
    
    #define four desired/destination points/coordinates
    #dst_points = np.float32([[1000,0],[940,img_height ], [300,img_height],[500,0]])
    dst_points = np.float32([[900,0],[900,img_height ], [300,img_height],[450,0]])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inverse_matrix = cv2.getPerspectiveTransform( dst_points,src_points)
    
    warped_image = cv2.warpPerspective(img, matrix, (img_width, img_height), flags = cv2.INTER_LINEAR)
    
    return warped_image, matrix, inverse_matrix


def apply_schanel_and_gradient(img, s_thresh=(90,255), sx_thresh=(20, 100)):
    img = np.copy(img)

    # Convert to HLS color space and separate the S channel
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    img_L = img_hls[:,:,1]
    img_S = img_hls[:,:,2]
    
    # Sobel x
    sobel_x = cv2.Sobel(img_L, cv2.CV_64F, 1,0) # Take the derivative in x
    abs_sobel_x = np.absolute(sobel_x) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel_x/np.max(abs_sobel_x))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]) ] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(img_S)
    s_binary[(img_S >= s_thresh[0]) & (img_S <= s_thresh[1]) ] = 1
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.    
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
               
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        
    #return color_binary
    return combined_binary

def find_lanes(binary_warped):
    
    input_img_height = binary_warped.shape[0]
    input_img_widht = binary_warped.shape[1]
    
    # Take a histogram of the bottom half of the image 
    #histogram = np.sum(binary_warped[binary_warped.shape[0]//3 : , : ],axis=0)

    # adjust 200 pixels from left and right
    histogram_offset = 200
    histogram = np.sum(binary_warped[binary_warped.shape[0]//3:,histogram_offset:binary_warped.shape[1]-histogram_offset], axis=0)
    
    # Create an output image to draw on and  visualize the result
    # out_img = np.dstack((binary_warped,binary_warped,binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    #midpoint = histogram.shape[0]//2   
    midpoint = histogram.shape[0]//2 
    leftx_base = np.argmax(histogram[:midpoint]) + histogram_offset
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint + histogram_offset

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = binary_warped.shape[0]//nwindows
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 85
    margin = 110
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    rectangles = []
    
    leftx_prev = leftx_base
    rightx_prev = rightx_base

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = input_img_height - ((window + 1) * window_height)
        win_y_high = win_y_low + window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # save the windows on the visualization image
        rectangles.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]
             
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds])) 

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            rightx_prev = rightx_current
        else:
            rightx_current = rightx_prev
            
    # Concatenate the arrays of indices   
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds] 
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, rectangles


def find_lanes_with_known_fits(binary_warped, known_left_fit, known_right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    margin = 85
    left_lane_inds = ((nonzerox > (known_left_fit[0]*(nonzeroy**2) + known_left_fit[1]*nonzeroy + known_left_fit[2] - margin)) & (nonzerox < (known_left_fit[0]*(nonzeroy**2) + known_left_fit[1]*nonzeroy + known_left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (known_right_fit[0]*(nonzeroy**2) + known_right_fit[1]*nonzeroy + known_right_fit[2] - margin)) & (nonzerox < (known_right_fit[0]*(nonzeroy**2) + known_right_fit[1]*nonzeroy + known_right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    out_left_fit = np.polyfit(lefty, leftx, 2)
    out_right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = out_left_fit[0]*ploty**2 + out_left_fit[1]*ploty + out_left_fit[2]
    right_fitx = out_right_fit[0]*ploty**2 + out_right_fit[1]*ploty + out_right_fit[2]
    
    return out_left_fit, out_right_fit, left_lane_inds, right_lane_inds

def visualize_sliding_windows(bin_img,left_lane_inds, right_lane_inds, rectangles):
    ## create an output image to draw on and visualize the result
    out_img = np.uint8(np.dstack((bin_img, bin_img, bin_img))*255)

    # Draw the rectangles on image
    for rect in rectangles:
        cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 3) 
        cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 3) 

    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img 

def generate_polyfit_plot_data(bin_img, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return ploty, left_fitx, right_fitx


def calc_curvature_and_center(bin_img, left_lane_idx, right_lane_idx):
    
    ## define range
    ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0])
    
    ## max y
    y_eval = np.max(ploty)

    # find nonzero pixels
    nonzero = bin_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_idx]
    lefty = nonzeroy[left_lane_idx] 
    rightx = nonzerox[right_lane_idx]
    righty = nonzeroy[right_lane_idx]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/900 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_curve = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_curve = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_radius = ((1 + (2*left_fit_curve[0]*y_eval*ym_per_pix + left_fit_curve[1])**2)**1.5) / np.absolute(2*left_fit_curve[0])
    right_radius = ((1 + (2*right_fit_curve[0]*y_eval*ym_per_pix + right_fit_curve[1])**2)**1.5) / np.absolute(2*right_fit_curve[0])
    
    ## distance from center
    center_idx = bin_img.shape[1]//2
    identified_lanes_center_idx = (min(leftx) + max(rightx))//2
    
    dist_from_cent = np.abs(center_idx - identified_lanes_center_idx)*xm_per_pix
    
    return np.array([left_radius, right_radius, dist_from_cent])


def paint_lanes_interest_region(img, bin_img, left_fit_poly, right_fit_poly, inverse_matrix):
    
    ## create copy of image
    img_copy = np.copy(img)
    
    ## define range
    ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0])
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ## fit lines
    left_fitx = left_fit_poly[0]*ploty**2 + left_fit_poly[1]*ploty + left_fit_poly[2]
    right_fitx = right_fit_poly[0]*ploty**2 + right_fit_poly[1]*ploty + right_fit_poly[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, inverse_matrix, (img_copy.shape[1], img_copy.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(img_copy, 1, newwarp, 0.3, 0)
    
    return result

def write_curvature_info_on_image(area_img, curvature_output):
    
    ## copy area image
    area_img_copy = np.copy(area_img)
    
    ## format text for overlay
    left_text = "Left Curve Radius: {0:.2f}m".format(curvature_output[0])
    right_text = "Right Curve Radius: {0:.2f}m".format(curvature_output[1])
    dist_text = "Distance from Center: {0:.2f}m".format(curvature_output[2])
    
    ## area_img writing
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(area_img_copy, left_text, (60,90), font, 1.25, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(area_img_copy, right_text, (60,140), font, 1.25, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(area_img_copy, dist_text, (60,190), font, 1.25, (255,255,255), 2, cv2.LINE_AA)

    return area_img_copy

def check_lanes_parallel_v1(left_x, right_x):
    thresh = .05
    lane_diff = np.subtract(right_x, left_x)
    
    #http://stackoverflow.com/questions/10542240/easy-way-to-test-if-each-element-in-an-numpy-array-lies-between-two-values
    lane_width = 780 
    
    lane_parallel = (lane_diff > lane_width*(1-thresh)).all() and (lane_diff < lane_width*(1+thresh)).all() 
    
    return lane_parallel

def check_lanes_parallel(left_x, right_x, lane_width_min = 500, lane_width_max = 700):
    
    lane_diff = np.subtract(right_x, left_x)

    lane_parallel = (lane_diff > lane_width_min).all() and (lane_diff < lane_width_max).all() 
    return lane_parallel

def is_lane_detection_valid(left_curv, right_curv, left2c, right2c, left_fit, right_fit):

    # check curvature
    if left_curv >= 200 and right_curv >= 200:
        flag_curv = True
    else:
        flag_curv = False
    # check horizontal distance
    dist = left2c + right2c
    if dist >= 3. and dist <= 6.:
        flag_dist = True
    else:
        flag_dist = False
    # check parallel
    left_slope = left_fit[0]
    right_slope = right_fit[0]
    if np.absolute(right_slope - left_slope) <= 9e-4:
        flag_paral = True
    else:
        flag_paral = False
    return (flag_curv and flag_dist and flag_paral)