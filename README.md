#**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline description
Image processing steps:
- Converting image to grayscale ;
![](https://github.com/Chenlu-Wu/CarND-LaneLines/blob/master/test_images_output/solidWhiteCurve.jpg_grayscale.jpg)

- Applying Gaussian blur on the gray image to smooth the edges and reduce the noise

![](https://github.com/Chenlu-Wu/CarND-LaneLines/blob/master/test_images_output/solidWhiteCurve.jpg_Gaussin_blur.jpg)

- Applying Canny transform to find the all the edges on the Guassian_blured image
![](https://github.com/Chenlu-Wu/CarND-LaneLines/blob/master/test_images_output/solidWhiteCurve.jpg_canny.jpg)

- Creating a all black mask with the same size as the original image, find the region of interest (ROI) of the image (which is the area contains the lanes we care), making a masked image by adding mask and image with ROI
![](https://github.com/Chenlu-Wu/CarND-LaneLines/blob/master/test_images_output/solidWhiteCurve.jpg_masked.jpg)

- Applying Houph transformation on the masked image to find the all the lines longer than 10 pixels, and then draw the lines in red, name this processed masked image as lined_image;
![](https://github.com/Chenlu-Wu/CarND-LaneLines/blob/master/test_images_output/solidWhiteCurve.jpg_hough.jpg)

- Adding the orignal image to the lined_image with weights α=0.8 and β=1.0, then save the image to test_image_output file.
![](https://github.com/Chenlu-Wu/CarND-LaneLines/blob/master/test_images_output/solidWhiteCurve.jpg_segment.jpg)

Modification for the draw_lines() function:

- Calculated the average slope of the line, if the slope is larger than 0.5 or smaller than -0.5, then draw the line, because slope larger than 0.5 usually is the right lane, and the slope smaller than -0.5 is the left lane, the slope in between might cause the line linked cross both right lane and left lane.



![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

- One potential shortcoming would be the lane tracking might be inaccurate if there is sharp turns on the road

- Another shortcoming could be if the region of interest contains a lot of noise, then the lane tracking would be inaccurate.


### 3. Suggest possible improvements to your pipeline

- A possible improvement would be to apply none-linear model to instead of linear model to find the lane, so we could improve the accuracy on the road with sharp turns

- Another potential improvement could be to try to reduce the noise, probably we could more restrictions like color, for example, except for the hough transform, we also require the color of the lane should be in the range of yellow or white.