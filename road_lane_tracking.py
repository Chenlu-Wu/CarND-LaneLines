#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math
get_ipython().run_line_magic('matplotlib', 'inline')


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    for line in lines:
        sumx1 = 0
        sumy1 = 0
        sumx2 = 0
        sumy2 = 0
        count = 0
        for x1,y1,x2,y2 in line:
            count=count+1
            sumx1+=x1
            sumy1+=y1
            sumx2+=x2
            sumy2+=y2
        avx1 = int(sumx1/count)
        avy1 = int(sumy1/count)
        avx2 = int(sumx2/count)
        avy2 = int(sumy2/count)
        slope = (avy1-avy2)/(avx1-avx2)
        intercept = int(avy1 - avx1*slope)
        if ((avy1-avy2)/(avx1-avx2)>0.5) or ((avy1-avy2)/(avx1-avx2)<(-0.5)):
            cv2.line(img, (avx1, avy1), (avx2, avy2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)



def find_lane(name, v1, v2 ):
    
    readname= 'test_images/' + name 
    img1=mpimg.imread(readname)
    plt.figure()
    plt.imshow(img1)
    
    grayimg1 = grayscale(img1)
    grayname = name +'_grayscale.jpg'
    cv2.imwrite(os.path.join('test_images_output/' , grayname), grayimg1)
    
    kernal_size = 3
    blurgrayimg1 = gaussian_blur(grayimg1, kernal_size)
    blurname = name +'_Gaussin_blur.jpg'
    cv2.imwrite(os.path.join('test_images_output/' , blurname), blurgrayimg1)
    
    
    low_threshold = 84
    high_threshold = 168
    edges1 = canny(blurgrayimg1, low_threshold, high_threshold)
    cannyname = name +'_canny.jpg'
    cv2.imwrite(os.path.join('test_images_output/' , cannyname), edges1)    
    
    imshape = img1.shape
    vertices = np.array([[(0,imshape[0]),v1, v2, (imshape[1],imshape[0])]], dtype=np.int32)
    #vertices = np.array([[0,imgshape1[0]], [450,315], [525,315], [imgshape1[1],imgshape1[0]]], dtype=np.int32)
    masked_img1 = region_of_interest(edges1, vertices)
    maskedname = name +'_masked.jpg'
    cv2.imwrite(os.path.join('test_images_output/' , maskedname), masked_img1)  
    
    
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_len = 10
    max_line_gap = 5
    
    max_line_gap_con = 120
    
    line_img1 = hough_lines(masked_img1, rho, theta, threshold, min_line_len, max_line_gap)
    houghname = name +'_hough.jpg'
    cv2.imwrite(os.path.join('test_images_output/' , houghname), cv2.cvtColor(line_img1, cv2.COLOR_RGB2BGR))
    #cv2.imwrite(os.path.join('test_images_output/' , houghname), line_img1)
    
    line_con_img1 = hough_lines(masked_img1, rho, theta, threshold, min_line_len, max_line_gap_con)
    
    final_seg_img1 = weighted_img(line_img1, img1, α=0.8, β=1., γ=0.)
    final_con_img1 = weighted_img(line_con_img1, img1, α=0.8, β=1., γ=0.)
    plt.figure()
    plt.imshow(final_seg_img1)
    plt.figure()
    plt.imshow(final_con_img1)
    #cv2.imwrite('sample_out_2.png', cv2.cvtColor(final_con_img1, cv2.COLOR_RGB2BGR)) 
    segname = name + '_segment.jpg'
    continame = name + '_thirdpass.jpg'
    cv2.imwrite(os.path.join('test_images_output/' , segname), cv2.cvtColor(final_seg_img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join('test_images_output/' , continame), cv2.cvtColor(final_con_img1, cv2.COLOR_RGB2BGR))
    
os.listdir("test_images/")    
name1 = 'solidWhiteCurve.jpg'
v1 = (457, 320)
v2 = (490, 315)
find_lane(name1, v1, v2)

name2 = 'solidWhiteRight.jpg'
v1 = (457, 320)
v2 = (490, 315)
find_lane(name2, v1, v2)

name3 = 'solidYellowCurve.jpg'
v1 = (444, 326)
v2 = (529, 326)
find_lane(name3, v1, v2)

name4 = 'solidYellowCurve2.jpg'
v1 = (444, 326)
v2 = (529, 326)
find_lane(name4, v1, v2)

name5 = 'solidYellowLeft.jpg'
v1 = (444, 326)
v2 = (529, 326)
find_lane(name5, v1, v2)

name6 = 'whiteCarLaneSwitch.jpg'
v1 = (444, 326)
v2 = (529, 326)
find_lane(name6, v1, v2)


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[24]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    grayimg1 = grayscale(image)
    #grayname = name +'_grayscale.jpg'
    #cv2.imwrite(os.path.join('test_images_output/' , grayname), grayimg1)
    
    kernal_size = 3
    blurgrayimg1 = gaussian_blur(grayimg1, kernal_size)
    
    
    low_threshold = 84
    high_threshold = 168
    edges1 = canny(blurgrayimg1, low_threshold, high_threshold)
    
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(444, 326), (529, 326), (imshape[1],imshape[0])]], dtype=np.int32)
    #vertices = np.array([[0,imgshape1[0]], [450,315], [525,315], [imgshape1[1],imgshape1[0]]], dtype=np.int32)
    masked_img1 = region_of_interest(edges1, vertices)
    
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_len = 20
    max_line_gap = 15
    
    max_line_gap_con = 90
    
    line_img1 = hough_lines(masked_img1, rho, theta, threshold, min_line_len, max_line_gap) 
    line_con_img1 = hough_lines(masked_img1, rho, theta, threshold, min_line_len, max_line_gap_con)
    
    #final_seg_img1 = weighted_img(line_img1, img1, α=0.8, β=1., γ=0.)
    result = weighted_img(line_con_img1, image, α=0.8, β=1., γ=0.)
    #plt.figure()
    #plt.imshow(final_seg_img1)

    return result



# In[27]:


'''
img1=mpimg.imread('test_images/solidWhiteCurve.jpg')
plt.figure()
plt.imshow(img1)
'''
white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[28]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))



# In[25]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[26]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))



challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

