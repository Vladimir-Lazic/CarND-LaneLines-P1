# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


### Reflection

### 1. Pipleline description

The pipeline for lane detection goes as follows: 

    1. Grayscale
    
        First we convert the input image to grayscale
    
    2. Gaussian blur 
    
        This is done so that the grayscale image will be smoother and the later 
        edge detection would be precise
        
    3. Canny edge detection 
    
        Canny edge detection will return a black and white image where 
        in white are the detected edges.
        
        Canny parameters:         
            low_threshold = 50
            high_threshold = 150
        
        However since we need to detect lanes on the road this is incomplete.
        
     4. Region masking
     
        Region masking is done to filter out all the reduntant edges. 
        
        We define a trapezoid as a geometrical figure for region masking. This is because
        based on the position of the lanes in the camera image they are located the lower
        half of the input image and as they go up the frame they become more and more narrow. 
        The lower corners of the trapezoid are the lower left and lower right bottom of the 
        input image. The top corners x and y coordinates are define the following way: 
            
            For X we define a roi_top_center which is the middle of the trapezoid top side. 
            The we define an offset which is either added or subtracted from the roi_top_center
            depending on if its the top left corner or the top right corner
            
            For Y we divide the height with the height_scale parameter. height_scale parameter
            was defined through trial and error by running the algorithm on an input image.
            
         After defining the trapezoid we apply the region masking function that only keeps the
         parts of the input image based that are within the defined region
         
      5. Hough transormation
      
         After performing the region masking we have an image with only lane edges on in. We 
         use the Hough transormation to get the coorinates of detected lanes. 
         
         Hough transformation parameters:         
             rho = 2  
             theta = np.pi / 180  
             threshold = 15  
             min_line_length = 40 
             max_line_gap = 20 
         
This process is used on every image in the input video. For the visualization of the result
we use the draw_lines function. The function takes as an input the result of the 
Hough transformation which is an numpy.ndarray containing the coordinates of each individual line.
The function then uses open cv line function to draw the lines on the image. Since the Hough
transformation returns an array of cooridinates of each individual line the modification was 
done in order the highlight the entire traffic lane. This is done using the following method:

       1. Slope calculation
       
          We calculate the slope of each line in order to determine weather it belongs to 
          the left lane or the right lane. The lines with the positive slope belong to
          the right lane and the lines with the negative slope belong to the left lane
          
       2. Points extrapolation 
       
           After determining wich line belongs to which lane then we need to determine 
           the start and end points of each traffic lane or the lowest and the highest point
           if we think of it as position within the image. Each coordinate is packed within the 
           array in the following way: [x1, y1, x2, y2]. For determining the lowest point for 
           the left lane we find the maximum value of y1 since that point is the closest to the 
           bottom of the image where the lane start is. The highest point is determined by lowest 
           value of y2 since that coordinate is closest to the image center. Note: since we have done
           region masking there should not be any point that has a Y value higher than the height of the
           trapezoid. For the right lanes the extrapolation is the same except the coordinates are 
           mirrored so we find the higest y2 and the lowest y1. 
           
           After defining the start and end points of each lane we use opencv.line function to 
           draw the lines on the image


### 2. Shortcomings


One shortcoming of the solution is in the region masking part of the code. Since 
the values used for defining the trapezoid for region masking are determined by 
trial and error they are specific for the input frames used. Now since not all 
vehicles have the same cameras mounted to the front we cannot assume the same 
height and width of the image and thus the region masking coordinates would not 
work on an image with different frame.

The second shortcoming could be in the extrapolation of lane points. The rather
simple of approach of determining the highest and lowest point by their Y axis is 
again specific for this input videos. For driving videos on different this 
approac might not work, especially if we have mulitple right and left lines detected. 


### 3. Improvements

Possible improvement of the first shortcoming would be to make the region masking more modules
and not dependent on the values that are determined though trial and error. One solution would
be to use ratios to determine how long and wide to lanes run along the input video and determine 
which ratios of image height and width to use to determine the top right and left corners of the 
trapezoid. 
