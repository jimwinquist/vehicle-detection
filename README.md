#Vehicle Detection Project

##Overview
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_noncar1.png
[image2]: ./output_images/car_noncar2.png
[image3]: ./output_images/car_noncar3.png
[image4]: ./output_images/hog_features.png
[image5]: ./output_images/search_windows.png
[image6]: ./output_images/detections.png
[image7]: ./output_images/heatmap.png
[image8]: ./output_images/final.png
[video1]: ./video/final.mp4

##Histogram of Oriented Gradients (HOG)

### Extracting HOG Features.

The code for this step is contained in lines 87 through 135 of the file called `vehicle_detection.py` in the function `extract_features()`.  

For this project Udacity provided a training set of car and non-car images that come from  a combination of the GTI vehicle image database and the KITTI vision benchmark suite. I began by reading in all of the `vehicle` and `non-vehicle` images to get a sense of the training data for extracting HOG features. Each image in the dataset is saved as a 64x64 PNG and read in for training. 

Here are some examples of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]
![alt text][image3]

I experimented with a wide range of color spaces and different parameters for color extraction, spatial binning and HOG feature extraction. To find the best parameters I started by using the highest resolution possible for each possible setting and then narrowing it down from there. The initial features included all HOG channels, 64x64 spatial bins, 64 histogram bins, 9 hog orientations, 8 pixels per cell, and 2 cells per block. This allowed me to isolate the effects that color space has on training a classifier. I then began testing each possible color space to see which one provided the highest accuracy on the test set. The accuracy results of these tests were:

| Color Space   | Accuracy      | 
|:-------------:|:-------------:|
| RGB			| 0.9814		| 
| HSV           | 0.9932        | 
| LUV           | 0.9904        |
| HLS           | 0.9907        |
| YUV           | 0.9870        |
| YCrCb 	    | 0.9913		|

'RGB' performed the worst, and the rest all had fairly similar accuracy with HSV getting the best overall accuracy of 0.9932 on the test set. Originally, I experimented with lowering some of the additional parameters to just use a single hog channel, or lowering the orientations, spatial_size and hist_bins significantly. While this served to lower the feature space, and increased training and testing time, it ended up having a negative impact on the accuracy of the classifier. In the end, I found that the parameters that worked the best for training a classifier while reducing the feature space as much as possible, were the `HSV` color space and HOG parameters of `hog_channel="ALL"`, `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `spatial_size=(32,32)`, and `hist_bins=32`.


Here is an example of the HOG features extracted using the final parameters I settled on for implementing my pipeline:

![alt text][image4]

###Choosing HOG parameters.

I experimented with various parameters and using different hog channels and in the end I chose the one that seemed to provide the best accuracy over the entire dataset. I ended up using all three HOG channels combined with color histogram features, and spatial histogram features. I think there is still a lot of exploration that could be done in this area though to find even better parameters. I was trying to balance the need for speed of prediction with getting the best accuracy possible, and I settled on the set of parameters I chose by trying to find the sweet spot where I wasn't including too many features, but just enough to accurately classify the test windows as vehicle or non-vehicle. However, with a runtime of nearly 2 seconds per frame, this isn't practical for real time detection in it's current form, and would require additional experimentation to reduce the feature and search space in order to increase the frame rate.

###Training a Classifier
The code for this step is contained in lines 300 through 360 of the file called `vehicle_detection.py` in the function `train_classifier()`.

I trained a linear SVM using a balanced sample of vehicle and non-vehicle images from the data set. Using the feature extraction method described above, I extracted features for both vehicle and non-vehicle images and scaled the features to have zero mean and unit variance. I then split the data into 80% training set and 20% test set for validating the accuracy of the classifier. With image in the 'HSV' color space, using all 3 HOG channels, combined with color and spatial histograms, I was able to achieve approximately 0.993 accuracy on the test set. Feeling satisfied with the trained classifier I started to move on to testing the pipeline on real images and video streams.

##Sliding Window Search

The code for this step is contained in lines 152 through 221 of the file called `vehicle_detection.py` in the functions `slide_window()` and `search_windows()`.

I tested out different window sizes looking for both the minimum window size that could accurately detect car positions, and the maximum window size that could accurately detect car positions. For the minimum window size I found that 48x48 windows worked relatively well, and for the maximum I chose 256x256. Then I chose three intermediate window sizes in between and staggered them to make sure that as the cars moved into the distance I could pick up subtle variations due to their depth in the image. Since the cars are constrained to the road way I also limited the search space to only look in the lower half of the image and not search areas that contain mostly sky, or trees or other environment features. For the overlap, I found that significant overlap didn't didn't help that much with detection and just increased the time it took to do detection, so I kept an overlap of around 0.5 for each window.

![alt text][image5]

###Vehicle Detection Pipeline

One of the more challenging parts of doing vehicle deteciton was in trying to eliminate the large number of false positives that occured in the sliding window search. To eliminate these, I first tried to make sure that the region of interest where I was searching was constrained to the road surface. I chose global values for the search space, but in the future I would like to experiment with deriving these values dynamically using the lane finding process that I've developed. 

Ultimately I searched on 5 scales using HSV color space, 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided the following result.  

Below is an example of the raw detections for a test image:

![alt text][image6]
---

## Video Implementation

###Here's a [link to my video result](./video/final.mp4)


###Eliminating False Positives

The code for this step is contained in lines 362 through 422 of the file called `vehicle_detection.py` in the function `pipeline()`. Here I have all the video processing steps for defining the search windows, averaging them over multiple frames, and applying a heatmap and threshold for eliminating false positives.

After using the extracted features to make detections for each frame of the video, the other thing that was critical was to apply a heat map to accumulate and average detections over several frames of video. By applying a threshold to the averaged heat map, it made sure that only really strong detections were kept for the final output. This greatly reduces the number of false positives by eliminating detections that only occur on a single frame but aren't detected on subsequent frames. This leaves you with only the detections that are truly associated with a vehicle and allows you to group the heat map detections into discrete labels associated with multiple vehicles being detected. Using this final heat map I was able to draw the bounding boxes to accurately identify the vehicles in the image.

### Raw Detections
![alt text][image6]

### Heat Map
![alt text][image7]

### Final Bounding Boxes
![alt text][image8]

---

##Discussion

Overall I am happy with the initial implementation, and the detection seems to be working reasonably well for detecting multiple cars in a video stream. My current pipeline still detects a few false positives here and there and I would like to continue to work to remove them through better thresholding. Another area that I would like to improve is getting abetter fit for the bounding boxes around the vehicles detected. Right now the bounding boxes are a little jumpy and don't always surround the entire vehicle. 

This pipeline has only been tested on a limited amount of video, which contains ideal weather conditions and lighting. To prove the robustness of the process, it is important to gather a lot more training data and test out detecting vehicles, in day time and night time on a variety of road surfaces, with varying lighting conditions. I also would like to spend a lot more time tuning the various parameters to improve the accuracy of the classifier, and improve the process for thresholding the heat map so that the detected bounding boxes search in a neighborhood of the detections, instead of applying a global threshold. There is also a lot more room to experiment with different color spaces, feature types and window size for tuning the overall process. Beyond tuning the current pipeline it would also be interesting to use a neural network for doing the classification and comparing how well it performs vs. the HOG linear support vector machine approach.

In addition to tuning the algorithm, it is also important to improve the runtime and memory requirements for this process. My pipeline currently only runs at 2 seconds per frame and would need to be sped up a lot to be useful in a real vehicle. I think reducing the search space and limiting the number of features to some core subset, as well as experimenting with different classifiers could help in both regards. One of the most difficult parts of this process is balancing the accuracy vs. runtime cost to get an acceptable result. For this project I chose to focus on accuracy, but in order to improve the runtime I think I would need to spend more time narrowing both the feature space for training and prediction, and narrowing down the window search space to get the right granularity without over searching, since this was responsible for a large part of the slow down in speed.
