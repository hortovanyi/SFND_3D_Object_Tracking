Track an Object in 3D Space
============================
## Final Report
Following is a final report write up for the Udacity Sensor Fusion Nanodegree section related to tracking an object in 3D using lidar and camera sensors to determine a time to collision for a vehicle that is in the same ego lane.
![camera image](images/KITTI/2011_09_26/image_02/data/0000000000.png)

## Match 3D Objects

The method `matchBoundingBoxes` was implemented in `camFusion_Student.cpp`. It takes as input both the current and previous data frames (which are already populated with YOLO detected bounding boxes for each frame) and a corresponding set of Key Point descriptor matches between the current and previous images contained in the data frame. The output is an updated `bbBestMatches` variable containing a unique set of previous bounding boxes matched to current bounding boxes. The bounding boxes are represented by a unique integer `boxId`.

![bounding box classification](output/Object_classification_screenshot_03.06.2020.png)

The algorithm first iterates through the matches, extracting the keypoints referenced. 

It then for each of the sets of bounding boxes determines if the keypoint is enclosed by it and records a unique set of box id matches along with updating a count (for the bounding box).

A counter is updated in `boundingBoxesMatches` for each box id (previous, current) pair match found. The counter determines, which pair has the most occurences of a match. As well it stores a complete set of box ids found for both current and previous.
```
prev boxId (KeyPoint size)s: 0 (516) 1 (454) 2 (57) 3 (339) 4 (8) 5 (209)
curr boxId (KeyPoint size)s: 0 (560) 1 (390) 2 (46) 3 (312) 4 (170) 5 (203) 6 (29) 7 (27) 8 (42) 9 (35) 10 (110)
bounding box max keypoints {prev,curr} (n_keypoints): {0,0}(515) {0,1}(1) {0,3}(117) {0,7}(10) {1,1}(369) {1,3}(1) {1,4}(169) {1,6}(4) {2,2}(45) {2,5}(14) {2,6}(4) {3,0}(128) {3,3}(309) {3,8}(21) {3,10}(88) {4,8}(4) {5,2}(13) {5,5}(200) {5,9}(6)
prevBoxIdSet: 0 1 2 3 4 5
currBoxIdSet: 0 1 2 3 4 5 6 7 8 9 10

```

To make sure that bounding box matches happen from candidates with the highest number of occurrences, a set of descending match counts is created.
```
prevBoxIdKeyPointCountDescending boxId (matches): 0 (516) 1 (454) 3 (339) 5 (209) 2 (57) 4 (8)
```

The `bbBestMatches` is updated with a previous and current box id pair, where the corresponding current box id, is a match with the same classification and the highest match count from the `boundingBoxesMatches` created earlier. A current box id can only be used once and the intiial set is populated from that determined in `currBoxIdSet`.

```
matching prevBoxId: 0 {0,0}(515)<- {0,1}(1)? {0,3}(117)? {0,7}(10) currBoxIdBest: 0 matches (shared, prev, curr): (515, 516, 560)
matching prevBoxId: 1 {1,1}(369)<- {1,3}(1)? {1,4}(169) {1,6}(4)? currBoxIdBest: 1 matches (shared, prev, curr): (369, 454, 390)
matching prevBoxId: 3 {3,0}(128)? {3,3}(309)<- {3,8}(21)? {3,10}(88)? currBoxIdBest: 3 matches (shared, prev, curr): (309, 339, 312)
matching prevBoxId: 5 {5,2}(13)<- {5,5}(200)<- {5,9}(6) currBoxIdBest: 5 matches (shared, prev, curr): (200, 209, 203)
matching prevBoxId: 2 {2,2}(45)<- {2,5}(14)x {2,6}(4) currBoxIdBest: 2 matches (shared, prev, curr): (45, 57, 46)
matching prevBoxId: 4 {4,8}(4)<- currBoxIdBest: 8 matches (shared, prev, curr): (4, 8, 42)
```
note: `<-` shows the highest pair found as it loops through, `?` box classification different, `x` current box id already allocated. 

The result of this is `bbBestMatches` updated with a previous box id, current box id pair as follows. The graphic also shows the colour coded matches with the box ids.
```
bbBestMatches {prev,curr}: {0,0} {1,1} {2,2} {3,3} {4,8} {5,5}
```

![bounding box best matches](output/BoundingBox_Best_Matches_screenshot_03.06.2020.png)

note: the box colours are unique for each bounding box match (so the left with the previous image should match to the current image on the right)

## Compute Lidar-based TTC

The method `computeTTCLidar` was implemented in `camFusion_Student.cpp`. It takes as input the previous and current lidar points with a frame rate (in Hz) to return a Time To Collision value in seconds.

An ego lane width of 4 meters is assumed. Any lidar point not within the ego lane is ignored.

The minimum closest point is found in front of the ego vehicle for previous and current lidar points.

The Time To Collisiion is computed 
```
TTC = minXCurr * dT / (minXPrev - minXCurr);
```
where `dT` is the time between the two measuremetns in second.

The lidar points are prefiltered in the method `filterLidarPointXOutliers` which was implemented in `FinalProject_Camera.cpp`. It calculates the std deviation - sigma and removes points that are 2 times sigma between ego and the vehicle ahead in the ego lane.

```
filter lidarPoints.size(): 307 mean: 7.61796 variance: 0.00481561 sigma: 0.0693946
outlier - d: -0.142964 (7.475,-0.821,-0.9)
```

![Top Down view of Lidar Points](output/Top-ViewPerspectiveLiDAR_data_currBB_lidarPoints_screenshot_03.06.2020.png)

## Associate Keypoint Correspondences with Bounding Boxes

This method `clusterKptMatchesWithROI` prepares the TTC computation based on camera measurements by associating keypoint correspondces to the bounding boxes which enclose them. Its implemented in `camFusion_Student.cpp`.

Euclidean distance calculation usese the following lambda function
```c++
auto match_distance = [kptsPrev, kptsCurr](const cv::DMatch &dmatch) -> double {
    // extract the keypoint to match on - this is faster then cv::norm
    cv::Point2f prev_pt = kptsPrev[dmatch.queryIdx].pt;
    cv::Point2f curr_pt = kptsCurr[dmatch.trainIdx].pt;

    cv::Point2f diff = prev_pt - curr_pt;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
};
```

`kptrPrev` and `kptsCurr` are supplied to the method as well as a vector of keypoint descriptor matches.

If the current key point contained in the match is enclosed in the passed itbounding box, its added into `bbMatches` and the euclidean distance between  the current and previous key points totaled.

A standard deviation - sigma, is calculated from all the `bbMatches` and outliers determined by the absolute distance being greater than sigma not recorded in `boundingBox.kptMatches`.

```
filter boxID:4 bbMatches.size(): 215 mean: 1.51281 variance: 50.6649 sigma: 7.11793
outlier - d: 102.967
```


## Compute Camera-based TTC


The method `computeTTCCamera` was implemented in `camFusion_Student.cpp`. It takes as input the previous and current keypointss, associated descriptor matches, with a frame rate (in Hz) to return a Time To Collision value in seconds.

Distance ratios are computed between all matched keypoints.

If there are not distance ratios found, NAN (Not a Number) is returned

The following code is used to calculate the Time To Collission and to deal with outliers.

```
std::sort(distRatios.begin(), distRatios.end());
long medIndex = floor(distRatios.size() / 2.0);
double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

double dT = 1 / frameRate;
TTC = -dT / (1 - medDistRatio);
```

## Performance Evaluation 1
Following are examples of where the TTC estimates from the lidar sensor do not seem plausible with associated observations and discussion.

### TTC liDAR Log
```
TTC: 12.9722 minXCurr: 7.913 minXPrev: 7.974 dT: 0.1
TTC: 12.264 minXCurr: 7.849 minXPrev: 7.913 dT: 0.1
TTC: 13.9161 minXCurr: 7.793 minXPrev: 7.849 dT: 0.1
TTC: 14.8865 minXCurr: 7.741 minXPrev: 7.793 dT: 0.1
TTC: 12.1873 minXCurr: 7.678 minXPrev: 7.741 dT: 0.1
TTC: 7.50199 minXCurr: 7.577 minXPrev: 7.678 dT: 0.1
TTC: 34.3404 minXCurr: 7.555 minXPrev: 7.577 dT: 0.1
TTC: 18.7875 minXCurr: 7.515 minXPrev: 7.555 dT: 0.1
TTC: 15.8894 minXCurr: 7.468 minXPrev: 7.515 dT: 0.1
TTC: 13.7297 minXCurr: 7.414 minXPrev: 7.468 dT: 0.1
TTC: 10.4914 minXCurr: 7.344 minXPrev: 7.414 dT: 0.1
TTC: 10.1 minXCurr: 7.272 minXPrev: 7.344 dT: 0.1
TTC: 9.22307 minXCurr: 7.194 minXPrev: 7.272 dT: 0.1
TTC: 10.9678 minXCurr: 7.129 minXPrev: 7.194 dT: 0.1
TTC: 8.09422 minXCurr: 7.042 minXPrev: 7.129 dT: 0.1
TTC: 8.81392 minXCurr: 6.963 minXPrev: 7.042 dT: 0.1
TTC: 10.2926 minXCurr: 6.896 minXPrev: 6.963 dT: 0.1
TTC: 8.30978 minXCurr: 6.814 minXPrev: 6.896 dT: 0.1
```
### Final Result Examination 1
![TTC Lidar 34.340420](output/FinalResults1_TTC_screenshot_04.06.2020.png)

```
prev filter lidarPoints.size(): 345 mean: 7.72909 variance: 0.00584972 sigma: 0.0764834
curr filter lidarPoints.size(): 315 mean: 7.67321 variance: 0.00524091 sigma: 0.0723942
TTC: 34.3404 minXCurr: 7.555 minXPrev: 7.577 dT: 0.1
```
#### Top View Previous LiDAR data
<img src="output/FinalResults1_TopViewLiDARdata_prev_screenshot_04.06.2020.png"  width="200px">

#### Top View Current LiDAR data
<img src="output/FinalResults1_TopViewLiDARdata_curr_screenshot_04.06.2020.png" width="200px">

### Observation and reasoning
The gap between the current and previous measurement dropped, possibly indicating a breaking event in the ego vehicle and a break release from the vehicle ahead.  This is supported by the sudden drop to 7.50199 TTC in the preceeding frame, which itself was a drop of ~5 seconds from its precedessor. The sudden drop means that the vehicle in front of the ego vehicle, slowed suddenly. All vehicles displayed have break lights on. Breaking events would alter the pitch of the vehicle and affect the time of flight calculations. When observing the top down views of the lidar data, the shapes appear to be slightly different which would be as a result of hitting the vehicle in different points.


### Final Result Examination 2
![TTC Lidar 10.2926](output/FinalResults2_TTC_screenshot_04.06.2020.png)

```
prev filter lidarPoints.size(): 287 mean: 7.03719 variance: 0.00167172 sigma: 0.0408867
prev filter lidarPoints.size(): 273 mean: 6.97324 variance: 0.00181314 sigma: 0.0425809
TTC: 10.2926 minXCurr: 6.896 minXPrev: 6.963 dT: 0.1
````
#### Top View Previous LiDAR data
<img src="output/FinalResults2_TopViewLiDARdata_prev_screenshot_04.06.2020.png"  width="200px">

#### Top View Current LiDAR data
<img src="output/FinalResults2_TopViewLiDARdata_curr_screenshot_04.06.2020.png" width="200px">

### Observation and reasoning

Minor noise in the lidar data probably caused by slightly different reflective points, has resulted in a once off increase in TTC. This can be observed by the difference in shape between the current and previous top views of the lidar data. As the vehicles near similar speeds leading to the difference between the two minimums of X coming closer to zero, it will also naturally cause the TTC to increase.
