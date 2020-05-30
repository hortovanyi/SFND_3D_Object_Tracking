
#include <iostream>
#include <algorithm>
#include <numeric>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{

    // clearout any existing keypoint matches against the boundingBox
    boundingBox.kptMatches.clear();

    // find matches enclosed in BB and accumulate distance between keypoints
    std::vector<cv::DMatch> bbMatches;
    double distanceTotal;

    auto match_distance = [kptsPrev, kptsCurr](const cv::DMatch &dmatch) -> double {
        // // extract the keypoint to match on
        // cv::Point2f prev_pt = kptsPrev[dmatch.queryIdx].pt;
        // cv::Point2f curr_pt = kptsCurr[dmatch.trainIdx].pt;

        // cv::Point2f diff = prev_pt - curr_pt;
        // return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
        return cv::norm(kptsCurr[dmatch.trainIdx].pt - kptsPrev[dmatch.queryIdx].pt);
    };

    for (const auto dmatch: kptMatches){
        // extract the keypoint to match on
        cv::KeyPoint curr_kp = kptsCurr[dmatch.trainIdx];

        if (boundingBox.roi.contains(curr_kp.pt)) {
            bbMatches.push_back(dmatch);
            distanceTotal+=match_distance(dmatch);
        }
    }

    // if we have matches filter outliers or append to kptMatches
    if (bbMatches.size() > 0 ) {
        // calcualte std deviation
        double mean = distanceTotal/bbMatches.size();
        
        auto add_square = [mean] (double sum, double i)
        {
            auto d = i - mean;
            return sum + d*d;
        };

        double varianceTotal = 0.0;
        for (const auto dmatch: bbMatches) {
            varianceTotal = add_square(varianceTotal, match_distance(dmatch));
    
        }
        double variance = varianceTotal / bbMatches.size();
        double sigma = sqrt(variance);

        cout <<"filter boxID:" << boundingBox.boxID << " bbMatches.size(): " <<  bbMatches.size();
        cout <<" mean: " << mean << " variance: " << variance <<" sigma: " << sigma << endl;

        // filter outliers or append the match
        for (const auto dmatch: bbMatches) {
            double d =  match_distance(dmatch) - mean;
            if (fabs(d) > sigma ) {
                cout << "outlier - d: " << d << endl;
            } else {
                boundingBox.kptMatches.push_back(dmatch);
            }
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    // double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    // double dT = 1 / frameRate;
    // TTC = -dT / (1 - meanDistRatio);

    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1.0/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane
    double laneLeft = 0 - laneWidth/2;
    double laneRight = 0 + laneWidth/2;

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (it->y > laneLeft && it->y < laneRight)
        minXPrev = minXPrev > it->x ? it->x : minXPrev;
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (it->y > laneLeft && it->y < laneRight)
        minXCurr = minXCurr > it->x ? it->x : minXCurr;
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    typedef std::map<cv::DMatch, int> MatchBBMaxOccurenceMap;
    typedef std::map<cv::DMatch, std::set<int>> MatchBBsMap;
    typedef std::map<int, std::set<cv::DMatch>> BBMatchesMmap;
    typedef std::map<int, size_t> BBDmatchOccurenceMap;
    typedef std::map<std::pair<int,int>, size_t> BBsMatchesMap;
    
    MatchBBsMap prevMatchBBs, currMatchBBs;
    BBMatchesMmap prevBBMatches, currBBMatches;
    BBDmatchOccurenceMap prevBoxSizes, currBoxSizes;

    for (cv::DMatch dmatch: matches) {

        // extract the keypoint to match on
        cv::KeyPoint prev_kp = prevFrame.keypoints[dmatch.queryIdx];
        cv::KeyPoint curr_kp = currFrame.keypoints[dmatch.trainIdx];

        // find the bounding boxes in prev and curr that encloses the match keypoint 
        auto bb_enclosed = [dmatch] (cv::KeyPoint kp_match, DataFrame *frame, MatchBBsMap &matchBBs, BBMatchesMmap &BBMatches) {
            for (auto bb: frame->boundingBoxes)
                if (bb.roi.contains(kp_match.pt)) {
                    matchBBs[dmatch].insert(bb.boxID);
                    BBMatches[bb.boxID].insert(dmatch);
                }
        };   

        bb_enclosed(prev_kp, &prevFrame, prevMatchBBs, prevBBMatches);
        bb_enclosed(curr_kp, &currFrame, currMatchBBs, currBBMatches);
    }

    // create a boxid to number of matches lookup
    cout << "prev boxId (dmatch size)s:";
    for(auto x: prevBBMatches) {
        prevBoxSizes[x.first]=x.second.size();
        cout <<" "<< x.first << " (" << x.second.size() << ")";
    }
    cout << endl;
    cout << "curr boxId (dmatch size)s:";
    for(auto x: currBBMatches) {
        currBoxSizes[x.first]=x.second.size();
        cout <<" "<< x.first << " (" << x.second.size() << ")";
    }
    cout << endl;

    // create a Dmatch to number of boxids lookup
    std::map<cv::DMatch, size_t> prevDmatchSizes, currDmatchSizes;
    for(auto x: prevMatchBBs) {
        prevDmatchSizes[x.first]=x.second.size();
    }
    for(auto x: prevMatchBBs) {
        currDmatchSizes[x.first]=x.second.size();
    }


    // only use the boxid that encloses a dmatch with having the highest number of keypoint matches
    auto max_occurence = [](MatchBBsMap matchBBs, BBDmatchOccurenceMap boxSizes, MatchBBMaxOccurenceMap &matchBBMaxOccurence) {
        for (auto x: matchBBs) {
            auto dmatch = x.first;
            auto boxIDs = x.second;

            int maxOccurenceBoxId = -1; 
            size_t maxOccurence = 0;
            for (int id: boxIDs){
                int occurence = boxSizes[id];
                if (occurence>maxOccurence) {
                    maxOccurence=occurence;
                    maxOccurenceBoxId=id;
                }
            }

            // store the max occurence
            if (maxOccurenceBoxId>=0)
                matchBBMaxOccurence[dmatch]=maxOccurenceBoxId;
        }
    };

    MatchBBMaxOccurenceMap prevMatchBBMax, currMatchBBMax;
    max_occurence(prevMatchBBs, prevBoxSizes, prevMatchBBMax);
    max_occurence(currMatchBBs, currBoxSizes, currMatchBBMax);


    // create the best box matches 
    BBsMatchesMap boundingBoxesMatches;
    std::set<int> prevBoxIdSet, currBoxIdSet;
    cout << "bounding box max matches {prev,curr} (n_dmatches):";
    for (auto x: currMatchBBMax) {
        cv::DMatch dmatch = x.first;
        int curr_box_id = x.second;
        int prev_box_id = prevMatchBBMax[dmatch];

        // sum how many bouding boxs this descriptor match is in
        size_t n_dmatches = prevDmatchSizes[dmatch] + currDmatchSizes[dmatch];
        auto id_pair = std::make_pair(prev_box_id, curr_box_id);

        // only interested in the max occurence feature for a box id pair
        if (n_dmatches > boundingBoxesMatches[id_pair]) {
            boundingBoxesMatches[id_pair]=n_dmatches;
            prevBoxIdSet.insert(prev_box_id);
            currBoxIdSet.insert(curr_box_id);
            cout << " {" <<prev_box_id <<","<<curr_box_id<<"}("<<n_dmatches<<")";  
        }
    }
    cout << endl;

    cout << "prevBoxIdSet:";
    for (auto id: prevBoxIdSet) {
        cout <<" " << id;
    }
    cout <<endl;

    cout << "currBoxIdSet:";
    for (auto id: currBoxIdSet) {
        cout <<" " << id;
    }
    cout <<endl;

    cout << "boundingBoxesMatches:";
    for (auto x: boundingBoxesMatches) {
        cout << " {" << x.first.first << "," <<x.first.second << "}(" << x.second<< ")";
    }
    cout <<endl;

    // find the best match from prev to curr by selecting the curr with the most matches 
    std::set<int> remainingCurrBoxIdSet = currBoxIdSet;
    for (auto id: prevBoxIdSet) {
        int currBoxIdBest = -1;
        size_t bestMatches = 0;
        for (auto x: boundingBoxesMatches){
            // look for a prev box id match
            int prevBoxId = x.first.first;
            int currBoxId = x.first.second;
            // continue to the next if we've already used this curr box id
            if (remainingCurrBoxIdSet.count(currBoxId)==0)
                continue;
            size_t n_matches = x.second;
            if (prevBoxId == id && n_matches >bestMatches) {
                currBoxIdBest = currBoxId;
                bestMatches = n_matches;
            }
        }
        // use the best curr match if one found and has not already be assigned
        if (currBoxIdBest>=0 && remainingCurrBoxIdSet.count(currBoxIdBest)>0){
            bbBestMatches.insert({id, currBoxIdBest});
            remainingCurrBoxIdSet.erase(currBoxIdBest);
        }
    }

    // print bbBestMatches
    cout << "bbBestMatches (prev,curr):";
    for (auto x: bbBestMatches){
        int prev_box_id = x.first;
        int curr_box_id = x.second;
        cout << " {" <<prev_box_id <<","<<curr_box_id<<"}";  
    }
    cout << endl;


}
