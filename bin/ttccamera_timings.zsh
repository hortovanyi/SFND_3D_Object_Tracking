#!/usr/bin/env zsh

#detectors = "HARRIS FAST BRISK ORB AKAZE SIFT"
#descriptors = "BRIEF ORB FREAK AKAZE SIFT"

for detector ( HARRIS FAST BRISK ORB AKAZE SIFT);
  do #echo "detector $detector";
     for descriptor (BRIEF ORB FREAK AKAZE SIFT);
       do #echo "descriptor $descriptor";
	  # echo "$detector-$descriptor";
	  if [[ "$detector" != "AKAZE" && "$descriptor" == "AKAZE" ]] 
	  then
	    echo "TTCCamera $detector-$descriptor | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |";
          else
	    ./3D_object_tracking $detector $descriptor|grep TTCCamera;
	  fi;
     done
done