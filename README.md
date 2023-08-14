## Purpose

The following source code is a minimal viable prototype to show some basic techniques of computer vision:
- Video processing and rudimentary image editing
- People detection
- Object tracking
- Human pose estimation

# Content of Folder
- README.md      This document
- Main.cpp       The sole source file
- CMakeLists.txt Instructions to build the project
- gump.mp4       Demonstration video
- models/yolo/   Neural network to detect humans
- models/pose/   Neural network to compute the skeleton. Due to file size, this folder cannot be shared over git lfs and is therefore missing.

# Installation, Prerequisites and Building

- sudo apt install libopencv-dev=4.5.4
- mkdir build
- cd build
- cmake ..
- make

# What to expect

Executing the generated runnable loads a video named "gump.mp4" out of the parent folder and immediatly starts the processing steps:
- yolo5 net is used to detect people. The first person found will be tracked
- KCF tracker tracks movement of the person
- If KCF fails, then yolo5 is used again to find the person
- The bounding box of the person is drawn onto the image
- In each frame, pose-net (disbaled by default) may be used to draw the skeleton of whatever it finds...

Output:
- An mp4 file named "results.mp4" is created inside the build folder. It is identical to the input video but also contains the bounding box plus the drawn skeleton
- The file bounding_boxes.txt is generated, which contains the information of the bounding box for each frame

# Sources used to obtain neural networks
- https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
- https://github.com/doleron/yolov5-opencv-cpp-python/blob/main/cpp/yolo.cpp

# Further ideas:
- go over source code
- optimize skeleton net to run on person if bbox is big enough, run only on bbox
- draw skeleton only if confidence is high enough
- compare against another algorithm (DeepSORT?)
- add docker container

