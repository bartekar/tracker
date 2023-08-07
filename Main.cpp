#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;
using namespace std;

/*

sources:
- object detection: code + neural network
https://github.com/doleron/yolov5-opencv-cpp-python/blob/main/cpp/yolo.cpp
- object tracking:
general pages...
- skeleton:
https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

*/

Scalar orange = Scalar(0, 127, 255); // bgr

struct MyBBox
{
  int class_id;
  float confidence;
  Rect bbox;
};

typedef struct MyBBox MyBBox;

void draw_bbox(Mat img, MyBBox bbox)
{
  Rect r = bbox.bbox;
  if (r.x + r.width > img.cols)
  {
    cout << "bbox is too wide"  << endl;
    return;
  }
  if (r.y + r.height > img.rows)
  {
    cout << "bbox is too high"  << endl;
    return;
  }


  rectangle(img, bbox.bbox, orange, 1);

  char buffer[5]; // e.g. 0.89 -> 4 chars + terminating char
  sprintf(buffer, "%.2f", bbox.confidence);
  putText(img, buffer, Point(r.x, r.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, orange);
}

void detect_humans(dnn::Net &nnet, Mat img, vector<struct MyBBox> &output)
{
  const float INPUT_WIDTH = 640.0;
  const float INPUT_HEIGHT = 640.0;
  const float SCORE_THRESHOLD = 0.4;
  const float NMS_THRESHOLD = 0.4;
  const float CONFIDENCE_THRESHOLD = 0.4;

  int mx = MAX(img.cols, img.rows);
  Mat img_square = Mat::zeros(mx, mx, CV_8UC3);
  img.copyTo(img_square(Rect(0, 0, img.cols, img.rows)));

  Mat input_img;
  dnn::blobFromImage(img_square, input_img, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false); // preprocessing
  nnet.setInput(input_img);

  vector<Mat> outputs;
  nnet.forward(outputs, nnet.getUnconnectedOutLayersNames());

  float x_factor = img_square.cols / INPUT_WIDTH;
  float y_factor = img_square.rows / INPUT_HEIGHT;

  float *data = (float *)outputs[0].data;

  const int dimensions = 85; // todo: understand magic number
  const int rows = 25200; // todo: this one too
  const int PERSON_CLASS = 0;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < rows; ++i)
  {
    float confidence = data[4];
    if (confidence >= CONFIDENCE_THRESHOLD)
    {
      float * classes_scores = data + 5;
      Mat scores(1, 80, CV_32FC1, classes_scores);
      Point class_id;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      if ((max_class_score > SCORE_THRESHOLD) && (PERSON_CLASS == class_id.x))
      {
        confidences.push_back(confidence);
        class_ids.push_back(class_id.x);
        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];
        int left = int((x - 0.5 * w) * x_factor);
        int top = int((y - 0.5 * h) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);
        boxes.push_back(Rect(left, top, width, height));
      }
    }
    data += 85;
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
  for (int i = 0; i < nms_result.size(); i++)
  {
    int idx = nms_result[i];
    MyBBox result;
    result.class_id = class_ids[idx];
    result.confidence = confidences[idx];
    result.bbox = boxes[idx];
    output.push_back(result);
  }
}

void paint_skeleton(Mat img, dnn::Net nnet)
{
  // Specify the input image dimensions
  int inWidth = 368;
  int inHeight = 368;
  // Prepare the frame to be fed to the network
  Mat inpBlob = dnn::blobFromImage(img, 1.0 / 255, Size(inWidth, inHeight), Scalar(0, 0, 0), false, false);
  // Set the prepared object as the input blob of the network
  nnet.setInput(inpBlob);

  Mat output = nnet.forward();

  int H = output.size[2];
  int W = output.size[3];

  int nPoints = 15;
  double thresh = 0.2;
  int frameWidth = img.cols;
  int frameHeight = img.rows;
  // find the position of the body parts
  vector<Point> points(nPoints);

  for (int n=0; n < nPoints; n++)
  {
    // Probability map of corresponding body's part.
    Mat probMap(H, W, CV_32F, output.ptr(0,n));

    Point2f p(-1,-1);
    Point maxLoc;
    double prob;
    minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
    if (prob > thresh)
    {
      p = maxLoc;
      p.x *= (float)frameWidth / W ;
      p.y *= (float)frameHeight / H ;
    }
    points[n] = p;
  }

  const int nPairs = 14;
  int pairs[nPairs*2] = {
            0,1, 1,2, 2,3, 3,4, 1,5, 5,6, 6,7, 1,14, 14,8, 8,9, 9,10, 14,11, 11,12, 12,13
        };
  for (int n = 0; n < nPairs; n++)
  {
    // lookup 2 connected body/hand parts
    Point2f partA = points[pairs[2*n+0]];
    Point2f partB = points[pairs[2*n+1]];

    if (partA.x<=0 || partA.y<=0 || partB.x<=0 || partB.y<=0)
        continue;

    line(img, partA, partB, Scalar(255,0,127), 8);
  }

  for (int n=0; n < nPoints; n++)
  {
    Point2f p = points[n];
    circle(img, cv::Point((int)p.x, (int)p.y), 8, Scalar(255,0,127), -1);
    cv::putText(img, cv::format("%d", n), cv::Point((int)p.x, (int)p.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 127, 0), 2);
  }

}

int main(int argc, char** argv)
{
  // some variables used during development
  const bool ENABLE_SKELETON = false; // en-/ disable the skeleton computations
  const bool ENABLE_DRAWING = false; // en-/ disable display of current frame + results
  Mat img;

  dnn::Net detector_nnet = dnn::readNet("../models/yolo/yolov5s.onnx"); // todo: remove unnecessary files out of the repository
  detector_nnet.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
  detector_nnet.setPreferableTarget(dnn::DNN_TARGET_CPU);


  // Specify the paths for the 2 files
  string protoFile = "../models/pose/pose_deploy_linevec_faster_4_stages.prototxt";
  string weightsFile = "../models/pose/pose_iter_160000.caffemodel";
  // Read the network into Memory
  dnn::Net skeleton_nnet;
  if (ENABLE_SKELETON)
    dnn::Net skeleton_nnet = dnn::readNetFromCaffe(protoFile, weightsFile);

  vector<MyBBox> people;

  // vars for tracker
  Ptr<Tracker> tracker;
  Rect tracker_box = Rect(0,0,0,0);
  tracker = TrackerMIL::create();

  VideoCapture cap("../gump.mp4");

  // Check if video was opened successfully
  if(!cap.isOpened())
  {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }

  Mat frame;
  cap >> frame;

  detect_humans(detector_nnet, frame, people);
  for (vector<MyBBox>::iterator iter = people.begin(); iter != people.end(); ++iter)
  {
    draw_bbox(frame, *iter);
  }
  tracker_box = people[0].bbox;
  people.clear();
  tracker->init(frame, tracker_box);

  const int DETECTION_PERIOD= 20;
  int loops_until_next_detection = 10;

  VideoWriter writer;
  int codec = VideoWriter::fourcc('a', 'v', 'c', '1');
  double fps = 24.0;
  string filename = "result.mp4";
  Size sizeFrame(1920,1080);
  writer.open(filename, codec, fps, sizeFrame, true);

  vector<int> bbox_data = vector<int>();

  int frame_number = 0;
  bbox_data.push_back(frame_number);
  bbox_data.push_back(tracker_box.x);
  bbox_data.push_back(tracker_box.y);
  bbox_data.push_back(tracker_box.width);
  bbox_data.push_back(tracker_box.height);

  while(1)
  {
    if (frame.empty())
      break;

    if (loops_until_next_detection-- <= 0)
    {
      detect_humans(detector_nnet, frame, people);
      for (vector<MyBBox>::iterator iter = people.begin(); iter != people.end(); ++iter)
      {
        draw_bbox(frame, *iter);
      }
      tracker_box = people[0].bbox;
      people.clear();
      tracker->init(frame, tracker_box);
      loops_until_next_detection = DETECTION_PERIOD;
    }
    else
    {
      bool ok = tracker->update(frame, tracker_box);
      if (ok)
      {
        rectangle(frame, tracker_box, orange, 1);
      }
    }
    bbox_data.push_back(++frame_number);
    bbox_data.push_back(tracker_box.x);
    bbox_data.push_back(tracker_box.y);
    bbox_data.push_back(tracker_box.width);
    bbox_data.push_back(tracker_box.height);

    if (ENABLE_SKELETON)
      paint_skeleton(frame, skeleton_nnet);

    // Display the resulting frame
    if (ENABLE_DRAWING)
    {
      imshow( "Frame", frame );
      char c=(char)waitKey(25); // Press  ESC on keyboard to exit
      if(c==27)
        break;
      }
    writer.write(frame);
    cap >> frame;
  }

  writer.release();
  cap.release(); // When everything done, release the video capture object
  destroyAllWindows(); // Closes all the frames

  std::ofstream out_file;
  out_file.open("bounding_boxes.txt", std::ios::app);
  out_file << "frame_number;x;y;width;height" << endl;
  for (int i = 0; i < frame_number; i++)
  {
    out_file << bbox_data.at(5*i) << ";" << bbox_data.at(5*i+1) << ";" << bbox_data.at(5*i+2) << ";" << bbox_data.at(5*i+3) << ";" << bbox_data.at(5*i+4) << endl;
  }
  out_file.close();

  return 0;
}
