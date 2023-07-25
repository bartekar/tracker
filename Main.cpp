#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*

sources:
- object detection: code + neural network
https://github.com/doleron/yolov5-opencv-cpp-python/blob/main/cpp/yolo.cpp
- object tracking:
TBD

*/

struct MyBBox
{
  int class_id;
  float confidence;
  Rect bbox;
};

typedef struct MyBBox MyBBox;

void draw_bbox(Mat img, Rect r)
{
  if (r.x + r.width > img.cols)
  {
    cout << "bbox is too high"  << endl;
    return;
  }
  if (r.y + r.height > img.rows)
  {
    cout << "bbox is too wide"  << endl;
    return;
  }

  for (int h=r.y; h < r.y+r.height; h++)
  {
    img.at<Vec3b>(h,r.x)[0] = 255;
    img.at<Vec3b>(h,r.x)[1] = 127;
    img.at<Vec3b>(h,r.x+r.width)[0] = 255;
    img.at<Vec3b>(h,r.x+r.width)[1] = 127;
  }
  for (int w=r.x; w < r.x+r.width; w++)
  {
    img.at<Vec3b>(r.y,w)[0] = 255;
    img.at<Vec3b>(r.y,w)[1] = 127;
    img.at<Vec3b>(r.y+r.height,w)[0] = 255;
    img.at<Vec3b>(r.y+r.height,w)[1] = 127;
  }
}

void detect_humans(Mat img, vector<struct MyBBox> &output)
{
  const float INPUT_WIDTH = 640.0;
  const float INPUT_HEIGHT = 640.0;
  const float SCORE_THRESHOLD = 0.2;
  const float NMS_THRESHOLD = 0.4;
  const float CONFIDENCE_THRESHOLD = 0.4;

  dnn::Net nnet = dnn::readNet("../yolov5s.onnx"); // todo: remove unnecessary files out of the repository
  nnet.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
  nnet.setPreferableTarget(dnn::DNN_TARGET_CPU);

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

  const int dimensions = 85;
  const int rows = 25200;

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
      if (max_class_score > SCORE_THRESHOLD)
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

int main(int argc, char** argv)
{
  if ( argc != 2 )
  {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }
  Mat input_img;
  input_img = imread( argv[1], IMREAD_COLOR ); // rgb image

  Rect bbox(100, 10, 200, 50);
  draw_bbox(input_img, bbox); // manipulating red and green channels (index 0 and 1)

  if ( !input_img.data )
  {
    printf("No image data \n");
    return -1;
  }

  Mat img;
  cvtColor( input_img, img, COLOR_RGB2BGR ); // rgb -> bgr, so that one can actually see the results
  // from now on it is bgr

//  dnn::Net net = dnn::readNet("../yolov5s.onnx"); // todo: remove unnecessary files out of the repository
//  net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
//  net.setPreferableTarget(dnn::DNN_TARGET_CPU);

  vector<MyBBox> people;
  detect_humans(img, people);
  for (vector<MyBBox>::iterator iter = people.begin(); iter != people.end(); ++iter)
  {
    draw_bbox(input_img, iter->bbox);
  }


  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  try
  {
    imwrite("modded.png", input_img, compression_params);
  }
  catch (const cv::Exception& ex)
  {
    fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
  }

  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", input_img);
  waitKey(0);
  return 0;
}
