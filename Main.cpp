#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

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

  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  try
  {
    imwrite("modded.png", img, compression_params);
  }
  catch (const cv::Exception& ex)
  {
    fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
  }

  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", img);
  waitKey(0);
  return 0;
}
