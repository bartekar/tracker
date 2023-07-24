cmake_minimum_required(VERSION 3.5)
project(Tracker)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Tracker Main.cpp )
target_link_libraries( Tracker ${OpenCV_LIBS} )

