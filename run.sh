#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <ex1|ex2|ex3|ex4|ex5>"
  exit 1
fi

# Set the executable to run
EXECUTABLE=$1

# Modify CMakeLists.txt
sed -i.bak "s|src/ex[0-9]/ex[0-9]_main.cpp|src/$EXECUTABLE/${EXECUTABLE}_main.cpp|g" CMakeLists.txt
sed -i "s|src/ex[0-9]/ex[0-9]_raster.h|src/$EXECUTABLE/${EXECUTABLE}_raster.h|g" CMakeLists.txt
sed -i "s|src/ex[0-9]/ex[0-9]_raster.cpp|src/$EXECUTABLE/${EXECUTABLE}_raster.cpp|g" CMakeLists.txt

# Remove and recreate the build directory
rm -rf build
mkdir build
cd build

# Run cmake and make
cmake -DCMAKE_BUILD_TYPE=DEBUG ..
make

# Execute the binary with the specified data file
./assignment4 ../data/bunny_ex2.json v
