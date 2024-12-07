#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <ex1|ex2|ex3|ex4> [additional parameters]"
  exit 1
fi


EXECUTABLE=$1
shift 
ADDITIONAL_PARAMS="$@"

if [[ ! "$EXECUTABLE" =~ ^ex[0-9]+$ ]]; then
  echo "Error: Invalid executable name. Use ex1, ex2, ex3, or ex4."
  exit 1
fi

sed -i.bak "s|src/ex[0-9]/ex[0-9]_main.cpp|src/$EXECUTABLE/${EXECUTABLE}_main.cpp|g" CMakeLists.txt
sed -i "s|src/ex[0-9]/ex[0-9]_raster.h|src/$EXECUTABLE/${EXECUTABLE}_raster.h|g" CMakeLists.txt
sed -i "s|src/ex[0-9]/ex[0-9]_raster.cpp|src/$EXECUTABLE/${EXECUTABLE}_raster.cpp|g" CMakeLists.txt

rm -rf build
mkdir build
cd build


cmake -DCMAKE_BUILD_TYPE=RELEASE ..
make

DATA_FILE="../data/bunny_${EXECUTABLE}.json"
if [ ! -f "$DATA_FILE" ]; then
  echo "Error: Data file $DATA_FILE not found!"
  exit 1
fi

./assignment4 "$DATA_FILE" $ADDITIONAL_PARAMS
