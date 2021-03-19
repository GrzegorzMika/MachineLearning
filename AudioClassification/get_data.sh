#!/bin/bash

source ./paths.sh

if [ ! -d $LOCATION ]; then
  echo "Creating directory $LOCATION"
  mkdir $LOCATION
else
  echo "Directory $LOCATION already exists"
fi

if [ -f $LOCATION$FILE ]; then
  echo "Skipping downloading of $FILE ..."
else
  echo "Downloading $FILE ..."
  wget https://www.openslr.org/resources/12/$FILE -P $LOCATION
  tar -xzf $LOCATION$FILE --directory $LOCATION
fi

echo "Ready!"