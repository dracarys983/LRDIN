#!/bin/bash

EXPECTED_ARGS=3
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` video outdir frames/sec [size=256]"
  exit $E_BADARGS
fi

NAME=${1%.*}
DIR=$2
FRAMES=$3
BNAME=`basename $NAME`
mkdir -m 755 $DIR/$BNAME

ffmpeg -i $1 -r $FRAMES $DIR/$BNAME/$BNAME.%4d.jpg
