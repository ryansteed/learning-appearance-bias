#!/usr/bin/env bash

for dir in $1*/; do
	#echo $dir;
	#echo $1$(basename $dir).csv;
	face_detection ./$1$dir > $1$(basename $dir).csv;
	echo Outputted faces detected to $1$(basename $dir).csv;
	#ls $1
done
