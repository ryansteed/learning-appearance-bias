#!/bin/bash
echo USAGE: ./classify_dir.sh [path/to/image_dir] [path/to/output_graph.pb] [path/to/output_labels.txt]
trap "exit" INT
source activate imagenet
mogrify -format jpg $1*.bmp
for filename in $1*.jpg; do
	python scripts/label_image.py \
		--graph=$2 \
		--labels=$3 \
		--input_layer=Placeholder \
		--output_layer=final_result \
		--image=$filename
done
