#!/bin/bash
trap "exit" INT
source activate imagenet
mogrify -format jpg $1*.bmp
for filename in $1*.jpg; do
	python label_image.py \
		--graph=/tmp/output_graph.pb \
		--labels=/tmp/output_labels.txt \
		--input_layer=Placeholder \
		--output_layer=final_result \
		--image=$filename
done