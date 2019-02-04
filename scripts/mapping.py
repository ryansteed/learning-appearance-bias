import sys
import csv
import os

def map(file_labels, file_emotions, num_classes, skip=1):
	# print(file_emotions)
	# print(file_labels)
	# print(num_classes)

	# initialize new csv
	writer = csv.writer(open("{}_mapping.csv".format(file_labels.strip(".csv")), "w+"), delimiter=',')
	f = open(file_labels)
	# iterate through labels file
	row_to_write = None
	header = []
	for i, row_label in enumerate(f):
		if i < skip-1:
			continue
		if i % (num_classes+skip) == 0:
			if i==skip+num_classes:
				writer.writerow(header)
			if row_to_write is not None:
				writer.writerow(row_to_write)
			row_to_write = []
			
			filename = row_label.split("/")[-1].strip(".jpg\n")
			print(filename)
			reader_emotions = csv.reader(open(file_emotions, 'r'))
			for j, row_emotion in enumerate(reader_emotions):
				row_emotion[-1] = row_emotion[-1].strip("\n")
				if j == 0:
					if i==skip-1:
						header = row_emotion
						header[0] = header[0].strip("\ufeff")
						row_to_write = []
					continue
				if row_emotion[1] == filename:
					row_to_write += row_emotion

		else:
			split = row_label.strip("\n").split(" ")
			prob = split[-1]
			class_name = " ".join(split[:-1])
			if i < skip+num_classes:
				header.append(class_name)
			row_to_write.append(prob)
	writer.writerow(row_to_write)
	f.close()
		

if __name__ == "__main__":
	try:
		map(sys.argv[1], sys.argv[2], int(sys.argv[3]))
	except IndexError as e:
		print("Not enough args")
		print("USAGE: python mapping.py [path/to/labels.csv] [path/to/emotions.csv] [num_classes]")

