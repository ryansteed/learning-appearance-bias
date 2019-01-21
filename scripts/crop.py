from PIL import Image
import sys
import csv
import os
import errno
import face_recognition
import enlighten

def crop(image_path, saved_location):
	"""
	@param image_path: The path to the image to edit
	@param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
	@param saved_location: Path to save the cropped image
	"""
	image_obj = face_recognition.load_image_file(image_path)
	face_locations = face_recognition.face_locations(image_obj)
	for face_location in face_locations:
		top, right, bottom, left = face_location
		# print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
		face_image = image_obj[top:bottom, left:right] 
		pil_image = Image.fromarray(face_image)
		pil_image.save(saved_location)
		# print("Saved cropped face to {}".format(saved_location))


def get_stem_path(image_path):
	return image_path.split("/")[-1]


if __name__ == '__main__':
	"""
	Receive input CSV file, then:
	1. Create new directory matching the name of the CSV + "cropped"
	2. For each image in CSV file, 
	crop according to saved coordinates and save to the new directory.
	"""
	try:
		file_string = sys.argv[1]
		dir_string_new = "{}_cropped".format(file_string)

		try:
			os.mkdir(dir_string_new)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise e

		manager = enlighten.get_manager()
		ticker = manager.counter(
			total=len(os.listdir(file_string)),
			desc='Images Processed',
			unit='images'
		)
		for filename in os.listdir(file_string):
			if filename.endswith(".jpg"):
				crop(
					"{}/{}".format(file_string, filename), 
					"{}/{}".format(dir_string_new, filename)
				)
			ticker.update()

	except IndexError as e:
		print("Not enough args!\nUSAGE: python crop.py [path/to/csv]")
