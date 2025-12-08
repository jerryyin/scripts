import os
import datetime
import PIL.Image
import PIL.ExifTags
import shutil


# source_dir = "/mnt/c/SAVE/Pictures/Soccer"
# source_dir = "/mnt/c/SAVE/Pictures/102CANON"
source_dir = "/mnt/c/SAVE/Pictures/Polar Caves Aug 2018"

dest_dir = "/mnt/c/SAVE/Pictures"


select_tags = {
  "Make",
  "Model",
  "DateTime",
  "DateTimeOriginal",
  "DateTimeDigitized",
  "GPSInfo",
}

skip_tags = {
  "MakerNote",
}

cameras = {
  "NIKON D3300" : "NIKON_D3300", 
  "Canon PowerShot SD1000" : "CANON_SD1000", 
}

def get_exif_tags(image_file):
  img = PIL.Image.open(image_file)
  exif_tags = {}
  for k,v in img._getexif().items():
    if k in PIL.ExifTags.TAGS:
      tag = PIL.ExifTags.TAGS[k]
      if tag in select_tags:
      # if tag not in skip_tags:
        exif_tags[tag] = v
  return exif_tags


def generate_name(year, camera, timestamp, location):
  return os.path.join(year,camera,timestamp)


def get_year(exif_tags):
  # date_and_time = exif_tags["DateTime"]
  date_and_time = exif_tags["DateTimeOriginal"]
  date = date_and_time.split()[0]
  year = date.split(":")[0]
  return year


def get_camera(exif_tags):
  camera = exif_tags["Model"]
  return cameras[camera]

  
def get_timestamp(exif_tags):
  date_and_time = exif_tags["DateTimeOriginal"]

  date = date_and_time.split()[0]
  year = int(date.split(":")[0])
  month = int(date.split(":")[1])
  day = int(date.split(":")[2])
  # print (date, year, month, day)
  
  time = date_and_time.split()[1]
  hour = int(time.split(":")[0])
  minute = int(time.split(":")[1])
  second = int(time.split(":")[2])
  # print (time, hour, minute, second)
  
  timestamp = datetime.datetime(year, month, day, hour, minute, second)
  return timestamp.strftime("%b_%d_%H_%M_%S")


def get_location(exif_tags):
  return None


def get_new_name(image_file):
  exif_tags = get_exif_tags(image_file)
  # for k,v in exif_tags.items():
  #   print("{} : {}".format(k,v))
  year = get_year(exif_tags)
  camera = get_camera(exif_tags)
  timestamp = get_timestamp(exif_tags)
  location = get_location(exif_tags)
  return generate_name(year, camera, timestamp, location)


def get_image_files(dirname):
  image_files = []
  for (dirpath, dirnames, filenames) in os.walk(dirname):
    for f in filenames:
      image_file = os.path.join(dirpath, f)
      image_files.append(image_file)
  return image_files


def copy_file(source, dest):
  _, ext = os.path.splitext(source)
  dest_file = "{}{}".format(dest, ext)
  incr = 0
  while os.path.isfile(dest_file):
    incr += 1
    dest_file = "{}_{:02d}{}".format(dest, incr, ext)
  print("{} --> {}".format(source, dest_file))
  shutil.copyfile(source, dest_file)


def main():
  images = get_image_files(source_dir)
  for image in images:
    new_name = get_new_name(image)
    new_name = os.path.join(dest_dir, new_name)
    copy_file(image, new_name)


if __name__ == '__main__':
  main()
