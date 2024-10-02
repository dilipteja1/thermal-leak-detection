
# Flir Image Extractor

FLIR thermal cameras like Flir E50 and Flir one Edge pro include both thermal and visual light
camera.

The resulting image is saved as a jpg image but both the original visual image and 
raw thermal sensor data are embedded in the jpg metadata

This python tool allows to extract the image metadata along with the meta data 
provided by the Thermal inspector and provide a data excel sheet.

The data sheet contains the image name along with the meta data attached to it.
This will be useful while collecting the data.
# Requirements
This tool relies on exiftool. It should be available to many operating systems including windows, mac and ubuntu

> Install Exiftool from https://exiftool.org/install.html based on your OS \
> git clone *this repo* 

It also needs Python and few python packages. Create a virtual env with the packages installed or install them globally 

> *#* pip install -r requirements.txt\

This will create an excel sheet with all the tags extracted and attached to the image.

# toml 
toml file is like a configuration file where one can change the configuration or settings. In our case, in toml file, except for the image tags, other meta data should be updated by the Inspector while collecting the data .

The **tags.toml file** is the file in which all the meta data tags related to a image are placed. They need to be updated by the user to which ever tags they want to extract \
> Provide images folder in the **path** tag in the tags.toml file \
> Provide window id in the **id** tag in the tags.toml file
1. images_path
2. builing tags
3. room tags
4. environment tags
5. object tags (window)
6. image tags

# Run the tool
> *#* python -m image_meta_data_extractor.py --toml tags.toml
