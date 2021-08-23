# Watt-hour Meter OCR

This project was created as my final year project for the csc599 course at the Lebanese American University in 2021. This software helps extracting the 6 numbers of a watt-hour meter from an image. Thus helping the workers collecting these numbers for thousands of meters in a very short time. 

# Using the software 
## Runing the CNN
> open "Watt-hour meter CNN"  directory from the Terminal.


#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt
```

#### Activate the environment
```bash
conda activate yolov4-cpu
```
Now that we have our environement ready, locate the model file in `checkpoints/custom-416`, this is the trained model.
#### Specify the images directory       
Move the images that you want to run the CNN on, to the `data/images/` folder. 

#### Run the CNN on the images

```bash
python3 detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images --crop
```
*notice the --crop flag, to crop the KWh class and put it in one folder. 

#### Locate the output images 
The output images detected from the `data/images` folder will be saved in `detections/`folder and  the cropped KWh in `detections/crop` folder


## Read the cropped KWh images 
> Open "Read Cropped" folder from the initial folder using terminal.

#### Specify the cropped images directory 
Move the images that were cropped by the CNN to the crop folder inside `Read Cropped/crop` folder.
This folder is the input images of the `read.py` python file.

Since this code runs in parallel you need to specify the number of threads to run, according to your cpu number of cores the number of threads should be 80%. To do this, edit the file `read.py` in line `171` to the number of threads you want. 

Now inside the `Read Cropped` folder, run the `read.py` file using the following command.

```bash
python3 read.py
```   
When the code is done, you can open the output.csv file to see the output.

#Getting Help 

If you have questions, concerns, bug reports, etc, please file an issue to <sakr.log@gmail.com> 

#License

MIT License

Copyright (c) 2021 Jad Jean Sakr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.







