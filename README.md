# 🐍Neural Network🐍 
Currently the code is an experiment to test Python/TensorFlow and Inception on the Matrix VM.
The code can run with the command "*python3 inception.py*"

I ran the script several time, the best precision I got war 99.37%

**INVOKING SCRIPT FROM NODE SERVER** 
```javascript
 var dataToSend;
 # use the path to the virtual env to invoke python script, then scriptname and image
 const python = spawn('/home/student/catenv2/bin/python3', ['pyimagere.py','test1.jpg']);
 // collect data from script
 python.stdout.on('data', function (data) {
  console.log('Pipe data from python3 script ...');
  dataToSend = data.toString();
 });
 // in close event we are sure that stream from child process is closed
 python.on('close', (code) => {
 console.log(`child process close all stdio with code ${code}`);
 // send data to browser
 res.send(dataToSend)
 });
```

**ATTENTION**: Currently the Matrix VM has Python 2.7 and 3.6 installed, to run the correct version use *python3* 

**ATTENTION**: The first time the code runs it downloads the Inception Neural Network, it is necessary internet connection.

The core of the code is composed of TensorFlow and Keras:
1. To learn more about [TensorFlow](https://www.tensorflow.org/) Dataflow Library. GitHub [TensorFlow](https://github.com/tensorflow) page. 
2. To learn more about [Keras](https://keras.io/) Neural Network library. GitHub [Keras](https://github.com/keras-team) page.

## Installation
Image Regognition demands the installation of some dependencies:
### PYTHON 3
The code runs on Python 3, therefore it is expected that you install Python3 with commands such:
```bash
apt-get update
apt-get -y upgrade
apt-get install -y python3-pip
apt-get install -y build-essential libssl-dev libffi-dev
apt-get install -y libsm6 libxrender1 libfontconfig1 libxext6 libxrender-dev
```

### CREATE PYTHON ENVIRONMENT 
Steps to install [virtualenv] (https://virtualenv.pypa.io/en/latest/)
as the Python container for Image Recognition. 
The following lines will install virtualenv and create an environment (named catenv) to run Python
```bash
sudo -H pip3 install --upgrade pip
sudo -H pip3 install virtualenv
virtualenv catenv
source catenv/bin/activate
```

### INSTALL PYTHON MODULES UNDER ENVIRONMENT
After activating the 'catenv' evironment it is necessary to install the last dependencies
The last dependencies for the Python
```bash
pip install numpy==1.18.1
pip install Pillow==7.0.0
pip install tensorflow-gpu==1.14.0
pip install Keras==2.3.1
pip install opencv-python==4.1.2.30

pip install ffmpeg-python==0.2.0
pip install python-vlc==3.0.7110
```

For the retraining neural network code it is also necessary to add Pandas and MatPlotLib
```bash
pip install pandas==1.0.1
pip install matplotlib==3.2.0
```
