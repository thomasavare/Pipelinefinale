# Pipeline overview

This project aims at doing a speech classification of waste among 50 different classes.

The Pipeline is divided in different steps:

1. Loading the differents models that are used.
2. Audio recording (start and stop)
3. Speech-to-text (automatic speech recognition) in various languages (X $\rightarrow$ english).
4. Speech classification
5. go to step 2 or stop.

# setting up the virtual environment

Using anaconda, we can set up the environment using the ```environment.yml``` file. simply use the command:

```conda create --file environment.yml```

if SoundDevice is not being installed, inside the previously created do ```pip install sounddevice```

# Using the pipeline

There are 2 diffent way to use the pipeline. the first one is just a simple run.

```
\.pipeline.py -l [language, default: "italian"] 
              -s [whisper size, default: "base"] 
              -d [device number, default: 0]
              -b [blocksize of recording, default: 16000]
              -f [name of file to save the audio]
```

for more information, just do ```\.pipeline --help```

The first one is the looped version, so we can do multiple classification without having to reload the models.

```
\. pipelineloop.py -l [language]
                   -s [whisper size]
                   -b [blocksize]
                   -f [file name to save the last recording] 
```