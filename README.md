# WaveNet

Implementation of WaveNet.

## Prepare data

Before using download LJSpeech dataset:

```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar.bz2
```

To setup docker container run ./build_image.sh and ./run_container.sh

## Train

run train.py.

Data will be logged in wandb.

## Inference

Download the final model from https://drive.google.com/file/d/127YxeN7ZBoMZ04_ThRvC1Lonclerl7hg/view?usp=sharing

running inference.py will generate audio for one random example from the dataset. Wav file will be saved in "gen.wav" as well as in wandb. If you want to generate audio for specific mel-spectrogram you should pass mel-spectrogram of shape (1, n_mels, length_of_audio) to function "inference_fast".
