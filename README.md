# Audio classification deep learning model
 
The original data given by the GTZAN Dataset are in .wav format. In Kaggle, there is also the csv files that hold the extracted data from the audio files. I wrote a custom script to re-generate a new csv file from the input .wav audio files. The script to extract the features from the wav files is given by _extract_audio_features.ipynb_. Details of how the data extraction is done is given in the notebook. The final csv file should have the shape of (9990. 58).

# Model Definition

The diagram below shows the model architecture

![download](https://user-images.githubusercontent.com/6497242/171231195-c07ed786-b7b1-40d0-9bce-668c35242774.png)

# Model Training

The input data is reduced in dimension from (9990, 58) to (9990, 56) after the file name column and label columns are dropped. The dataset is then split into the train (80%) test (20%). The model training is done with the Model Checkpointing to save only the model when the validation accuracy is better than the earlier epoch, and also EarlyStopping to end the training if the validation accuracy does not improve afer a number of epochs. The model should complete training in less than 1 min.

# Model Evaluation

The model is able to hit classification accuracy of ~90.0% when evaluated against the test set. 
