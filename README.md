# Emotion Detection from EEG



A KNN classifier to predict human emotions from EEG data

  - Due to an EULA, dataset is not included
  - The average accuracy results are 82.33% (valence) and 87.32% (arousal).

# Steps:
The preprocessed data is used for training the classifier. Steps involved in training the dataset:

- Extracting the dataset
- Finding features
- Reducing the dimension
- Training
- checking classifier efficiency


### Dataset Description

The [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) dataset consists of two parts:

* The participant ratings, physiological recordings and face video of an experiment where 32 volunteers watched 40 music videos. 
* EEG and physiological signals were recorded and each participant also rated the videos as above. 
* For a single participant the data is subdivided into label array and eeg_data array

| data | dim | contents |
| ------ | ------ |------|
| eeg_data | 40 x 40 x 8064	| video/trial x channel x data |
| labels | 40 x 4	 | video/trial x label (valence(1-9), arousal(1-9), dominance(1-9), liking(1-9)|




### Keypoints

- EEG data of ~10 participants is extracted then converted into vectors which is used as training data.
- The train_std.csv contains standard deviation of data of all 32 electrodes from each participant.
- Additional csv files are respective labels for the train.csv data.
- I have created KNN(k=3) classifier from scratch using [scipy.spatial.distance.canberra](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.canberra.html) as distance metric. 
- Now create feature vector from our operational data
- I have used valence-arousal model for classification.
- Based on that model, 5 class of emotions can be detected using my approach.

### Procedure
Install the dependencies and devDependencies and start running knn_predict.py.

```sh
$ cd Emotion-detection-from-EEG
$ python knn.predict.py
```

### Todos

 - Perform same task with an SVM
 - Training an DNN for increasing accuracy 


### Development

Want to contribute? Great!
You can [contact](mailto:shubhpachchigar@gmail.com) me for any suggestion or feedback!


License
----

MIT
