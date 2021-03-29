# Classification Analysis for Motion Data

The data is motion data, the aim is to decode the motion state from it.

## Raw Data

Data is from nowhere.
The data has been restored as .csv file.

- The 1st column is label state;
- The 2nd column is empty;
- The remaining columns are data:
  - There are 120 columns in total;
  - Every 6 columns formate a group for 6 channels;
  - Every group refers the data of 10 ms;
  - Thus, the data refers 200ms in total.

## Classification

### SVC Raw Classification

- File: [raw_classification.py](./raw_classification.py);
- Method: Classification using SVC using all 120 channels as features.

### Visualization

- File: [visualization.py](./visualization.py)
- Method: Generate data visualization for the events and channels.

### CNN Classification

- File: [cnn.py](./cnn.py)
- Method: Classification using CNN.

### SVM Raw Cov Classification

- File: [svm_raw_cov.py](./svm_raw_cov.py)
- Method: Classification using SVC using all 120 channels and cov features across channels.
- Detail: The 120 channels are of 20 times x 6 channels.
  The method is computing the covariance across channels.
  The results turn no improve.
