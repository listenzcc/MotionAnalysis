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
- Method: Classification using SVC using all 120 features and cov features across channels.
- Detail: The 120 features are of 20 times x 6 channels.
  The method is computing the covariance across channels.
  The results turn no improve.

### SVM Generation and Channel Classification

- Files: [svm_channels.py](./svm_channels.py), [svm_channels_check.py](./svm_channels_check.py), [svm_generation.py](./svm_channels.py), [svm_generation_check.py](./svm_channels_check.py)
- Aim: To find out whether there are time or channel selection, we perform time generation and channel selection performance.
- Results: It turns out the time point and channel selection is not doing well in classification analysis.

### Visualization Motion Animation

- File: [view.py](./view3d/view.py)
- Method: Plot the animation in plotly.

### SVM with Accumulate Feature

- File: [svm_accumulate.py](./svm_accumulate.py)
- Method: A trying to use the accumulated feature,
  where the features in 20 times are accumulated one-by-one.
  The aim is to reduce the noise during the 20 times.
- Result: It turns out the accumulation method is reducing the classification accuracy.
  It is another failed trying.
