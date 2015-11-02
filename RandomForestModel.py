import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import csv as csv

# ______________________________________________READ DATA________________________________________________

train_df = pd.read_csv('Data/train.csv', header=0)
test_df = pd.read_csv('Data/test.csv', header=0)

# ______________________________________________CLEAN DATA______________________________________________

# Calculate the variance of each pixel in the training data
variances = [ np.var(train_df['pixel' + str(i)].values) for i in range(784) ]

# Remove all pixels with zero variance
train_df = train_df.drop(["pixel" + str(i) for i in range(len(variances)) if variances[i] == 0], axis=1)

# Apply the same transformation to the testing data
test_df = test_df.drop(["pixel" + str(i) for i in range(len(variances)) if variances[i] == 0], axis=1)

# ______________________________________________TRAIN CLASSIFIER______________________________________


clf = ExtraTreesClassifier(100)
train_data = train_df.values
X, Y = train_data[0::, 1::], train_data[0::, 0]
clf.fit(X, Y)

# ______________________________________________PREDICT_________________________________________________

output = clf.predict(test_df)

# __________________________________________WRITE TO OUTPUT FILE________________________________________

predictions_file = open("makeSubmission.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(list(range(1,1+len(output))),output))
predictions_file.close()
