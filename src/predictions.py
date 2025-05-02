from dataset_loader import featuresTest , labelsTest
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Testing data
x_test = featuresTest
y_test = labelsTest

# X.test.shape()
# (20, 41, 153)
model = "D:\Semesters\FYP\Development\Final Year Project\models\version3.keras"
prediction = model.predict(x_test)

# convert probability into binary
binaryPred =  (prediction > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, binaryPred)
cm_labels = ['non-violence','violence']
disp=ConfusionMatrixDisplay(cm, display_labels=cm_labels)
disp.plot()
plt.show()