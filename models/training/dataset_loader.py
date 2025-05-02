from data_preprocessing import extract_pose_features


# Run two times one for test and one for train
datasetPath = "D:\Semesters\FYP\Development\Final Year Project\data"
featuresTest , labelsTest = extract_pose_features(datasetPath, "test")

featuresTrain , labelsTrain = extract_pose_features(datasetPath, "train")
# featuresTrain.shape()
# (3143, 41, 153)


# Dataset folder structure should be 
# dataset
# |___
# |    test
# |    |___
# |    |   violence
# |    |___non-violence
# |    |
# |    train
# |    |___
# |    |   violence
# |    |___non-violence