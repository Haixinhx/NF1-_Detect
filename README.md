# NF1-_Detect
The img model_data store the pretrained weights.
The train.py file is used to train the model. Before training, set the learning rate, backbone(can choose vgg and resnet),and other hyperparameter.
The logs file store the trained result of the model.
The before folder in dataset store the .json and original picture.
Use dataset.py to transfor the .json file to .png file.
The voc_dataset.py is used to split the dataset into training set and testing set, and create the VOC dataset.
The predict.py file is used to predict
