#[note] : This pre-processing and training the model has been done in Google Collab.
# The image data set is in my [naresh] Google Drive , It's made public.
#data collected from kaggle and some my custom pictures.


import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dropout , Flatten , Dense , Input , Model , AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array , load_img
from tensorflow.keras.utils import to_categorical


DIR = r"drive/MyDrive/Colab Notebooks/maskImages" #Loaction of masked and unmasked images.
CATEGORIES = ["with_mask", "without_mask"]
data = [] #Image array list
labels = [] #With mask / without mask.

#Images are loaded from the respective diretories. 
print("Progres 10% : Loading the image dataset")

for category in CATEGORIES:
    path = os.path.join(DIR , category)
    for img in os.listdir(path):
    	img_path = os.path.join(path , img)
    	image = load_img(img_path , target_size = (224, 224)) #224 is dimention --> To reduce the process load
    	image = img_to_array(image) #converting the img to array values
    	image = preprocess_input(image) #Pre-processed to use in MobileNet model
        
        #Now appending the image array into data list.
    	data.append(image)
    	labels.append(category)

# Coverting categories to categorical numerical value.
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#Converting data and label lists to numpy array for deep learning.
data = np.array(data, dtype = "float32")
labels = np.array(labels)

#Splitting the data into test(20%) and train. 
(trainX, testX, trainY, testY) = train_test_split(data, labels , test_size = 0.20, stratify = labels, random_state = 30)


# Data Augumentation -- > Generating multiple data using exixsting data with slight modification.
data_aug = ImageDataGenerator(
	rotation_range = 20,
	zoom_range = 0.15,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.15,
	horizontal_flip = True,
	fill_mode = "nearest")

# load the MobileNetV2 network and the head FC (top) layer sets are kept off.
baseModel = MobileNetV2(weights = "imagenet", include_top = False , input_tensor = Input(shape = (224, 224, 3)))

# Constructing the head of the model that will be placed on top of the the Base model.
headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7, 7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(128 , activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2 , activation = "softmax")(headModel)

# Place the head FC model on top of the base model whcih will be trained.
model = Model(inputs=baseModel.input , outputs=headModel)

# loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False #freezed


# Initializing the initial learning rate, No of epochs and Batch Size to train.
INIT_LR = 1e-4 #0.001
EPOCHS = 20
BS = 32 #Batch Size


# Compiling the model.
print("Progress 30% : Compiling the model...")
opt = Adam(lr = INIT_LR, decay = INIT_LR/EPOCHS)
model.compile(loss = "binary_crossentropy" , optimizer = opt , metrics = ["accuracy"])

# train the head of the network
print("Progress : Training Head of the network...")
Head_Model = model.fit(
	data_aug.flow(trainX, trainY, batch_size = BS),
	steps_per_epoch = len(trainX)//BS,
	validation_data = (testX, testY),
	validation_steps = len(testX)//BS,
	epochs = EPOCHS)

# Predicting the testing dataset with the batch size 32
print("Progress : Evaluating the Network")
predIdxs = model.predict(testX , batch_size = BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs , axis = 1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis = 1) , predIdxs , target_names = lb.classes_))

# serialize the model to disk
print("Progress 90% : Writing the Mask Detector Model")
model.save("mask_trained.model", save_format = "h5")

print("Progress 98% : Successfully trained the images.")
print("Progress 99%: Displaying the Accuracy and Loss in Training Phase")


# PLotting the loss and accuracy of training phase using matplotlib.pyplot
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), Head_Model.history["loss"] , label = "train_loss")
plt.plot(np.arange(0, N), Head_Model.history["val_loss"] , label = "val_loss")
plt.plot(np.arange(0, N), Head_Model.history["accuracy"] , label = "train_acc")
plt.plot(np.arange(0, N), Head_Model.history["val_accuracy"] , label = "val_acc")
plt.title("Loss and Accuracy in Training Phase.")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("lossANDaccuracy.png")

print("Progress 100% : Training done , now time to deploy the application.")
