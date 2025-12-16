# Pothole-Detection-Using-YOLOv8
Requirements and Installations
  pip install -r requirements.txt
Training of YOLO Model on Custom Dataset
Download the pothole detection dataset from roboflow.com.
Training YOLOv8 Model with Custom Dataset using Colab
Now go to the ‘Runtime‘ menu, select ‘Change runtime type‘, choose ‘T4 GPU‘ for the Hardware accelerator, and save it.
Let’s check whether the GPU is running perfectly or not using the following command:
  !nvidia-smi
Next, install ultralytics on your colab workspace using the following command:
  !pip install ultralytics
Now open your Google Drive and navigate to ‘My Drive.’ Now create a folder named ‘Datasets‘ under ‘My Drive’ and inside the ‘Datasets’ folder create one more folder ‘Pothole.’
Let’s open the unzipped dataset folder, select all items present there, and drop them into the ‘Pothole’ folder on Google Drive. It may take a while so wait until it is finished.
Now open the ‘data.yaml‘ file in the text editor and modify the path variable to: “../drive/MyDrive/Datasets/Pothole”.The final ‘data.yaml‘ file will look like the following:
      path:../drive/MyDrive/DataSets/Pothole
      train: ../train/images
      val: ..valid/images
      nc: 1
      names: ['pothole']
Now, let’s go back to our Google Colab dashboard. You need to mount your Google Drive with the Colab. Insert the following command in a new cell and run it:
   from google.colab import drive
   drive.mount('/content/drive')
Now we will start training our YOLO model with our pothole detection dataset. Again, create a new cell, insert the command below, and run it.
   !yolo task=detect mode=train model=yolov8l.pt data=../content/drive/MyDrive/Datasets/Pothole/data.yaml epochs=100 imgsz=640

The whole training can take around 1 – 2 hours even more to complete.
After the completion of the training, go to the ‘Files‘ section in your Colab dashboard and navigate through these folders: ‘runs’ -> ‘detect’ -> ‘train’ -> ‘weights’. Inside the ‘weights‘ folder, you will see ‘best.pt‘ and ‘last.pt,‘ these two files. Download ‘best.pt‘ from there.
Place the Downloaded YOLO Model
In the previous section, we trained our YOLO model with a custom pothole detection dataset and downloaded a file named ‘best.pt.’ Now place this file inside all folder where best.pt3 is located.
