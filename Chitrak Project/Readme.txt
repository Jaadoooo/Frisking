Frisking Detectron folder contains the Detectron and YOLO models for predicting Frisking. Run the Main.py script and ensure the correct video path in the 1Detectron.py file and 4Final.py file.

The output_video.mp4 file will be created which would the be final output.

1Detectron.py file and 4Final.py file may take several minutes to load, so be patient. 


-------------------x------------------------------x------------------------x-----------------------x-----------


Frisking RNN folder contains Detectron model to extract keypoints and then train the RNN model with it. 

First run the dataProcessing.py file to bring your COCO data into a suitable format on which we can train the RNN model. Store the images in "Images_Folder". Pass the correct path of the input JSON file

Next run the model.py file and set the correct path of the recently obtained data.json file. After training the wights of the model will be saved 

Run the 1Predict.py and 2Predict.py files to obtain the prediction whether the video is Frisked or not.



