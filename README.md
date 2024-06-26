# ASL Recognition with Convolutional Neural Network

It is project aimed at helping people with speaking limitations. It detects and identifies symbols of American sign language in real time using a pre trained CNN network and random forest classifier.</br>
The detector used is the BlazePalm detector from Google's mediapipe library which uses hand landmarks and their positions in 2d space to identify the symbols.</br>
1. Run the collect_data.py file which will take images from the systems webcam to use as input data.</br>
2. Run create_dataset.py and train_classifier.py respectively.</br>
3. To detect symbols run inference_classifier.py. </br>

