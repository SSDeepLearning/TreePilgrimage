# p1s3
This Tree Pilgrimage project applies AI and computer vision techniques to identify the trees around us and bring out their beauty for the world to appreciate. 
The model distinguishes between two specific types of trees found in our neighborhood: the Weeping Willow, and the Peruvian Pepper tree. 
The pepper tree has a characteristic bark that distinguishes it, and of-course the red pepper fruits. 
On the other hand, the weeping willow has its own unique look with the gracefully arched branches that dangle down towards the ground, and create a cozy arbor in its shade

Here are the instructions to run this project on your local machine.
1. Download the whole p1s3 package  on your local machine.
2. Unzip it and go to https://drive.google.com/drive/folders/1LncqrpmWLHA6o2ThPlu2XVR58k4-xZxZ?usp=sharing
  a. Download the "p1s3_large_model_files" folder to your local machne.
  b. Open it and open the 'models' folder inside it.
  c. Copy the resnext101.pt file and paste it under the following path in your locally downloaded p1s3 package.
      p1s3/model/classifier/pretrained/models
  d. Now, copy the "best_check_point_pretrained_model" and "best_check_point_pretrained_model_dict" files from the G-drive path - p1s3_large_model_files/pretrained/ and paste it in the local under the following path.
      p1s3/model/classifier/pretrained
  e. Do the same for "best_check_point_pretrained_model" file under the p1s3_large_model_files/ui/ folder and paste in local path  p1s3/ui/
3. Now run the requirements.py file and make sure all the necessary dependencies are installed.
4. Now, Open the terminal (or other OS eqivalent) and locally go to  p1s3_large_model_files/ui/ path.
5. Run the following commands on your terminal.
    a. python setup.py build
    b. python setup.py install
    c. python setup.py bdist_wheel
7. Under the ui folder run the following command.
    strealit run image.py
7. A browser window will open up with the project UI. Playaround and enjoy.
  
