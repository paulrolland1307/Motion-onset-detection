For the execution, compile the file MotionDetection.cpp with helpers.hpp, timing.h and filters.h, make sure the required files exist (described in the following) and execute the generated runnable file. This trains the models for motion and grasping detection, and apply them to some trial of about 10s. The script prints the starts and ends of motion and grasping, as well as the computation and prediction times required.


In order for the main script to work, some data files are required in a folder called Resources : 
- training_Motion.txt : N x 36 matrix where each row contains all the features for each muscle (in the order : Root Mean Square, Waveform Length, Zero Crossing) for some time window as well as the correponding label (ie motion (1) or no motion (0))
- testing_Motion.txt : has the exact same structure as training_Motion.txt
- training_Grasp.txt : N x 36 matrix where each row contains all the features for each muscle (in the order : Root Mean Square, Waveform Length, Zero Crossing) for some time window as well as the correponding label (ie grasping (1) or no grasping (0))
- testing_Motion.txt : has the exact same structure as training_Grasp.txt
- testingRaw.txt : 12 x T matrix where each row contains the EMG signal sent by a muscle with frequency 1500 Hz (can be modified in the code).
- A_low.txt and B_low.txt : contain the coefficient for the low-pass filter
- MVC.txt : contain the Maximum Voluntary Contraction for all the muscles


The project is composed of 6 files :
- MotionDetection.cpp : contains the main function, which simply makes calls to three functions : computeFilterCoeff() which computes the coefficients required by the filter, trainAllModels() which trains the two SVM algorithms for both motion and grasping detection, and startPredictions() which applies the trained models to some trial when only given the raw data (ie filter the signal one time window at a time, compute the features and predict the corresponding label).
- helpers.cpp and helpers.cpp : contain the helper functions that are used from the main function.
-timing.cpp and timing.h : helper functions for running time estimation.
- filters.h : helper functions for signal filtering.
