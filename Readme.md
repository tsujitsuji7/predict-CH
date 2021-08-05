The results obtained using and modifying this code and hydration data set may be used in any publications provided that its use is explicitly acknowledged. A suitable reference is: Junji Hyodo, Kota Tsujikawa, Motoki Shiga, Yuji Okuyama, and Yoshihiro Yamazaki*, “Accelerated discovery of proton-conducting perovskite oxide by capturing physicochemical fundamentals of hydration”, ACS Energy Letters, 6(2021) 2985-2992. (https://doi.org/10.1021/acsenergylett.1c01239)



This repository contains

1. Experimental data and elemental information data 

2. Python code for prediction of proton concentration 


The details of folders and data files:

**data**

**experiment_data.csv**

Temperature-dependent experimental data of the proton concentration

**elements_data.csv**

The element information used to calculate descriptors.

Ionic radius indicates Shannon-Prewitt Effective Ionic Radius.
Since the ionic radii of 12 coordination number of Li+1, Mg+2, and Pr+3 are missing in our reference database, the values closest to the coordination number (Li+1 : Ⅷ, Mg+2 : Ⅷ, Pr+3 : Ⅸ) are described in this file.

**features.csv**

The list of descriptor names used for prediction

 

**code**

**predict_CH.py**

predict_CH.py is a Python code for predicts the proton concentration for the chemical composition specified on the console. Training dataset and prediction results  are output to a folder "03output/prediction".


The following is an example of prediction for composition specified in console. 

```python
python3 predict_CH.py


Enter the host element occupying the A-site

Ex:Ba

Ba

Enter the valence of the A-site host element

Ex:2

2

Enter the host element occupying the B-site

Ex:Zr

Zr

Enter the valence of the B-site host element

Ex:4

4

Enter the fraction that describes the composition of the B-site host element

Ex:0.8

0.8

Enter the dopant element occupying the B-site

Ex:Sc

Sc

Enter the valence of the B-dopant

Ex:3

3
```

**mylib.py**

defines functions for computing descriptors and combining and output dataset

 

**output**

is a folder to store combined training and test dataset, and prediction results.
After implementing "predict_CH.py", the following folders will be generated.

**prediction**

contains a folder that stores the experimental conditions used to predict proton concentration (VirtualExpCondition) and a folder that stores the descriptors calculated by ‘predict_CH.py’ in 02code (VirtualExperiment).

**VirtualExpCondition**

contains a csv file that describes the experimental conditions (measurement temperature, water vapor partial pressure, etc.) of the candidate composition.

**VirtualExperiment**

This folder contains training dataset and prediction data.
