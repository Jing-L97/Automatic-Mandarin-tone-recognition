# Automatic-Mandarin-tone-recognition
This project compares the performance of Conv 2D CNN and Random Forest Classifier with controlled input features for tone classification of 4 Mandarin Chinese tones.

Dataset used was the Tone Perfect dataset from Michigan State University (https://tone.lib.msu.edu/). The training data used was a monosyllabic Mandarin Chinese dataset of 9,860 audio files. The neural network was trained on either male, female, or combined data, and for each dataset split, mel-frequency cepstral coefficients (MFCC) are extracted and fed as input features into the CNN. For the Random Classifier, the normalized pitch contours, creaky voice features(shimmer, jitter and HNR) are concatenated. 
For more information, please refer to the link here https://sites.google.com/view/jliucog/research?authuser=0
