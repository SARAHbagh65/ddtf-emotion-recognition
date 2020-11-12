# ddtf-emotion-recognition
the aim of this project is recognition five emotional states using pre-trained CNN models and one effective connectivity measure from Electroencephalogram (EEG) signals. EEG signals were from the public databases of DEAP and MAHNOB-HCI. Processing steps were as below: 
1-Pre-processing EEG signals and remove noise and artifacts, to this end EEGs were monitored and pre-processed using EEGLAB toolbox, MATLAB software. 
2-Estimate dDTF effective connectivity measure. to this end, we used SIFT toolbox, MATLAB software. Each dDTF is a channel*channel*frequency components*time frames. We have two parameters of time and frequency, so we can represent brain flow information between multiple channels at different frequency bands (delta, theta, alpha, beta and gamma) at various time frames.    
3-Represent dDTF as color image to feed CNNs, fine-tune them and finally classify five emotional states.
4-Evaluate the recognition task. 
