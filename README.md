# siamese_network

This code was developed in order to test the Siamese Network for detecting ischemic stroke in CT scan based on brain symmetry. For this purpose, two architectures were developed, SiameseNet (Siamese Network) and SimResNet-18 (Siamese Network + ResNet-18). As baseline architecture the ResNet-50 was used.

The image processing folder contains the files:
- skull_stripping.py: file that receives the CT brain scans in NIfTI format (.nii), removes the non-brain tissue and converts the 3D data to 2D, saving the images in .png.
- head_tilt_correction.py: receives the images (.png) and corrects the head tilt. The images resulting from this process constitute the final images used for training and testing.

The stratified 5-fold cross validation method was used to train the models. The selection of the samples is done according to the file Asymmetry2_v2.xlsx, where the slices are grouped by folders (in a total of 5).
