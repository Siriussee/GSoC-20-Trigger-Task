# Solution of GSoC2020 Trigger Task


## Dataset Desc

Dataset: https://www.dropbox.com/s/c1pzdacnzhvi6pm/histos_tba.20.npz?dl=0

the dataset is an npz file containing 2 numpy arrays ‘variables’ and ‘parameters’. 

### variables

The first array ‘variables’ contains 87 features, 84 of which serve as input variables and 3 other serving as road variables (pattern straightness, zone, median theta, respectively).

**Note:** in the dataset, a unit in phi is 1/60 = 0.01667 deg, a unit in theta is approx 0.285 deg.

| **Column range**                      | 0-11           | 12-23            | 24-35         | 36-47     | 48-59       | 60-71             | 72-83 | 84     | 85   | 86   |
| ------------------------------------- | -------------- | ---------------- | ------------- | --------- | ----------- | ----------------- | ----- | ------ | ---- | ---- |
| **Particle**   **hit**   **features** | Phi Coordinate | Theta coordinate | Bending angle | Time info | Ring number | Front/   rear hit | Mask  | X_road |      |      |

### parameters

| Column     | 0                                      | 1         | 2         |
| ---------- | -------------------------------------- | --------- | --------- |
| Parameters | q/pt   charge over transverse momentum | Phi angle | Eta angle |

For our purpose, we are only interested in the muon hits detected at the CSC chambers. Therefore, feel free to drop the data columns belonging to the other stations from the RPCs and GEMs.

![1583217022295](C:\Users\Siriu\AppData\Roaming\Typora\typora-user-images\1583217022295.png)

## TASK 1:  MUON MOMENTUM INFERENCE USING DEEP NEURAL NETWORKS

- In the first task, we want you to perform classification where muon momenta are clustered into 4 ranges of absolute $p_t$ ranges: 0-10 GeV, 10-30 GeV, 30-100 GeV and >100 GeV. Develop a **Fully-Connected Network** using a framework of your choice and evaluate its ability in classifying muon momentum ranges using the raw data muon data that we provided. 
- Then, investigate any improvements offered by convolutional layers by implementing a **Convolutional Neural Network (CNN)** separately. 
- Having trained these two models, **indicate the set of hyperparameters** that you have tuned prior to obtaining the optimal results and provide a visualization of the loss and metrics as a function of the number of epochs. Finally, show the model’s ability to generalize to new datasets using the best suitable performance measurements.
- Next, try to **regress on the pt (or 1/pt) directly**

## TASK 2:  IMAGE-BASED CLASSIFICATION OF MUON MOMENTA

- Similar to Task 1, this task requires you to **implement a FCN and a CNN**. Only this time, you have to **project the hits provided by the raw data into images**. There are different ways to approach muon trajectories as images, so we rely on your creativity to come up with efficient solutions to deal with such sparse data. 
- As previously asked, you should **report the set of hyperparameters** that were tried during training and **present niche visualizations of the training process** and the model’s performance when tested on new data.
- Report the noted differences between the model performance resulting from the image-based approach and that of the model trained on raw data.

## TASK 3:  JET AS GRAPHS

### DATASET DESCRIPTION FOR TASK 3

For Task 3, you will use ParticleNet’s data for Quark/Gluon jet classification available [here](https://zenodo.org/record/3164691#.Xk1VwS2B1QI). 

In particle physics, a jet is associated with a spray of particles spread across the detector, and we are interested in identifying the elementary particle that initiated the decays forming this jet.  The dataset contains 2 million jet events detected at CMS with a transverse momentum range 

![img](file:///C:/Users/Siriu/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

 and rapidity |y|<1.7. These jets were constructed with the anti-kt algorithm with R=0.4. The dataset contains information about the particles in each jet. It is split into 20 Npz files[[1\]](#_ftn1) each of which contains 2 Numpy arrays: X of shape (100000,M,4) where 100,000 is the number of jets per file,  M is the max multiplicity of the jets and 4 is the number of features per point cloud (particle). The four features are the particle’s pt, rapidity, azimuthal angle and pdgid. The second Numpy file y and contains the class label for each jet: label 0 is assigned for gluons and label 1 for quarks. Please read the data description on the provided link for further details.

- Implement a **Message Passing Neural Network (MPNN)** with a framework of your choice (PyTorch Geometric, Deep Graph Library, GraphNets) to **classify jets as being quarks or gluons**. 
- Provide a description on what considerations you have taken to project this point-cloud dataset to a set of inter-connected nodes and edges (What graph topology have you used? Why do you believe it is efficient?)

## OPTIONAL

Can we inspire from the approach in Task 3 to implement graph approaches on the muon data used in tasks 1 and 2? Discuss any potential **graphical approach that come to your mind and mention the limitations**, if any. Feel free to submit a **pseudo-code** for such a process. 



> To load a npz file ‘example.npz’:  load the file using numpy: *data= np.load(‘example.npz’).* Check its content by running *data.files* which outputs the names of the stored numpy arrays (In our case: [‘variables’,‘parameters’] )