Supplemental code and data to the manuscript of Cao, Yaqiang, et al. "Hi-TrAC reveals fractal nesting of super-enhancers" bioRxiv(2022).      

Briefly, the data used are located in the data directory and compiled from ENCODE ChIP-seq data for active sub-TADs defined by Hi-TrAC data. The data are pre-separated into training, testing, and validation (8:1:1) randomly for reproducible analysis of the result in the manuscript. You can do the separation from all.txt. The code of deepModel.py was used to train the model, and generate the figures of Figure S1C, Figure 3D ROC curves, and Figure 3E. 

Required python packages:
```
pandas
numpy 
tqdm 
scikit-learn==0.23.2
tensorflow==1.10.0 
```

If the model.h5 has been generated, please do not run the whole code again, as the training will continue with the saved model and will lead to different feature importances. Just delete the model.h5 and re-run the code.  
