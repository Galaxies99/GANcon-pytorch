# Data Preprocessing

If you are using datasets that have not been divided into training set, validation set and testing set yet, you can use our tools to preprocess the dataset by simply following the steps below:

1. Put the data into the `data` folder.
2. Create two folders named `feature` and `label`, and put the features and labels into the corresponding folder respectively.
   - Notice the features should have an extension of `.npy.gz` (zipped file of the numpy `npy` file), and the labels should have an extension of `.npy` (the numpy `.npy` file).
   - Notice the filename of feature and its label should be the same to ensure one-to-one correspondence. For example: feature file `ABCDEF.npy.gz` in the `feature` folder corresponds to label file `ABCDEF.npy` in the `label` folder.
   - Notice if there is a `reference.json` file in the `data` folder, the preprocessing procedure will divide the dataset according to `reference.json` file. So if you are using a new dataset, make sure to delete the old `reference.json` file. If you want to divide the data exactly the same as the last time, just keep `reference.json` and continue executing the following steps.
3. Execute the following commands

   ```bash
   python utils/preprocessing.py
   ```

4. After finishing executing, the dataset is divided into the training set, validation set and testing set into `train`, `val` and `test` folders respectively, and the correspondences is stored in the `reference.json` file.
