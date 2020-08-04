# RecSeats

**Author of the code:** Anon.

This repository contains the source code and the results of "RecSeats: A Hybrid Convolutional Neural Network Choice Model for Seat Recommendations at Reserved Seating Venues".

The article propose a hybrid framework combining choice modelling and convolutional networks for the recommendation of locational choice data (RecSeats). 
We build a discrete choice model based on an individual-choice-level feature space, and then combine this model with a deep learning stage which not only improves accuracy but also makes it robust to the amount of data available per customer. 

For reasons of data anonymization, only the files concerning Locational Choice Experiment are available here for reproducibility.

## Code structure

### Data

This folder contains all the data used in the project. To date, it only one available folder containing locational choice experiment data is available online.
Details about those datasets can be found here : https://seatmaplab.com/public/locationalchoicedatasets/

These are csv files where each line represents an available seat in a given configuration, which allows the input to be reconstructed, with a label indicating whether or not it has been chosen.
To avoid having to rebuild the input at each run, two subfolders *numpy* and *dataloader* allow to save them, respectively for our individual-level and deep models.

Each experiment also has a JSON file containing all these parameters (filenames, room size, padding, etc.).


### Save

This folder contains all the saved outputs related to the models:

* save/accuracies: store the metrics results for the models.
* save/best_models: store the pytorch trained models (for CNN/CDNN/Hybrid).

### Src

This folder contains all the source code.

* src/deep: contains code for torch tensor preprocessing and deep models training and evaluation
* src/models: contains all the tested models (Individual-levels, CNN, CDNN, Hybrid), and a folder containing JSON files for the hyperparameters of each of them.
* src/preprocessing: contains code for input generation from csv file and feature computation for individual-level part.
* src/visualisation: contains visualization functions.

Each models is turned in one of the jupyter notebook.


## Models Hyperparameters

Here is a table for each model of the selected hyperparameters. To retrieve the results obtained, simply change the values for the one in the corresponding table, in the parameters file in *src/model/parameters*

### Individual-level models

#### Logistic Regression

|        Dataset        |penalty|  solver  | max_iter | POS features | PS features | R2 feature | R3 feature |
|-----------------------|:-----:|:--------:|:--------:|:------------:|:-----------:|:----------:|:----------:|
| E4-Concert-Singles.FC | None  |newton-cg |   300    |     True     |     True    |   True     |   True     |
| E2-Movie-Singles.FC   | None  |newton-cg |   300    |     True     |     True    |   False    |   False    | 
| Concert Hall data     | None  |newton-cg |   300    |     True     |     True    |   True     |   True     | 


#### Support Vector Machines

|        Dataset        |   C   |  kernel  | max_iter | POS features | PS features | R2 feature | R3 feature |
|-----------------------|:-----:|:--------:|:--------:|:------------:|:-----------:|:----------:|:----------:|
| E4-Concert-Singles.FC |  100  |  linear  |   2000   |     True     |     True    |   True     |   True     |
| E2-Movie-Singles.FC   |   50  |  rbf     |   2000   |     True     |     True    |   True     |   True     | 
| Concert Hall data     |  100  |  linear  |   2000   |     True     |     True    |   True     |   True     | 



#### Gradient Boosted Trees

|        Dataset        |  lr   |n_estimators|max_depth |min_samples_split|min_samples_leaf|max_features|
|-----------------------|:-----:|:----------:|:--------:|:---------------:|:--------------:|:----------:|
| E4-Concert-Singles.FC |  0.1  |     200    |     2    |         2       |        1       |      2     |
| E2-Movie-Singles.FC   |  0.1  |     200    |     2    |         2       |        1       |      2     | 
| Concert Hall data     |  0.1  |     200    |     3    |         2       |        1       |      2     | 


|        Dataset        | POS features | PS features | R2 feature | R3 feature |
|-----------------------|:------------:|:-----------:|:----------:|:----------:|
| E4-Concert-Singles.FC |     True     |     True    |   True     |   True     |
| E2-Movie-Singles.FC   |     True     |     True    |   False    |   True     | 
| Concert Hall data     |     True     |     True    |   True     |   True     | 


#### Random Forests


|        Dataset        |n_estimators|max_depth |min_samples_split|min_samples_leaf|max_features|
|-----------------------|:----------:|:--------:|:---------------:|:--------------:|:----------:|
| E4-Concert-Singles.FC |     200    |     2    |         2       |        1       |      2     |
| E2-Movie-Singles.FC   |     200    |     5    |         3       |        2       |      2     | 
| Concert Hall data     |     200    |     5    |         2       |        1       |      2     | 


|        Dataset        | POS features | PS features | R2 feature | R3 feature |
|-----------------------|:------------:|:-----------:|:----------:|:----------:|
| E4-Concert-Singles.FC |     True     |     True    |   True     |   True     |
| E2-Movie-Singles.FC   |     True     |     True    |   False    |   False    | 
| Concert Hall data     |     True     |     True    |   True     |   True     | 


### Deep Models

#### CNN

|        Dataset        |nb_conv_layers|batch_size |   lr   |
|-----------------------|:------------:|:---------:|:------:|
| E4-Concert-Singles.FC |       3      |     32    |  1e-4  |
| E2-Movie-Singles.FC   |       3      |     32    |  1e-4  | 
| Concert Hall data     |       4      |     32    |  1e-4  | 


#### CDNN

|        Dataset        | nb_channels |batch_size |   lr   |
|-----------------------|:-----------:|:---------:|:------:|
| E4-Concert-Singles.FC |       3     |     32    |  5e-3  |
| E2-Movie-Singles.FC   |       3     |     32    |  5e-3  | 
| Concert Hall data     |       4     |     32    |  5e-3  | 


### Hybrid Models 

The same hyperparameters are kept for both models combined.

|        Dataset        |  combination  |  lr  | momentum | weight decay | alpha_init |
|-----------------------|:-------------:|:----:|:--------:|:------------:|:----------:|
| E4-Concert-Singles.FC |    GBT+CNN    | 1e-3 |    0.9   |     0.8      |     0.5    |
| E2-Movie-Singles.FC   |     RF+CNN    | 1e-2 |    0.9   |     0.5      |     0.5    | 
| Concert Hall data     |   GBT+CDNN    | 3e-2 |   0.99   |     0.7      |     0.5    |




