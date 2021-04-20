# Meta-Inductive Node Classification across Graphs 
We provide the implementaion of MI-GNN model.

The repository is organised as follows:
- dataset/: contains 5 benchmark datasets: Flickr, Yelp, Cuneiform, COX2 and DHFR. Note: all the five datasets are processed datasets and we can directly use them once we download all this repository. 
- sub_data.py: The code to create social sub-graph data "Sub_Flickr" or "Sub_Yelp", 800 small ego-networks,  extracted from an online image sharing social network and online user reviews network. Note that if we want to use "Sub_Flickr" or "Sub_Yelp" as the dataset and we don't have the 'data' directory(the processed data), we need to run this file before running main.py. And if we want "Sub_Flickr"， we need to type the string 'Flickr' as the input, if we want "Sub_Yelp"， we need to type the string 'Yelp' as the input.
- tudata.py, mydataset.py : Data preprocessing for Cuneiform, COX2 and DHFR.
- main.py: The main entrance of the model. You can change dataset name, l2 coefficient(note that for Sub_Flickr the l2_coef =1, and for other datasets, the l2_coef = 0.001), task_lr(for Cuneiform, Sub_Flickr, Sub_Yelp, 0.5, others, 0.005)in line 250--260.
- models/: contains our model. 
- learner_1.py: The two gnn layers of SGC, having neighboring aggregation, because we have not done that before.
- learner_2.py: The two gnn layers of SGC, no neighboring aggregation, because we have done that before.
- chemical.py, scaling_sgc.py, translation_sgc.py: About the scaling and shifting transformation
- earlystopping.py: The earlystopping function


## Requirements

  To install requirements:

    pip install -r requirements.txt

## Train and test

  To train and test the model in the paper(note that: (1)the following single file includes all the data split into training and testsing set; (2) we just need to run this single file and all things can be done, including data preprocessing, training and testing):
  
    python main.py
    





