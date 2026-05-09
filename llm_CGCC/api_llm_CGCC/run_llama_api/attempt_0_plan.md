* Load the training data from 'data/train/' and labels from 'data/train_labels.csv' using the provided `nx_to_pyg` function.
* Define a Graph Neural Network (GNN) model using torch_geometric, with two GCN-style message passing layers and a graph pooling step.
* Train the model using the Adam optimizer and cross-entropy loss, with class weights to handle imbalanced classes.
* Validate the model on a stratified 30% subset of the training data and monitor the validation accuracy and macro-F1 score.
* Use the trained model to predict labels for the test graphs in 'data/test/'.
* Write the predictions to a submission.csv file in the required format.
* Ensure the model completes training within 60 minutes on CPU and uses only the allowed libraries.
* Set seeds for numpy, torch, and random to ensure reproducibility.