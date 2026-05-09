* Load the training data from 'data/train/' and labels from 'data/train_labels.csv' using pandas and networkx.
* Replace `nx.read_gpickle` with the correct method to load the graph from a pickle file, which is `pickle.load`.
* Preprocess the graph data by extracting node features (centered & scaled x and y coordinates, normalized node degree) and building the adjacency matrix.
* Implement a Graph Convolutional Network (GCN) using torch_geometric with two GCN layers and a graph pooling step.
* Train the model using the Adam optimizer and cross-entropy loss with class weights.
* Validate the model on a stratified split of the training data and calculate the validation accuracy and macro-F1 score.
* Use the trained model to predict labels for the test graphs in 'data/test/'.
* Write the predictions to a submission.csv file in the required format.
* Ensure the model is run on CPU and completes within 60 minutes.
* Modify the data loading to handle PyTorch Geometric Data objects correctly.
* Fix the error caused by the index being out of bounds for the dimension.
* Use a batch size of 1 to avoid issues with different graph sizes.
* Use the `to` method to move the data to the correct device (CPU or GPU).
* Normalize the edge index to ensure it is within the bounds of the node features.