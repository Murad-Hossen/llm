* Modify the `GraphDataset` class to return a batch of graphs and labels that can be handled by the `DataLoader`.
* Update the `DataLoader` to use a custom collate function that can handle batches of `Data` objects.
* Validate the model on a stratified 30% subset of the training data and monitor the validation accuracy and macro-F1 score.
* Use the trained model to predict labels for the test graphs in 'data/test/'.
* Write the predictions to a submission.csv file in the required format.