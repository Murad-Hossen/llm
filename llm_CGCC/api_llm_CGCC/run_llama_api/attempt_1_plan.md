* Import the `pickle` module to fix the `NameError`.
* Update the `DataLoader` import to use `torch.utils.data.DataLoader` instead of `torch_geometric.data.DataLoader` to fix the deprecation warning.
* Ensure the model is correctly handling the graph data and labels.
* Validate the model on a stratified 30% subset of the training data and monitor the validation accuracy and macro-F1 score.
* Use the trained model to predict labels for the test graphs in 'data/test/'.
* Write the predictions to a submission.csv file in the required format.