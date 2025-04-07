'''
This script will be used to label new batches of data with the help of the Batch Predictive Assistant
(BPA). This BPA has been trained on specific data that reflects the upcoming unlabeled data based on
ResNet-50 embeddings.

1. Check /batch dir.


2. Not Empty: Search batch.json for the batch that the images belong to. Use corresponding 
["model_name"] as the BPA for the remaining images in /batch. Label data.


3. When /batch eventually becomes empty while labeling: extract 1,000 images (new batch) from /frames


4. Create embeddings for new batch and run top_k query, creating a training set


5. Train new batch PA


6. Continue labeling with new BPA, uploading processed images' embeddings to Pinecone simultaneously.
{return to step 3 when /batch empty}
'''