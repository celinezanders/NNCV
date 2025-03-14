Overview
Thank you for participating in our image segmentation challenge! To submit your model for evaluation, please make sure that your submission contains the following three files:
----------------------------------------------------------------------------------------
1. model.py: 
	This file should contain a Python class called 'Model' that defines your model architecture. The Model class should inherit from torch.nn.Module and implement a forward method that takes an input tensor and produces a tensor of the same shape representing the model's prediction.

2. model.pth: 
	This file contains the weights for your trained model. Please ensure that your model can be loaded using the following code:
	model = Model()
	model.load_state_dict(torch.load(PATH))

3. preprocess.py: 
	This file should define two functions that will be used in the evaluation program:

	preprocess(img): 
		This function receives the original image as a PIL.Image object in its original shape. Your preprocess function should process the image to the appropriate format expected by your model. Please note that you should not include any code to send the image to a GPU, as this will be done automatically in the evaluation script.

	postprocess(prediction, shape): 
		This function receives the output directly from your model as a tensor, as well as the shape of the original image. Your postprocess function should process the output to the original size of the image, and output an np.array of shape [X, Y, n], where X and Y are the original image sizes, and n is the number of class labels per pixel.

	You may add additional functions to preprocess.py, but they should only be used as helper functions for your preprocess and postprocess functions.


----------------------------------------------------------------------------------------
Submission Instructions
To submit your solution, please zip the three files (model.py, model.pth, and preprocess.py) into a single file named submission.zip. 

Goodluck submitting! We look forward seeing your results!