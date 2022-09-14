----------------------------------------------------------------------------------------------------------------
How to run the code:

1. Replace the "484_test_file.txt" with the new test dataset 
and pay attention to keep the same name, which is "484_test_file.txt", to be the file name of the new test dataset text file.

2. Open the python file "Yuxi_code.py". Run the file.

----------------------------------------------------------------------------------------------------------------
Some explanations for other file in the folder:
(1) "484_train_file.txt" file contain 15000 sentiments and reviews from the train dataset given on miner.
(2) "predict_text.txt" is an example of the final result run by my python file "Yuxi_code.py". 
      Every time of running will replace the previous output in this file and output new result to this file. 

----------------------------------------------------------------------------------------------------------------
where your k-NN implementation is:
The code of line 30-73 in the python file "Yuxi_code.py" is my k-NN implementation.

----------------------------------------------------------------------------------------------------------------
Some other information:
(1) There are comments for every part of my code so it is easy to understand what I want to do in every part.
(2) 
Code running time:
The data pocessing only need about 1 or 2 minutes to run. 
Using my k-NN model to predict 15000 reviews in the train dataset (calculate the distance of every review from 15000 reviews in the train dataset) need to take about 7 hours on my computer

