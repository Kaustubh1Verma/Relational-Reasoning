# Relational-Reasoning
Keras implementation of Relational Reasoning to solve Visual QA problem

Refer
"A simple neural network module for relational reasoning"
Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap
https://arxiv.org/pdf/1706.01427.pdf

Please refer to the report for the details [Report](https://github.com/Kaustubh1Verma/Relational-Reasoning/blob/master/LearnOrBurn_Hackathon_Writeup.pdf)

## CLEVR dataset (https://cs.stanford.edu/people/jcjohns/clevr/).
Download the Cleverv1.0 dataset and put it in the home folder as the other codes.

## Architecture
Images are processed using a CNN, while the questions are processed using an LSTM.  These tensors are then decomposed into objects and fed as input into the RN module.
![Alt text](CLEVR.png?raw=true "Title")


## Experiments
Training the network with different no of samples(questions) such 200000/60000,etc. although the training part is not completed.Would take large no of epochs.

### Sort of CLVER implementation
This is the implementation on sort of clver dataset which is being generated inside the code. Needs no extra file or data other than notebook itself.
Good results are obtained on relational and non-relational questions.

