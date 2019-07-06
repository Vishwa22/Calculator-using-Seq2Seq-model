# Arithmetic Calculations using seq2seq model

The goal of this task is to build a calculator performing arithmetic operations like addition, subtraction, multiplication using a sequence-to-sequence model.

### What is meant by sequene-to-sequence model?
Simple. A neural network which takes in a sequence, processes it and outputs a sequence that has a relevant significane to the input. 

Introduced for the first time in 2014 by Google, a sequence to sequence model aims to map a fixed length input with a fixed length output where the length of the input and output may differ.

An obvious example is language translation by machine. It takes a sentence in one language, processes it meaning understands its meaning and then outputs a sentence in another language which means the same as input.

### What kind of neural network can perform the above task?

Encoder-Decoder is a successful architecture for Seq2Seq tasks with different lengths of input and output sequences. 

![image](https://github.com/tensorflow/nmt/blob/master/nmt/g3doc/img/encdec.jpg?raw=true)

The main idea is to use two recurrent neural networks, where the first neural network *encodes* the input sequence into a *real-valued vector* and then the second neural network *decodes* this vector into the output sequence. While building the neural network, we will specify some particular characteristics of this architecture.

The real-valued vector mentioned above indicates a number vector which signifies the meaning of input to the model as extracted by the model.

#### Our task at hand:

Our task is to train a model which takes in "2+5" and outputs "7"...

So "2+5" is the input to the encoder and "7" is the output of the decoder.

Like every other tasks performed by neural networks, here we will need to preprocess the input in the format that can be understood by the netowrk. So will will create a vocabulory of symbols and associate with each symbol, some id value in vocabulory which can later be used to map to an embedding for the input sequence.


Our input comprises of integers ranging from *0 to 9* and any arithmetic operator * + -. These are called symbols.
So we will create a vocabulory which comprises of above symbols and also some extra symbols like "$", "^", "#".

* '^' symbol will be passed to the network to indicate the beginning of the decoding procedure.
* '$' symbol will be used to indicate the end of a string, both for input and output sequences.
* '#' symbol will be used as a padding character to make lengths of all strings equal within one training batch.


Vocabulory Dictionary, entry format <symbol:id> 
{'#': 0, '^': 1, '$': 2, '*': 3, '+': 4, '-': 5, '1': 6, '2': 7, '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14, '0': 15}

PADDING:

Entire purpose of padding is to create batches of input sequences with equal sizes.

Keep in mind that the original characters of the sequence and the end symbol is considered as valid part of the input and represents the *actual length* of the input.

We will have a predefined *padding_len* and all the input sequences and ground-truth output sequences will be either cut-off or padded with # to match the length equal to *padding_len*.

So remember that for *embedding layer* in our neural network, we will need ids to lookup into the embedding matrix. So the ids corresponding to elements in our sequences can be found from the vocabulory defined. 

So input '214-369' will look like 

[7, 6, 9, 5, 8, 11, 14, 2]

And output '-155' will look like [5, 6, 10, 10, 2, 0, 0, 0]
for our neural network.

### Encoder architecture

The first RNN of our current seq2seq architecture is called an encoder and serves for encoding an input sequence to a real-valued vector. Input of this RNN is an *embedded input batch*.

The final encoder state will be passed to the second RNN (decoder).

### Decoder architecture

Decoder takes an input the final state of the encoder as the source information. And it performs two separate tasks.
*Training* and *Inference*

# <img src="https://github.com/hse-aml/natural-language-processing/blob/master/week4/encoder-decoder-pic.png?raw=true" style="width: 100px">

Training:

Here, along wth the final encoder state, the decoder is feeded in as input the actual ground truth sequence (as shown in the figure above). This is the phase of learning.

Inference:

Here, decoder gets only the final state of encoder. And then as it receives the 'beginning of decoding' symbol, it generates first symbol as output and then feeds it at next time step its own generated output symbol as input(as shown in the figure above)

Also note in the figure above, the transparent parts. They are simply ignored. In decoder, it is masked out in the loss computation. In encoder, the green state is considered as final and passed to the decoder.

There are many ways to perform decoding. Decoding methods include greedy, sampling, and beam-search decoding. Here, we will discuss the greedy decoding strategy.

The idea is simple.

1. We encode the source sentence in same way during training as well as inference. Encoder's final state is defined as  *final_encoder_state*, and is used to initialize the decoder.
2. The decoding process is started as soon as the decoder receives a starting symbol **^**.
3. For each timestep on the decoder side, we treat the RNN's output as a set of logits. We choose the most likely symbo, the id associated with the maximum logit value, as the emitted symbol (this is the "greedy" behavior). For example in Figure 3, the word **1** has the highest probability in the first decoding step. We then feed this symbo as input to the next timestep.
4. The process continues until the end-of-sentence marker **$** is produced as an output symbol.

Step 3 is what makes inference different from training. Instead of always feeding the correct target symbols as an input, inference uses symbol generated by the decoder.

The IPython Notebook shared here shows the seq2seq model training for addition, subtraction and multiplication of integers ranging from 0 to 999. And *mean absolute error* is computed at the end of each epoch. 































