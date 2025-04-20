Question 1 (5 Marks) Build an RNN based seq2seq model, which contains the following layers: (i) input layer for character embeddings (ii) one encoder RNN, which sequentially encodes the input character, sequence (Latin) (iii) one decoder RNN which takes the last state of the encoder as input and produces one output character at a time (Devanagari). your local machine and push everything ed reflect how the code has evolved 2 of 2 The code should be flexible such that the dimension of the input character embeddings, the hidden states of the encoders and decoders, the cell (RNN, LSTM, GRU) and the number of layers in the encoder and decoder can be changed. (a) What is the total number of computations done by your network? (assume that the input embedding size is mmm, encoder and decoder have 1 layer each, the hidden cell state is k for both the encoder and decoder, the length of the input and output sequence is the same, i.e., T, the size of the vocabulary is the same for the source and target language, i.e., V) (b) What is the total number of parameters in your network? (assume that the input embedding size is mmm, encoder and decoder have 1 layer each, the hidden cell state is k for both the encoder and decoder and the length of the input and output sequence is the same, i.e., T, the size of the vocabulary is the same for the source and target language, i.e., V) c) Use the best model from your sweep and report the accuracy on the test set and Provide sample inputs from the test data and predictions made by your best model.

# Building an RNN

Given: Input embedding size (m)=256 Hidden state size (encoder + decoder) ( k )=256 Sequence length (input and output) ( T ) = 20(say) Vocabulary size (source and target) ( V ) = 60(say) Using 1-layer LSTM for both encoder and decoder No attention Inference is greedy decoding Total computations = T×[8k(m+k)+kV] substituting above values = 20×[8×256(256+256)+256×60] = 21278720 Therefore, Total number of computation done by the network = 21.28 million operations

Total Parameters = 2Vm+8k(m+k+1)+kV+V substituting above values = 2×60×256+8×256(256+256+1)+256×60+60 = 1100464 Therefore, total number of parameters in your network = 1.10 million parameters




Question 2: (5 Marks) Your task is to finetune the GPT2 model to generate lyrics for English songs. You can refer to (https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and huggingface-f3acb35bc86a) and follow the steps there. This blog shows how to fine-tune the GPT2 model to generate headlines for financial articles. Instead of headlines, you will a use lyrics so you may find the following datasets useful for training: https://data.world/datasets/lyrics https://www.kaggle.com/paultimothymooney/poetry Guidelines: We will check for coding style, clarity in using functions, and a README file with clear instructions on training and evaluating the model. We will also run a plagiarism check to ensure that the code is not copied (0 marks in the assignment if we find that the code is plagiarised by other students). We will check the number of commits made by you and then give marks accordingly. you will be expected to answer questions, explain the code, etc. You also need to provide a link to your github code as shown below. Follow good softwanm engineering practices. Please do not write all code on your local machine and push every to github on the last day. The commits in github should reflect how the code has eva during the course of the assignment.


# 🎶 GPT-2 Song Lyrics Generation with Fine-Tuning

This project fine-tunes the *GPT-2* model on a custom dataset of song lyrics and uses it to generate new song lyrics based on a prompt. The model is trained using the *Hugging Face Transformers* library and supports fast generation of lyrics.

## 🚀 Features

- *Fine-Tuning*: Fine-tune GPT-2 on your own song lyrics dataset.
- *Lyrics Generation*: Generate song lyrics using a pre-trained model.
- *No Setup on Your Own Dataset*: Only requires a .txt file with song lyrics to fine-tune and generate lyrics.
- *Customizable*: Change training parameters like epochs, batch size, etc.

## 🧠 Model

- *Model Used*: [gpt2](https://huggingface.co/gpt2) (can also use gpt2-medium or gpt2-large)
- *Framework*: [transformers](https://huggingface.co/docs/transformers) by Hugging Face
- *Training*: Fine-tuning GPT-2 on your custom song lyrics dataset.

## 📦 Requirements

Before you begin, install the following libraries:

```bash
pip install transformers datasets --quiet
