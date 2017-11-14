# Usage
`python train_bi.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48  --model_file model/model --gpuid 0`

# Part 1
For RNNLM, we use `nn.ModuleList()` to represent a sequential NN.
```
    def forward(self, input_batch):
        """
        input shape seq_len, batch_size
        ouput shape sequence_length, batch_size, vocab_size
        """
        output = input_batch
        for layer in self.layers:
            output = layer(output)
        return output
```
And we have class `Linear` `LogSoftmax` `Embedding` `RNN` which all are the subclass of `nn.Module` and overwrite their `forward()` method respectively.
# Part 2
For bi-directional RNNLM, we just need to modify the RNN layer.
```
class RNNLM(nn.Module):
    def __init__(self, vocab_size, bi_directional=False):
        #some code
        self.layers.append(RNN(self.embedding_size, self.hidden_size / num_dir, bi_directional=bi_directional))
class RNN(nn.Module):
    def __init__(self, in_size, out_size, bi_directional=False):
````
# Part 3
`python decoder.py > ouput` would load saved model and perform cloze task.

From this part, we implemented the dropout layer.
