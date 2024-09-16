# Sequence-to-Sequence Transformer Task


## Major args inputs with respect to the problem

    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length of Transformer encoder')   # Encoder seq length 
    parser.add_argument('--label_len', type=int, default=10, help='start token length of Transformer decoder')     # Length of the Start Token 
    parser.add_argument('--pred_len', type=int, default=90, help='prediction sequence length')                     # Prediction sequence length -- Start token + 90+predictions


    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')    # Encoder input size
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')    # Decoder input size
    parser.add_argument('--c_out', type=int, default=1, help='output size')            #output size 


## ETT/dataloader.py file handles data.
## models/model.py has the implementation.
## results

final predictions : 
Epoch: 5, Steps: 2148 | Train Loss: 0.0294454 Vali Loss: 0.0882389 Test Loss: 0.2292483

## Use python3 main_informer.py for execution