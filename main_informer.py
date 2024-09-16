import argparse
import os
import torch
import sys

from exp.exp_transformer import exp_transformer

# Initialize argument parser
parser = argparse.ArgumentParser(description='Transformer for ET dataset')

# Add all the arguments as before
parser.add_argument('--data', type=str, default='ETTm1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--features', type=str, default='MS', help='forecasting task')   #multivariate to univariate 
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--freq', type=str, default='m', help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=100, help='input sequence length of Transformer encoder')   # Encoder seq length 
parser.add_argument('--label_len', type=int, default=10, help='start token length of Transformer decoder')     # Length of the Start Token 
parser.add_argument('--pred_len', type=int, default=90, help='prediction sequence length')                     # Prediction sequence length -- Start token + 90+predictions


parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')    # Encoder input size
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')    # Decoder input size
parser.add_argument('--c_out', type=int, default=1, help='output size')            #output size 

# Model hyperparameters
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

# Parse the arguments
args = parser.parse_args()

# Set variables from parsed args
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data = args.data
root_path = args.root_path
features = args.features
target = args.target
freq = args.freq
checkpoints = args.checkpoints
seq_len = args.seq_len
label_len = args.label_len
pred_len = args.pred_len
enc_in = args.enc_in
dec_in = args.dec_in
c_out = args.c_out

d_model = args.d_model
n_heads = args.n_heads
e_layers = args.e_layers
d_layers = args.d_layers
s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
d_ff = args.d_ff
factor = args.factor
padding = args.padding
distil = args.distil
dropout = args.dropout
attn = args.attn
embed = args.embed
activation = args.activation
output_attention = args.output_attention
do_predict = args.do_predict
mix = args.mix
cols = args.cols
num_workers = args.num_workers
itr = args.itr
train_epochs = args.train_epochs
batch_size = args.batch_size
patience = args.patience
learning_rate = args.learning_rate
des = args.des
loss = args.loss
lradj = args.lradj
use_amp = args.use_amp
inverse = args.inverse

# Display configurations
print('Configurations:')
for arg, value in vars(args).items():
    print(f'{arg}: {value}')

# Start experiment
Exp = exp_transformer

for ii in range(itr):
    setting = '{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
        data, features, seq_len, label_len, pred_len, d_model, n_heads, e_layers, d_layers, d_ff, 
        attn, factor, embed, distil, mix, des, ii)

    exp = Exp(args)  # Pass args to the experiment class
    # print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)

    # print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # exp.test(setting)

    # if do_predict:
    #     print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    #     exp.predict(setting, True)

    # torch.cuda.empty_cache()
