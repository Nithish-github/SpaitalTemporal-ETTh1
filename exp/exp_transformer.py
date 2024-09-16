
from data.data_loader import Dataset_ETT_minute ,Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import sys



class exp_transformer(Exp_Basic):

    def __init__(self, args):
        super(exp_transformer, self).__init__(args)

    def _build_model(self):

        model_dict = {
            'Transformer':Informer,
        }

        e_layers = self.args.e_layers
        model = model_dict['Transformer'](
            self.args.enc_in,
            self.args.dec_in, 
            self.args.c_out, 
            self.args.seq_len, 
            self.args.label_len,
            self.args.pred_len, 
            self.args.factor,
            self.args.d_model, 
            self.args.n_heads, 
            e_layers, # self.args.e_layers,
            self.args.d_layers, 
            self.args.d_ff,
            self.args.dropout, 
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTm1':Dataset_ETT_minute
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq

        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )


        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        

        return data_set, data_loader

    def _select_optimizer(self):

        param_sgd = dict(self.model.encoder.attn_layers.named_parameters())
        param_adam = dict(self.model.named_parameters())

        param_sgd_name = param_sgd.keys()
        param_sgd_name = ['encoder.attn_layers.' + name for name in param_sgd_name]
        [param_adam.pop(name) for name in param_sgd_name]

        model_optim1 = optim.Adam(param_adam.values(), lr=self.args.learning_rate)
        model_optim2 = optim.SGD(param_sgd.values(), lr=10 * self.args.learning_rate)
        return model_optim1 , model_optim2
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark , batch_y_loss) in enumerate(vali_loader):
            pred = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), batch_y_loss.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        model_optim1, model_optim2 = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark , batch_y_loss) in enumerate(train_loader):
                iter_count += 1

                # model_optim.zero_grad()
                model_optim1.zero_grad()
                model_optim2.zero_grad()

                batch_y_loss = batch_y_loss.float().to(self.device) #(batch,pred_len+label_len,time_feature)


                pred = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = criterion(pred, batch_y_loss) #pred:(batch,pred_len,1)

                
                train_loss.append(loss.item())



                if (i+1) % 10==0:

                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    #save the predictions
                    self.plot_predictions(pred, batch_y_loss, i)
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim1)
                    scaler.step(model_optim2)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim1.step()
                    model_optim2.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch+1, self.args) #原始
            adjust_learning_rate(model_optim1, epoch+1, self.args)
            adjust_learning_rate(model_optim2, epoch+1, self.args) 

        self.model.cuda()
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{}, mae:{}, mspe:{}'.format(rmse, mae, mspe))  #mse, mae

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):


        batch_x = batch_x.float().to(self.device) #(batch,seq_len,feature)
        batch_y = batch_y.float() #(batch,pred_len+label_len,feature)
    
        batch_x_mark = batch_x_mark.float().to(self.device) #(batch,seq_len,time_feature)
        batch_y_mark = batch_y_mark.float().to(self.device) #(batch,pred_len+label_len,time_feature)

        # # # Print shapes of the tensors
        # print(f"batch_x shape: {batch_x.shape}")  # (batch, seq_len, feature)
        # print(f"batch_y shape: {batch_y.shape}")  # (batch, pred_len + label_len, feature)

        # print(f"batch_x_mark shape: {batch_x_mark.shape}")  # (batch, seq_len, time_feature)
        # print(f"batch_y_mark shape: {batch_y_mark.shape}")  # (batch, pred_len + label_len, time_feature)

        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark) #(batch,pred_len,1)

        # print("model output shape",outputs.shape)

        # print("Prediction output is ",outputs.shape)

        f_dim = -1 if self.args.features=='MS' else 0
        
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device) ##(batch,pred_len,1)

        return outputs
    



    def plot_predictions(self, preds, trues,epoch):


        # Convert to NumPy arrays for plotting
        preds_np = preds.detach().cpu().numpy()
        trues_np = trues.detach().cpu().numpy()
        # Extract the last dimension (the predicted value)
        preds = preds_np[:, :, -1]  # Shape becomes (batch_size, sequence_length)
        trues = trues_np[:, :, -1]  # Shape becomes (batch_size, sequence_length)

        # Indexes for plotting
        idx_true = [i for i in range(trues.shape[1])]  # Full sequence index for true values
        idx_pred = [i for i in range(preds.shape[1])]  # Full sequence index for predicted values

        # Ensure indices and predictions/truth have the same shape
        assert len(idx_true) == trues.shape[1], f"idx_true length {len(idx_true)} does not match true sequence length {trues.shape[1]}"
        assert len(idx_pred) == preds.shape[1], f"idx_pred length {len(idx_pred)} does not match predicted sequence length {preds.shape[1]}"

        # Creating the plot
        plt.figure(figsize=(15, 6))
        plt.rcParams.update({"font.size": 18})

        plt.grid(visible=True, which='major', linestyle='-')
        plt.grid(visible=True, which='minor', linestyle='--', alpha=0.5)
        plt.minorticks_on()

        # Plot true sequence (input + target)
        plt.plot(idx_true, trues[0], 'o-.', color='blue', label='True Sequence', linewidth=1)

        # Plot prediction sequence
        plt.plot(idx_pred, preds[0], 'o-.', color='limegreen', label='Predicted Sequence', linewidth=1)

        # Set title, labels, and legend
        plt.title(f"True vs Predicted Sequence at Epoch {epoch}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()

        # Save the plot
        plt.savefig(f"Epoch_{str(epoch)}.png")
        plt.close()

