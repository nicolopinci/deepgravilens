import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Transformer1d(nn.Module):
    def __init__(self, d_model, nhead, num_layers=4):
        super(Transformer1d, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def forward(self, x):

        out = self.transformer_encoder(x)

        return out

    
class LoNet(nn.Module):
    def __init__(self, in_channels = 4, input_size = 4, num_classes = 4, ts_length = None, informed = None, ds_name = None):
        super(LoNet, self).__init__()

        hidden_size_GRU = None
        if ts_length == 7:
            hidden_size_GRU = 32
        elif ts_length == 14:
            hidden_size_GRU = 64
        else:
            raise Exception("Invalid timeseries length")

        # Network Components
        self.cnn = DeepCNN(ts_length = ts_length, informed = informed, num_classes = num_classes)
        self.gru = ShallowGRU(ts_length = ts_length, informed = informed, num_classes = num_classes)
        
        cnn_model_path = '../models/' + ds_name + '/' + ds_name + '_DeepCNN_informed_network.pt'
        gru_model_path = '../models/' + ds_name + '/' + ds_name + '_ShallowGRU_informed_network.pt'
        
        self.cnn.load_state_dict(torch.load(cnn_model_path))  # model trained with images
        self.gru.load_state_dict(torch.load(gru_model_path))  # model trained with time series

        tr_input_size = 2*64
        
        self.informed = informed
        if informed:
            tr_input_size += 2*in_channels

        self.Transformer1d = Transformer1d(d_model= tr_input_size, nhead=4)
        
        self.fcn = nn.Sequential(
            nn.Linear(tr_input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        
        self.out_fc = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes),
        )
        

    def forward(self, x, y, return_features = False):
        gru_output = self.gru(x, y, return_features = True)
        cnn_output = self.cnn(x, y, return_features = True)
        means = torch.mean(x, 1)
        stds = torch.std(x, 1)

        if self.informed:
            full_output = torch.cat((means, stds, gru_output, cnn_output), dim=1)
        else:
            full_output = torch.cat((gru_output, cnn_output), dim=1)

        
        fuse_cat = self.Transformer1d(full_output)
        fuse_cat = fuse_cat.view(fuse_cat.size(0), -1)

        out = self.fcn(fuse_cat)
        
        if return_features:
            return out
        
        out = self.out_fc(out)
        
        out = F.log_softmax(out, dim=1)
        return out
    
    
    
class MuNet(nn.Module):
    def __init__(self, in_channels = 4, num_classes = 4, image_side = 45, ts_length = None, informed = None, ds_name = None):
        super(MuNet, self).__init__()

        cnn_input_size = (image_side * image_side) * in_channels
        self.informed = informed

        # Network Components
        self.fc = SmallImageFC(ts_length = ts_length, informed = informed, num_classes = num_classes)
        self.cnn = DeepCNN(ts_length = ts_length, informed = informed, num_classes = num_classes)
        self.gru = ShallowGRU(ts_length = ts_length, informed = informed, num_classes = num_classes)
        
        fc_model_path = '../models/' + ds_name + '/' + ds_name + '_SmallImageFC_informed_network.pt'
        cnn_model_path = '../models/' + ds_name + '/' + ds_name + '_DeepCNN_informed_network.pt'
        gru_model_path = '../models/' + ds_name + '/' + ds_name + '_ShallowGRU_informed_network.pt'
        
        self.fc.load_state_dict(torch.load(fc_model_path))
        self.cnn.load_state_dict(torch.load(cnn_model_path))
        self.gru.load_state_dict(torch.load(gru_model_path))
        

        linear_input_size = 128

        if informed:
            linear_input_size += 2*in_channels
            
        self.fc_cnn = nn.Linear(64, 32)

        
        self.fcn = nn.Sequential(
            nn.Linear(linear_input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        
        self.out_fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes),
        )
        
        
        

    def forward(self, x, y, return_features = False):
        fc_output = self.fc(x, y, return_features = True)
        gru_output = self.gru(x, y, return_features = True)
        cnn_output = self.cnn(x, y, return_features = True)
        
        cnn_output = self.fc_cnn(cnn_output)
        
        flattened_image = torch.flatten(y, 1)

        if self.informed:
            means = torch.mean(x, 1)
            stds = torch.std(x, 1)

        if self.informed:
            out = torch.cat((fc_output, cnn_output, gru_output, means, stds), dim=1)
        else:
            out = torch.cat((fc_output, cnn_output, gru_output), dim=1)

        fcn_out =  self.fcn(out)
        
        if return_features:
            return fcn_out
        
        final_output = self.out_fc(fcn_out)
        
        final_output = F.log_softmax(final_output, dim=1)

        return final_output
    


    
class ShallowGRU(nn.Module):
    def __init__(self, in_channels=4, input_size=4, num_classes=4, ts_length=None, informed=None):
        super(ShallowGRU, self).__init__()
        
        if ts_length == 7:
            hidden_size = 8  # rnn hidden unit
        else:
            hidden_size = 16

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )
        
        self.bn = nn.BatchNorm1d(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
        )
        self.final_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, img, return_features = False):
        r_out, _ = self.rnn(x, None)
        r_out = r_out[:, -1, :]
        r_out = self.bn(r_out)
        out = self.classifier(r_out)
        
        if return_features:
            return out
        
        out = self.final_classifier(out)
        out = F.log_softmax(out, dim=1)
        
        return out
    
    
        

    
    

    
class SmallImageFC(nn.Module):
    def __init__(self, in_channels = 4, num_classes = 4, image_side = 45, ts_length = None, informed = None):
        super(SmallImageFC, self).__init__()

        cnn_input_size = (image_side * image_side) * in_channels
        self.fc1 = nn.Linear(cnn_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, num_classes)
       
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(16)
        self.bn6 = nn.BatchNorm1d(8)
        self.bn7 = nn.BatchNorm1d(num_classes)

    
    def forward(self, x, y, return_features = False):
        flattened_image = torch.flatten(y, 1)

        full_output = self.fc1(flattened_image)
        full_output = self.bn1(full_output)
        full_output = F.relu(full_output)
        
        full_output = self.fc2(full_output)
        full_output = self.bn2(full_output)
        full_output = F.relu(full_output)
        
        full_output = self.fc3(full_output)
        full_output = self.bn3(full_output)
        full_output = F.relu(full_output)
        
        full_output = self.fc4(full_output)
        full_output = self.bn4(full_output)
        full_output = F.relu(full_output)
        
        if return_features:
            return full_output
        
        full_output = self.fc5(full_output)
        full_output = self.bn5(full_output)
        full_output = F.relu(full_output)
        
        full_output = self.fc6(full_output)
        full_output = self.bn6(full_output)
        full_output = F.relu(full_output)
        
        full_output = self.fc7(full_output)
        full_output = self.bn7(full_output)   

        final_output = F.log_softmax(full_output, dim=1)

        return final_output
    
    
    
class DeepCNN(nn.Module):
    def __init__(self, in_channels=4, input_size=4, num_classes=4, ts_length=None, informed=None):
        super(DeepCNN, self).__init__()
        
        #Network Components
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=64, 
                               kernel_size=15, 
                               stride=3,
                               padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(in_channels=64, 
                               out_channels=128,
                               kernel_size=5, 
                               stride=2,
                               padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(in_channels=128, 
                               out_channels=256,
                               kernel_size=5, 
                               stride=2,
                               padding=2)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.dropout2 = nn.Dropout2d(0.5)
        
        self.fc1 = nn.Linear(in_features=2304, 
                             out_features=512)
        
        self.fc2 = nn.Linear(in_features=512, 
                             out_features=128)

        self.fc3 = nn.Linear(in_features=128,
                             out_features=64)
        
        self.fc4 = nn.Linear(in_features=64,
                             out_features=num_classes)
        
    def forward(self, ts, x, return_features = False):
        #Network Flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        if return_features:
            return x
        
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

    
    


class GloNet(nn.Module):
    def __init__(self, num_classes=4, ts_length=None, image_side=45, num_channels=4, informed=None, ds_name=None):
        super(GloNet, self).__init__()
        tr_input_size = 4 * 45 * 45 + ts_length * 4

        self.informed = informed
        if informed:
            tr_input_size += 2 * num_channels

        hidden_dim = ts_length*4*2

        self.fcn = nn.Sequential(
            nn.Linear(tr_input_size, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )
        
        self.out_fc = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x, y, return_features=False):
        flattened_ts = torch.flatten(x, start_dim=1)
        flattened_image = torch.flatten(y, start_dim=1)
                        
        means = torch.mean(x, 1)
        stds = torch.std(x, 1)

        if self.informed:
            nn_input = torch.cat((means, stds, flattened_ts, flattened_image), dim=1)
        else:
            nn_input = torch.cat((flattened_ts, flattened_image), dim=1)

        
        # Classification
        out = self.fcn(nn_input)
        
        if return_features:
            return out
        
        out = self.out_fc(out)    
        out = F.log_softmax(out, dim=1)

        return out
    

    
    

    

    
class EvidentialLoNet(nn.Module):
    def __init__(self, in_channels = 4, input_size = 4, num_classes = 4, ts_length = None, informed = None, ds_name = None):
        super(EvidentialLoNet, self).__init__()
        
        self.num_classes = num_classes

        hidden_size_GRU = None
        if ts_length == 7:
            hidden_size_GRU = 32
        elif ts_length == 14:
            hidden_size_GRU = 64
        else:
            raise Exception("Invalid timeseries length")

        # Network Components
        self.cnn = DeepCNN(ts_length = ts_length, informed = informed, num_classes = num_classes)
        self.gru = ShallowGRU(ts_length = ts_length, informed = informed, num_classes = num_classes)
        
        cnn_model_path = '../models/' + ds_name + '/' + ds_name + '_DeepCNN_informed_network.pt'
        gru_model_path = '../models/' + ds_name + '/' + ds_name + '_ShallowGRU_informed_network.pt'
        
        self.cnn.load_state_dict(torch.load(cnn_model_path))  # model trained with images
        self.gru.load_state_dict(torch.load(gru_model_path))  # model trained with time series
        
        
        self.fcnn_gru = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(),
            nn.Linear(32, 4),
            nn.Softplus()
        )
        
        self.fcnn_cnn = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(),
            nn.Linear(32, 4),
            nn.Softplus()
        )
        

        
    def Evidence_fusion(self,D_alpha):
        '''
        :param D_alpha: the Dirichlet distribution parameters of two sense
        :return: the Combined Dirichlet distribution parameters
        '''


        def Evidence_two(alpha1,alpha2):
            '''
            :param alpha1: Dirichlet distribution parameters of aerial image
            :param alpha2: Dirichlet distribution parameters of ground image
            :return: Combined Dirichlet distribution parameters
            '''
            alpha, E, c, S, u = dict(), dict(), dict(), dict(), dict()
            alpha[0] = alpha1
            alpha[1] = alpha2
            # calculate E,b,S,u of each view
            for view_num in range(len(alpha)):
                E[view_num] = alpha[view_num]-1
                S[view_num] = torch.sum(alpha[view_num], dim=1, keepdim=True)
                c[view_num] = E[view_num] / (S[view_num].expand(E[view_num].shape))
                u[view_num] = self.num_classes / S[view_num]
            # c0 @ c1
            cc = torch.bmm(c[0].view(-1, self.num_classes, 1), c[1].view(-1, 1, self.num_classes))
            u1_expand = u[0].expand(c[0].shape)
            # c0 * (1-u0)
            view1_weight = torch.mul(c[0],(1-u1_expand))
            u2_expand = u[1].expand(c[0].shape)
            # c1 * (1-u1)
            view2_weight = torch.mul(c[1],(1-u2_expand))
            # c0 * (1-u0) of one view all classes
            v1weigt_all = torch.mul((1-u1_expand),(1-u1_expand))
            # c1 * (1-u1) of one view all classes
            v2weigt_all = torch.mul((1-u2_expand),(1-u2_expand))


            cc_diag = torch.diagonal(cc,dim1=-2,dim2=-1).sum(-1)
            # calculate b_after
            c_total = (torch.mul(c[0], c[1]) + view1_weight + view2_weight) /((cc_diag.view(-1, 1).expand(c[0].shape))
                                                                          +v1weigt_all+v2weigt_all+torch.mul((1-u1_expand),(1-u2_expand)))
            # calculate u_after
            u_total = torch.mul(1-u[0], 1-u[1]) / ((cc_diag.view(-1, 1).expand(u[0].shape))+torch.mul(1-u[0],1-u[0])
                                           +torch.mul(1-u[1],1-u[1])+torch.mul(1-u[0], 1-u[1]))

            # calculate S_after
            S_total = self.num_classes / u_total

            # calculate E_after
            E_total = torch.mul(c_total, S_total.expand(c_total.shape))
            # calculate alpha_after
            alpha_total = E_total + 1
            return alpha_total

        alpha_after = Evidence_two(D_alpha[0], D_alpha[1])
        return alpha_after
    
    
    def forward(self, x, y, return_features = False):
        
        reciprocal_loss_gru = 0
        reciprocal_loss_cnn = 0
        
        with torch.no_grad(): # freeze the backbone
            self.gru.eval()
            self.cnn.eval()
            gru_output = self.gru(x, y, return_features = True)
            cnn_output = self.cnn(x, y, return_features = True)

        gru_output = self.fcnn_gru(gru_output) # evidence GRU
        cnn_output = self.fcnn_cnn(cnn_output) # evidence CNN
        
        D_alpha_gru = gru_output + 1
        D_alpha_cnn = cnn_output + 1
        
        alpha = {0: D_alpha_gru, 1: D_alpha_cnn}
        
        alpha_after = self.Evidence_fusion(alpha)
        
        evidence_after = alpha_after - 1
        
        return evidence_after, alpha
    
    
    
    
    
    
class EvidentialMuNet(nn.Module):
    def __init__(self, in_channels = 4, input_size = 4, num_classes = 4, ts_length = None, informed = None, ds_name = None):
        super(EvidentialMuNet, self).__init__()
        
        self.num_classes = num_classes

        hidden_size_GRU = None
        if ts_length == 7:
            hidden_size_GRU = 32
        elif ts_length == 14:
            hidden_size_GRU = 64
        else:
            raise Exception("Invalid timeseries length")

         # Network Components
        self.fc = SmallImageFC(ts_length = ts_length, informed = informed, num_classes = num_classes)
        self.cnn = DeepCNN(ts_length = ts_length, informed = informed, num_classes = num_classes)
        self.gru = ShallowGRU(ts_length = ts_length, informed = informed, num_classes = num_classes)
        
        fc_model_path = '../models/' + ds_name + '/' + ds_name + '_SmallImageFC_informed_network.pt'
        cnn_model_path = '../models/' + ds_name + '/' + ds_name + '_DeepCNN_informed_network.pt'
        gru_model_path = '../models/' + ds_name + '/' + ds_name + '_ShallowGRU_informed_network.pt'
        
        self.fc.load_state_dict(torch.load(fc_model_path))
        self.cnn.load_state_dict(torch.load(cnn_model_path))
        self.gru.load_state_dict(torch.load(gru_model_path))
        
        self.fcnn_gru = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(),
            nn.Linear(32, 4),
            nn.Softplus()
        )
        
        self.fcnn_image = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(),
            nn.Linear(32, 4),
            nn.Softplus()
        )
        
        self.fc_cnn = nn.Linear(64, 32)
        

        
    def Evidence_fusion(self,D_alpha):
        '''
        :param D_alpha: the Dirichlet distribution parameters of two sense
        :return: the Combined Dirichlet distribution parameters
        '''


        def Evidence_two(alpha1,alpha2):
            '''
            :param alpha1: Dirichlet distribution parameters of aerial image
            :param alpha2: Dirichlet distribution parameters of ground image
            :return: Combined Dirichlet distribution parameters
            '''
            alpha, E, c, S, u = dict(), dict(), dict(), dict(), dict()
            alpha[0] = alpha1
            alpha[1] = alpha2
            # calculate E,b,S,u of each view
            for view_num in range(len(alpha)):
                E[view_num] = alpha[view_num]-1
                S[view_num] = torch.sum(alpha[view_num], dim=1, keepdim=True)
                c[view_num] = E[view_num] / (S[view_num].expand(E[view_num].shape))
                u[view_num] = self.num_classes / S[view_num]
            # c0 @ c1
            cc = torch.bmm(c[0].view(-1, self.num_classes, 1), c[1].view(-1, 1, self.num_classes))
            u1_expand = u[0].expand(c[0].shape)
            # c0 * (1-u0)
            view1_weight = torch.mul(c[0],(1-u1_expand))
            u2_expand = u[1].expand(c[0].shape)
            # c1 * (1-u1)
            view2_weight = torch.mul(c[1],(1-u2_expand))
            # c0 * (1-u0) of one view all classes
            v1weigt_all = torch.mul((1-u1_expand),(1-u1_expand))
            # c1 * (1-u1) of one view all classes
            v2weigt_all = torch.mul((1-u2_expand),(1-u2_expand))


            cc_diag = torch.diagonal(cc,dim1=-2,dim2=-1).sum(-1)
            # calculate b_after
            c_total = (torch.mul(c[0], c[1]) + view1_weight + view2_weight) /((cc_diag.view(-1, 1).expand(c[0].shape))
                                                                          +v1weigt_all+v2weigt_all+torch.mul((1-u1_expand),(1-u2_expand)))
            # calculate u_after
            u_total = torch.mul(1-u[0], 1-u[1]) / ((cc_diag.view(-1, 1).expand(u[0].shape))+torch.mul(1-u[0],1-u[0])
                                           +torch.mul(1-u[1],1-u[1])+torch.mul(1-u[0], 1-u[1]))

            # calculate S_after
            S_total = self.num_classes / u_total

            # calculate E_after
            E_total = torch.mul(c_total, S_total.expand(c_total.shape))
            # calculate alpha_after
            alpha_total = E_total + 1
            return alpha_total

        alpha_after = Evidence_two(D_alpha[0], D_alpha[1])
        return alpha_after
    
    
    def forward(self, x, y, return_features = False):
        
        reciprocal_loss_gru = 0
        reciprocal_loss_image = 0
        
        with torch.no_grad(): # freeze the backbone
            self.gru.eval()
            self.cnn.eval()
            self.fc.eval()
            
            fc_output = self.fc(x, y, return_features = True)
            gru_output = self.gru(x, y, return_features = True)
            cnn_output = self.cnn(x, y, return_features = True)
            
        cnn_output = self.fc_cnn(cnn_output) 
        image_output = torch.cat((fc_output, cnn_output), dim = 1)

        gru_output = self.fcnn_gru(gru_output) # evidence GRU
        image_output = self.fcnn_image(image_output) # evidence image
        
        D_alpha_gru = gru_output + 1
        D_alpha_image = image_output + 1
        
        alpha = {0: D_alpha_gru, 1: D_alpha_image}
        
        alpha_after = self.Evidence_fusion(alpha)
        
        evidence_after = alpha_after - 1
        
        return evidence_after, alpha