import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

from mmi_loss import MMI
import evaluation
from util import next_batch
from mmd_loss import MMD


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, feature Z^v.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, feature Z^v.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction samples.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, feature Z^v.
              x_hat:  [num, feat_dim] float tensor, reconstruction samples.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Apadc():
    """Apadc module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]

        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)

    def train(self, config, logger, x1_train, x2_train, Y_list, mask, optimizer, device):
        """Training the model.
            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari
        """

        # Get complete data for training
        flag_1 = (torch.LongTensor([1, 1]).to(device) == mask).int()
        Y_list = torch.tensor(Y_list).int().to(device).squeeze(dim=0).unsqueeze(dim=1)
        Tmp_acc, Tmp_nmi, Tmp_ari = 0, 0, 0
        for epoch in range(config['training']['epoch'] + 1):
            X1, X2, X3, X4 = shuffle(x1_train, x2_train, flag_1[:, 0], flag_1[:, 1])
            loss_all, loss_rec1, loss_rec2, loss_mmi, loss_mmd = 0, 0, 0, 0, 0
            for batch_x1, batch_x2, x1_index, x2_index, batch_No in next_batch(X1, X2, X3, X4, config['training']['batch_size']):
                if len(batch_x1) == 1:
                    continue
                index_both = x1_index + x2_index == 2                      # C in indicator matrix A of complete multi-view data
                index_peculiar1 = (x1_index + x1_index + x2_index == 2)    # I^1 in indicator matrix A of incomplete multi-view data
                index_peculiar2 = (x1_index + x2_index + x2_index == 2)    # I^2 in indicator matrix A of incomplete multi-view data
                z_1 = self.autoencoder1.encoder(batch_x1[x1_index == 1])   # [Z_C^1;Z_I^1]
                z_2 = self.autoencoder2.encoder(batch_x2[x2_index == 1])   # [Z_C^2;Z_I^2]
                
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_1), batch_x1[x1_index == 1])
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_2), batch_x2[x2_index == 1])
                rec_loss = (recon1 + recon2)                               # reconstruction losses \sum L_REC^v

                z_view1_both = self.autoencoder1.encoder(batch_x1[index_both])
                z_view2_both = self.autoencoder2.encoder(batch_x2[index_both])
                
                if len(batch_x2[index_peculiar2]) % config['training']['batch_size'] == 1:
                    continue
                z_view2_peculiar = self.autoencoder2.encoder(batch_x2[index_peculiar2])
                if len(batch_x1[index_peculiar1]) % config['training']['batch_size'] == 1:
                    continue
                z_view1_peculiar = self.autoencoder1.encoder(batch_x1[index_peculiar1])

                w1 = torch.var(z_view1_both)
                w2 = torch.var(z_view2_both)
                a1 = w1 / (w1 + w2)
                a2 = 1 - a1
                # the weight matrix is only used in MMI loss to explore the common cluster information
                # z_i = \sum a_iv w_iv z_i^v, here, w_iv = var(Z^v)/(\sum a_iv var(Z^v)) for MMI loss
                Z = torch.add(z_view1_both * a1, z_view2_both * a2)
                # mutual information losses \sum L_MMI^v (Z_C, Z_I^v)
                mmi_loss = MMI(z_view1_both, Z) + MMI(z_view2_both, Z)

                view1 = torch.cat([z_view1_both, z_view1_peculiar, z_view2_peculiar], dim=0)
                view2 = torch.cat([z_view2_both, z_view1_peculiar, z_view2_peculiar], dim=0)
                # z_i = \sum a_iv w_iv z_i^v, here, w_iv = 1/\sum a_iv for MMD loss
                view_both = torch.add(view1, view2).div(2)
                # mean discrepancy losses   \sum L_MMD^v (Z_C, Z_I^v)
                mmd_loss = MMD(view1, view_both, kernel_mul=config['training']['kernel_mul'], kernel_num=config['training']['kernel_num']) + \
                           MMD(view2, view_both, kernel_mul=config['training']['kernel_mul'], kernel_num=config['training']['kernel_num'])

                # total loss
                loss = mmi_loss + mmd_loss * config['training']['lambda1'] + rec_loss * config['training']['lambda2']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_mmd += mmd_loss.item()
                loss_mmi += mmi_loss.item()

            if (epoch) % config['print_num'] == 0:
                # output = "Epoch: {:.0f}/{:.0f} " \
                #          "==> REC loss = {:.4f} " \
                #          "==> MMD loss = {:.4f} " \
                #          "==> MMI loss = {:.4e} " \
                #     .format(epoch, config['training']['epoch'], (loss_rec1 + loss_rec2), loss_mmd, loss_mmi)
                output = "Epoch: {:.0f}/{:.0f} " \
                    .format(epoch, config['training']['epoch'])
                print(output)
                # evalution
                scores = self.evaluation(config, logger, mask, x1_train, x2_train, Y_list, device)
                if scores['kmeans']['ACC'] >= Tmp_acc:
                    Tmp_acc = scores['kmeans']['ACC']
                    Tmp_nmi = scores['kmeans']['NMI']
                    Tmp_ari = scores['kmeans']['ARI']
        return Tmp_acc, Tmp_nmi, Tmp_ari

    def evaluation(self, config, logger, mask, x1_train, x2_train, Y_list, device):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval()
            
            flag = mask[:, 0] + mask[:, 1] == 2           # complete multi-view data
            view2_missing_idx_eval = mask[:, 0] == 0      # incomplete multi-view data
            view1_missing_idx_eval = mask[:, 1] == 0      # incomplete multi-view data

            common_view1 = x1_train[flag]
            common_view1 = self.autoencoder1.encoder(common_view1)
            common_view2 = x2_train[flag]
            common_view2 = self.autoencoder2.encoder(common_view2)
            y_common = Y_list[flag]

            view1_exist = x1_train[view1_missing_idx_eval]
            view1_exist = self.autoencoder1.encoder(view1_exist)
            y_view1_exist = Y_list[view1_missing_idx_eval]
            view2_exist = x2_train[view2_missing_idx_eval]
            view2_exist = self.autoencoder2.encoder(view2_exist)
            y_view2_exist = Y_list[view2_missing_idx_eval]

            # Since the distributions of different views have been aligned,
            # it is OK to obtain the common features for clustering by the following two approaches

            # (1) z_i = \sum a_iv w_iv z_i^v, here, w_iv = var(Z^v)/(\sum a_iv var(Z^v))
            # w1 = torch.var(common_view1)
            # w2 = torch.var(common_view2)
            # a1 = w1 / (w1 + w2)
            # a2 = 1 - a1
            # common = torch.add(common_view1 * a1, common_view2 * a2)

            # (2) z_i = \sum a_iv w_iv z_i^v, here, w_iv = 1/\sum a_iv
            common = torch.add(common_view1, common_view2).div(2)

            latent_fusion = torch.cat([common, view1_exist, view2_exist], dim=0).cpu().detach().numpy()
            Y_list = torch.cat([y_common, y_view1_exist, y_view2_exist], dim=0).cpu().detach().numpy()

            scores, _ = evaluation.clustering([latent_fusion], Y_list[:, 0])
            # print("\033[2;29m" + 'Common features ' + str(scores) + "\033[0m")
            self.autoencoder1.train(), self.autoencoder2.train()
            
        return scores
