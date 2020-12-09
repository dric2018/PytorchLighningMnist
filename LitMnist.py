import torch 
import torch.nn as nn
import torchvision
from torchvision import transforms
import pytorch_lightning as pl 
import os
import argparse 


parser = argparse.ArgumentParser()

parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--arch', default='cnn', type=str, help='One of cnn, lstm, gru, rnn')
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--test_batch_size', default=32, type=int)


class CNNDigitClassifier(pl.LightningModule):
    def __init__(self):
        super(CNNDigitClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3))
        self.conv2 = torch.nn.Conv2d(32, 64, (3, 3))
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.conv3 = torch.nn.Conv2d(64, 128, (3, 3))
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.bn2 = torch.nn.BatchNorm2d(128)

        self.classifier = torch.nn.Linear(128 * 5 * 5, 10)


    def forward(self, x):
        x = torch.relu(self.conv1(x))        
        x = torch.relu(self.conv2(x))   
        x = torch.relu(self.pool1(self.bn1(x)))

        x = torch.relu(self.conv3(x))        
        x = torch.relu(self.pool2(self.bn2(x)))
        x = self.classifier(x.view(-1, 128 * 5 * 5))

        return x

    
    def training_step(self, batch, batch_idx):
        x , y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        acc = (preds.argmax(1) == y).float().mean()

        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x , y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        val_loss = torch.nn.CrossEntropyLoss()(logits, y)
        val_acc = (preds.argmax(1) == y).float().mean()

        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(lr=1e-2, params=self.parameters())
        return opt


class RNNDigitClassifier(pl.LightningModule):
    def __init__(self, input_size=28, seq_len =28, hidden_size=256,num_layers=2, out_size=10, device='cuda'):
        super(RNNDigitClassifier, self).__init__()
        self.save_hyperparameters()

        # arch 
        self.rnn_layer = nn.RNN(input_size=self.hparams.input_size, 
                                hidden_size=self.hparams.hidden_size,
                                num_layers=self.hparams.num_layers,
                                batch_first=True)
        
        # classifier
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.out_size)


    def forward(self, x):
        # defince initial hidden state
        h0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size).to(self.hparams.device)

        # forward pass
        out, _ = self.rnn_layer(x.squeeze(dim=1), h0)
        out = out[:, -1, :]
        logits = self.fc(out) 

        return logits

    
    def training_step(self, batch, batch_idx):
        x , y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        acc = (preds.argmax(1) == y).float().mean()

        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x , y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        val_loss = torch.nn.CrossEntropyLoss()(logits, y)
        val_acc = (preds.argmax(1) == y).float().mean()

        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    def configure_optimizers(self):
        opt = torch.optim.SGD(lr=1e-2, params=self.parameters())
        return opt


class LSTMDigitClassifier(pl.LightningModule):
    def __init__(self, input_size=28, seq_len =28, hidden_size=256,num_layers=2, out_size=10, device='cuda'):
        super(LSTMDigitClassifier, self).__init__()
        self.save_hyperparameters()

        # arch 
        self.lstm_layer = nn.LSTM(input_size=self.hparams.input_size, 
                                hidden_size=self.hparams.hidden_size,
                                num_layers=self.hparams.num_layers,
                                batch_first=True)
        
        # classifier
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.out_size)


    def forward(self, x):
        # defince initial hidden state
        h0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size).to(self.hparams.device)
        C0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size).to(self.hparams.device)
        # forward pass
        out, _ = self.lstm_layer(x.squeeze(dim=1), (h0, C0))
        out = out[:, -1, :]
        logits = self.fc(out) 

        return logits

    
    def training_step(self, batch, batch_idx):
        x , y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        acc = (preds.argmax(1) == y).float().mean()

        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x , y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        val_loss = torch.nn.CrossEntropyLoss()(logits, y)
        val_acc = (preds.argmax(1) == y).float().mean()

        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(lr=1e-2, params=self.parameters())
        return opt




class GRUDigitClassifier(pl.LightningModule):
    def __init__(self, input_size=28, seq_len =28, hidden_size=256,num_layers=2, out_size=10, device='cuda'):
        super(GRUDigitClassifier, self).__init__()
        self.save_hyperparameters()

        # arch 
        self.gru_layer = nn.GRU(input_size=self.hparams.input_size, 
                                hidden_size=self.hparams.hidden_size,
                                num_layers=self.hparams.num_layers,
                                batch_first=True)
        
        # classifier
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.out_size)


    def forward(self, x):
        # defince initial hidden state
        h0 = torch.zeros(self.hparams.num_layers, x.size(0), self.hparams.hidden_size).to(self.hparams.device)

        # forward pass
        out, _ = self.gru_layer(x.squeeze(dim=1), h0)
        #out = out.reshape(out.shape[0], -1)
        out = out[:, -1, :] # keep the last hidden state only
        logits = self.fc(out) 

        return logits

    
    def training_step(self, batch, batch_idx):
        x , y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        acc = (preds.argmax(1) == y).float().mean()

        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x , y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        val_loss = torch.nn.CrossEntropyLoss()(logits, y)
        val_acc = (preds.argmax(1) == y).float().mean()

        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return val_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(lr=1e-2, params=self.parameters())
        return opt

if __name__ == "__main__":

    args = parser.parse_args()

    train_ds = torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

    val_ds = torchvision.datasets.MNIST('./files/', train=False, download=True,
                                transform=transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))



    train_dl = torch.utils.data.DataLoader(dataset=train_ds, 
                                            batch_size=args.train_batch_size, 
                                            shuffle=True, 
                                            num_workers=os.cpu_count())


    val_dl = torch.utils.data.DataLoader(dataset=val_ds, 
                                        batch_size=args.test_batch_size, 
                                        shuffle=False, 
                                        num_workers=os.cpu_count())


    if args.arch == "cnn":
        model = CNNDigitClassifier()
    elif args.arch == 'rnn':
        model = RNNDigitClassifier()
    elif args.arch == "lstm":
        model = LSTMDigitClassifier()
    else:
        model = GRUDigitClassifier()

    args = parser.parse_args()
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.num_epochs)

    trainer.fit(model=model, train_dataloader=train_dl, val_dataloaders=val_dl)
