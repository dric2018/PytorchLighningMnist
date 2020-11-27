import torch 
import torchvision
from torchvision import transforms
import pytorch_lightning as pl 

import argparse 





class DigitClassifier(pl.LightningModule):
    def __init__(self):
        super(DigitClassifier, self).__init__()
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





if __name__ == "__main__":
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



    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=2048, shuffle=True, num_workers=4)
    val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=1024, shuffle=False, num_workers=4)

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)

    model = DigitClassifier()

    args = parser.parse_args()
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.num_epochs)

    trainer.fit(model=model, train_dataloader=train_dl, val_dataloaders=val_dl)
