import torch
from torch.nn import functional as F
from torch import nn
import torch
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics.classification import accuracy
import layer_modules as cl
import importlib


from helper_modules import compute_preds_one_hot
import networks_helpers as nh

class Conv4(LightningModule):

    def __init__(self,config, num_flat_features=8192,):
        super(Conv4, self).__init__()

        self.save_hyperparameters(config)
        self.num_flat_features = num_flat_features
        self.criterion = nn.CrossEntropyLoss()
        self.training_acc = accuracy.Accuracy()
        self.validation_acc = accuracy.Accuracy()
        self.test_acc = accuracy.Accuracy()
        self.lr = self.hparams.lr
        self.layer_config = {"strategy":self.hparams.strategy}
        self.layer_config.update(config)
        self.strategy = importlib.import_module("strategies."+self.hparams.strategy)

        self.example_input_array = torch.rand(1,3,32,32)
  

        self.conv1 = cl.PrunableConv2d(3, 64, 3, config=self.layer_config)
        self.conv2 = cl.PrunableConv2d(64, 64, 3, config=self.layer_config)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = cl.PrunableConv2d(64, 128, 3, config=self.layer_config)
        self.conv4 = cl.PrunableConv2d(128, 128, 3, config=self.layer_config)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = cl.PrunableLinear(self.num_flat_features, 256, config=self.layer_config)
        self.fc2 = cl.PrunableLinear(256, 256, config=self.layer_config)
        self.fc3 = cl.PrunableLinear(256, 10, config=self.layer_config)



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = nh.add_model_specific_args(parent_parser)
        return parser

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool1(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pool2(out)
        out = out.view(-1, self.num_flat_features)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        
        x, y = batch
        logits = self(x)
        train_acc = self.training_acc(compute_preds_one_hot(logits), y)
        loss = self.criterion(logits, y)

        self.log("train_acc",train_acc, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)


        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        val_acc = self.validation_acc(compute_preds_one_hot(y_hat),y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        test_acc = self.test_acc(compute_preds_one_hot(y_hat),y)
        self.log('test_loss', loss, logger=True)
        self.log('test_acc', test_acc, logger=True)

        return loss

    def configure_optimizers(self):
        return self.strategy.optimizers_configuration(self)
    
 
class Conv2(LightningModule):

    def __init__(self,config, num_flat_features=16384,):
        super(Conv2, self).__init__()

        self.save_hyperparameters(config)
        self.num_flat_features = num_flat_features
        self.criterion = nn.CrossEntropyLoss()
        self.training_acc = accuracy.Accuracy()
        self.validation_acc = accuracy.Accuracy()
        self.test_acc = accuracy.Accuracy()
        self.lr = self.hparams.lr
        self.layer_config = {"strategy":self.hparams.strategy}
        self.layer_config.update(config)
        self.strategy = importlib.import_module("strategies."+self.hparams.strategy)

        self.example_input_array = torch.rand(1,3,32,32)

        self.conv1 = cl.PrunableConv2d(3, 64, 3, config=self.layer_config)
        self.conv2 = cl.PrunableConv2d(64, 64, 3, config=self.layer_config)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.fc1 = cl.PrunableLinear(self.num_flat_features, 256, config=self.layer_config)
        self.fc2 = cl.PrunableLinear(256, 256, config=self.layer_config)
        self.fc3 = cl.PrunableLinear(256, 10, config=self.layer_config)
       
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = nh.add_model_specific_args(parent_parser)
        return parser

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool1(out)
        out = out.view(-1, self.num_flat_features)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        
        x, y = batch
        logits = self(x)
        train_acc = self.training_acc(compute_preds_one_hot(logits), y)
        loss = self.criterion(logits, y)

        self.log("train_acc",train_acc, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)


        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        val_acc = self.validation_acc(compute_preds_one_hot(y_hat),y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        test_acc = self.test_acc(compute_preds_one_hot(y_hat),y)
        self.log('test_loss', loss, logger=True)
        self.log('test_acc', test_acc, logger=True)

        return loss

    def configure_optimizers(self):
        return self.strategy.optimizers_configuration(self)        

class Conv6(LightningModule):

    def __init__(self,config, num_flat_features=4096,):
        super(Conv6, self).__init__()

        self.save_hyperparameters(config)
        self.num_flat_features = num_flat_features
        self.criterion = nn.CrossEntropyLoss()
        self.training_acc = accuracy.Accuracy()
        self.validation_acc = accuracy.Accuracy()
        self.test_acc = accuracy.Accuracy()
        self.lr = self.hparams.lr
        self.layer_config = {"strategy":self.hparams.strategy}
        self.layer_config.update(config)
        self.strategy = importlib.import_module("strategies."+self.hparams.strategy)

    
     
        self.conv1 = cl.PrunableConv2d(3, 64, 3, config=self.layer_config)
        self.conv2 = cl.PrunableConv2d(64, 64, 3, config=self.layer_config)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = cl.PrunableConv2d(64, 128, 3, config=self.layer_config)
        self.conv4 = cl.PrunableConv2d(128, 128, 3, config=self.layer_config)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv5 = cl.PrunableConv2d(128, 256, 3, config=self.layer_config)
        self.conv6 = cl.PrunableConv2d(256, 256, 3, config=self.layer_config)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.fc1 = cl.PrunableLinear(self.num_flat_features, 256, config=self.layer_config)
        self.fc2 = cl.PrunableLinear(256, 256, config=self.layer_config)
        self.fc3 = cl.PrunableLinear(256, 10, config=self.layer_config)
     
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = nh.add_model_specific_args(parent_parser)
        return parser

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool1(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pool2(out)
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = self.pool3(out)
        out = out.view(-1, self.num_flat_features)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        
        x, y = batch
        logits = self(x)
        train_acc = self.training_acc(compute_preds_one_hot(logits), y)
        loss = self.criterion(logits, y)

        self.log("train_acc",train_acc, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)


        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        val_acc = self.validation_acc(compute_preds_one_hot(y_hat),y)
    
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        test_acc = self.test_acc(compute_preds_one_hot(y_hat),y)
        self.log('test_loss', loss, logger=True)
        self.log('test_acc', test_acc, logger=True)

        return loss

    def configure_optimizers(self):
        return self.strategy.optimizers_configuration(self)