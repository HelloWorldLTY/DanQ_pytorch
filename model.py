import torch
from timm import create_model
from torchvision import transforms, datasets
import lightning as L


### The input data should have shape (batch, channels  (original 4, ATGC), length (original 1000, DNA sequence length))
class DanQ(nn.Module):
    def __init__(self,classes=919,linear_units=48640,activate='relu',*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if activate == 'relu':
            activation = nn.ReLU()
        elif activate == 'exp':
            activation = ExpActivation()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=320,kernel_size=19,padding=9),
            activation,
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(input_size=320, hidden_size=320, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(linear_units, 925),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(925, classes)
        )
    
    def forward(self, x):
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        out,(hn,cn) = self.lstm(x)
        out = out.transpose(1,2)
        out = out.contiguous().view(x.size()[0],-1)
        out = self.fc(out)
        return out
    
class DanQRegression(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DanQ(classes = 919)
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        val_loss = self.loss_fn(outputs, targets)
        self.log("val_loss", val_loss)
        return val_loss
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.0001)
