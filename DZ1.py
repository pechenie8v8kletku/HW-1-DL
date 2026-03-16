import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split,DataLoader
from tqdm import tqdm
import itertools
from torch.utils.tensorboard import SummaryWriter
class ConvModel(nn.Module):
    def __init__(self,in_channels=1,out_channels=10,num_layers=5,kernel_size=5,percent=0.2,batch_norm=True,pool="NONE"):
        super(ConvModel,self).__init__()
        self.Layers=nn.ModuleList()


        if pool=="NONE":
            self.pool=nn.Identity()
        elif pool=="AVG":
            self.pool=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        elif pool=="MAX":
            self.pool=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        for i in range(num_layers):
            conv=nn.Sequential(
            nn.Conv2d(1,1,kernel_size=kernel_size,stride=1,padding=int((kernel_size-1)//2),bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=percent),)
            if batch_norm:
                conv.append(nn.BatchNorm2d(1))
            self.Layers.append(conv)
        self.flatten = nn.Flatten()
        self.final=nn.Linear(in_features=28*28, out_features=out_channels, bias=True)
    def forward(self,x):
        for layer in self.Layers:
            x=layer(x)
            x=self.pool(x)
        x=self.flatten(x)
        x=self.final(x)
        return x
def test():
    x = torch.randn((3, 1, 28, 28))
    model = ConvModel(in_channels=1, out_channels=10,pool="AVG")
    preds = model(x)
    print(preds)


train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
val_size=int(0.15 * len(train_dataset))
train_size=len(train_dataset)-val_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])




DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH=40
train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

param_grid = {
    "num_layers": [5,7,9,11],
    "kernel_size": [7,9],
    "percent": [0.03],
    "pool": ["AVG"]
}

keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys,v)) for v in itertools.product(*values)]
def train(params):
    model=ConvModel(**params).to(DEVICE)
    loss_fn=nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    run_name = "_".join([f"{k}={v}" for k, v in params.items()])
    writer = SummaryWriter(f"runs/{run_name}")

    for epoch in range(EPOCH):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=False)
        for x, y in loop:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        train_loss=total_loss / len(train_loader)
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                loss = loss_fn(pred, y)
                total_loss += loss.item()
                predicted = torch.argmax(pred, dim=1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        val_loss=total_loss / len(val_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")
    writer.close()


def main():
    for params in experiments:
        train(params)

if __name__ == "__main__":
    main()

