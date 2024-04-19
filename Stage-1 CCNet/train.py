import torch
import argparse
from torch.utils.data import DataLoader

from utils import save_model
from modules import resnet, network, contrastive_loss, contrastive_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--num_workers', type=int, default=28, help='num workers')
parser.add_argument('--resnet', type=str, default='ResNet18', help='resnet')
parser.add_argument('--start_epoch', type=int, default=1, help='start epoch')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--instance_temperature', type=float, default=0.5, help='instance temperature')
parser.add_argument('--cluster_temperature', type=float, default=1.0, help='cluster temperature')
parser.add_argument('--feature_dim', type=int, default=128, help='feature_dim')
parser.add_argument('--model_path', type=str, default='./model', help='model_path') 
parser.add_argument('--seed', type=str, default=9, help='seed')
args = parser.parse_args()

def train():
    model.train()
    loss_epoch = 0
    for step, (x_i, x_j, _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i, x_j = x_i.to("cuda"), x_j.to("cuda")
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        
        if step % 250 == 0:
            print(f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        
        loss_epoch += loss.item()
    return loss_epoch


torch.cuda.manual_seed(args.seed)
model = network.Network(resnet.get_resnet(args.resnet), args.feature_dim, 2).to("cuda")
model = torch.nn.DataParallel(model, device_ids=[0, 1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, torch.device("cuda")).to("cuda")
criterion_cluster = contrastive_loss.ClusterLoss(class_num=2, temperature=args.cluster_temperature, device=torch.device("cuda")).to("cuda")

data_loader = DataLoader(
    contrastive_dataset.ContrastiveDataset("dataset.csv"),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.num_workers,
)

for epoch in range(args.start_epoch, args.epochs):
    lr = optimizer.param_groups[0]["lr"]
    loss_epoch = train()
    if epoch % 25 == 0:
        save_model(args, model, optimizer, epoch)
    print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
save_model(args, model, optimizer, args.epochs)