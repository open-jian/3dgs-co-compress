import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
from torch.utils.tensorboard import SummaryWriter
import argparse

torch.autograd.set_detect_anomaly(True)


def default_checkpoint_root():
    output_root = os.path.abspath(
        os.environ.get(
            "OUTPUT_ROOT",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "Output"),
        )
    )
    return os.path.join(output_root, "colasplat", "autoencoder", "ckpt")


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--checkpoint_root', type=str, default=default_checkpoint_root())
    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    data_dir = f"{dataset_path}/language_features"
    checkpoint_dir = os.path.join(args.checkpoint_root, args.dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_dataset = Autoencoder_dataset(data_dir)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        drop_last=False  
    )
    
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logdir = checkpoint_dir
    # tb_writer = SummaryWriter(logdir)

    best_eval_loss = 100.0
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for idx, feature in enumerate(train_loader):
            data = feature.to('cuda')
            outputs_dim3 = model.encode(data)
            outputs = model.decode(outputs_dim3)
            
            l2loss = l2_loss(outputs, data) 
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * 0.001
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter = epoch * len(train_loader) + idx
            # tb_writer.add_scalar('train_loss/l2_loss', l2loss.item(), global_iter)
            # tb_writer.add_scalar('train_loss/cos_loss', cosloss.item(), global_iter)
            # tb_writer.add_scalar('train_loss/total_loss', loss.item(), global_iter)
            # tb_writer.add_histogram("feat", outputs, global_iter)
            
        print("loss:{:.8f}".format(loss))
        if epoch > 0.95*num_epochs:
            eval_loss = 0.0
            model.eval()
            for idx, feature in enumerate(test_loader):
                data = feature.to('cuda')
                with torch.no_grad():
                    outputs = model(data) 
                loss = l2_loss(outputs, data) + cos_loss(outputs, data)
                eval_loss += loss * len(feature)
            eval_loss = eval_loss / len(train_dataset)
            print("eval_loss:{:.8f}".format(eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_ckpt.pth'))

            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{epoch}_ckpt.pth'))
            
    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))
