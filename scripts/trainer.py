import torch
from tqdm import tqdm
from torch.optim import Adam, SGD
import dgl
from bjCnet import Encoder, GatedGCN, MyHingeLoss

device = torch.device('cuda:0')

gconv = GatedGCN(300, 7, hidden_dim=300, num_layers=5).to(device)


class Trainer:
    def __init__(self, network_params_path):
        self.network_params_path = network_params_path

    def train(self, encoder_model, dataloader, optimizer, Loss):
        encoder_model.train()
        epoch_loss = 0

        for dataset in dataloader:
            bug_graphs = dataset[0]
            patch_graphs = dataset[1]
            clean_graphs = dataset[2]
            optimizer.zero_grad()

            b_g = encoder_model(bug_graphs)
            b_g = [encoder_model.encoder.project(g) for g in dgl.readout_nodes(b_g, "X")]

            p_g = encoder_model(patch_graphs)
            p_g = [encoder_model.encoder.project(g) for g in dgl.readout_nodes(p_g, "X")]

            c_g = encoder_model(clean_graphs)
            c_g = [encoder_model.encoder.project(g) for g in dgl.readout_nodes(c_g, "X")]

            b_g = torch.stack(b_g, dim=0)
            p_g = torch.stack(p_g, dim=0)
            c_g = torch.stack(c_g, dim=0)

            loss = Loss(b_g, p_g, c_g)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss

    def train_4_ft(self, encoder_model, dataloader, optimizer, Loss):
        encoder_model.train()
        epoch_loss = 0

        for data, label in dataloader:
            optimizer.zero_grad()
            gs = encoder_model(data)
            g_feats = [encoder_model.encoder.project(g) for g in dgl.readout_nodes(gs, "X")]

            label_tensor = torch.tensor(label, dtype=torch.long).to(device)
            g_feats_tensor = torch.stack(g_feats).to(device)

            loss = Loss(g_feats_tensor, label_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss

    def pretrain(self, dataloader):
        encoder_model = Encoder(encoder=gconv).to(device)

        optimizer = Adam(encoder_model.parameters(), lr=0.001)
        Loss = MyHingeLoss().to(device)

        with tqdm(total=100, desc='(T)') as pbar:
            for epoch in range(1, 101):
                loss = self.train(encoder_model, dataloader, optimizer, Loss)
                pbar.set_postfix({'loss': loss})
                pbar.update()

            torch.save({'model': encoder_model.state_dict()}, self.network_params_path)

    def fine_tuning(self, dataloader):
        encoder_model = Encoder(encoder=gconv)
        encoder_model.load_state_dict(torch.load(self.network_params_path, map_location=device)["model"])
        project_dim = 300 * 5
        encoder_model.encoder.project = torch.nn.Sequential(
            torch.nn.Linear(project_dim, project_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(project_dim, project_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(project_dim, 2)
        )

        encoder_model = encoder_model.to(device)

        Loss = torch.nn.CrossEntropyLoss().to(device)

        output_params = list(map(id, encoder_model.encoder.project.parameters()))
        feature_params = filter(lambda p: id(p) not in output_params, encoder_model.parameters())

        optimizer = SGD([{'params': feature_params},
                         {'params': encoder_model.encoder.project.parameters(), 'lr': 0.01}],
                        lr=0.01, weight_decay=0.001)

        with tqdm(total=100, desc='(T)') as pbar:
            for epoch in range(1, 101):
                loss = self.train_4_ft(encoder_model, dataloader, optimizer, Loss)
                pbar.set_postfix({'loss': loss})
                pbar.update()

            torch.save({'model': encoder_model.state_dict()}, self.network_params_path)

    def non_ContNet_train(self, dataloader):
        encoder_model = Encoder(encoder=gconv).to(device)
        Loss = torch.nn.CrossEntropyLoss().to(device)
        optimizer = Adam(encoder_model.parameters(), lr=0.001)

        with tqdm(total=100, desc='(T)') as pbar:
            for epoch in range(1, 101):
                loss = self.train_4_ft(encoder_model, dataloader, optimizer, Loss)
                pbar.set_postfix({'loss': loss})
                pbar.update()
            torch.save({'model': encoder_model.state_dict()}, self.network_params_path)
