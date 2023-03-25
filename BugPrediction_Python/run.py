import os
import torch
from tqdm import tqdm
from torch.optim import Adam, SGD
import dgl
from util.MyDataLoader import MyDataset4Test, MyDataset4Train
from dgl.dataloading import GraphDataLoader
from util.Config import Config

from networks.bjCnet import Encoder, GatedGCN, MyHingeLoss, GraphConvNet

from util.Evaluation import Evaluation

epoch_loss_list = []

device = Config.device

gconv = GatedGCN(300, 7, hidden_dim=300, num_layers=5).to(device)


def train(encoder_model, dataloader, optimizer, Loss):
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

    epoch_loss_list.append(epoch_loss)
    return epoch_loss


def train_4_ft(encoder_model, dataloader, optimizer, Loss):
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

    epoch_loss_list.append(epoch_loss)
    return epoch_loss


def test(encoder_model, dataloader, model_name, for_all=False):
    encoder_model.eval()
    X = []
    y = []
    with torch.no_grad():
        for data, label in dataloader:
            g = encoder_model(data)
            g_feats = dgl.readout_nodes(g, "X")
            for g_feat in g_feats:
                out = encoder_model.encoder.project(g_feat)
                X.append(out)

            y.append(label)

    X = torch.stack(X, dim=0)
    y = torch.cat(y, dim=0)

    if for_all:
        return [X, y]

    evaluation = Evaluation(X, y, model_name)
    evaluation.evaluate_4_nn()


def pretrain(training_dataloader, testing_dataloader):
    encoder_model = Encoder(encoder=gconv).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.001)
    Loss = MyHingeLoss().to(device)

    if not os.path.exists(Config.network_params_path):
        with tqdm(total=100, desc='(T)') as pbar:
            for epoch in range(1, 101):
                loss = train(encoder_model, training_dataloader, optimizer, Loss)
                pbar.set_postfix({'loss': loss})
                pbar.update()

            network_params_path = Config.network_params_path

            torch.save({'model': encoder_model.state_dict()}, network_params_path)

        Evaluation.plot_loss(epoch_loss_list, "bjCnet_pt")

    encoder_model = Encoder(encoder=gconv).to(device)
    encoder_model.load_state_dict(torch.load(Config.network_params_path, map_location=Config.device)["model"])
    test(encoder_model, testing_dataloader, "bjCnet_pt")


def fine_tuning(dataloader):
    if not os.path.exists(Config.network_params_path_ft):
        encoder_model = Encoder(encoder=gconv)
        encoder_model.load_state_dict(torch.load(Config.network_params_path, map_location=Config.device)["model"])
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
                loss = train_4_ft(encoder_model, dataloader, optimizer, Loss)
                pbar.set_postfix({'loss': loss})
                pbar.update()

            torch.save({'model': encoder_model.state_dict()}, Config.network_params_path_ft)

        Evaluation.plot_loss(epoch_loss_list, "bjCnet_ft")

    encoder_model = Encoder(encoder=gconv).to(device)
    encoder_model.load_state_dict(torch.load(Config.network_params_path_ft, map_location=Config.device)["model"])
    test(encoder_model, dataloader, "bjCnet_ft")


def non_ContNet_train(dataloader):
    if not os.path.exists(Config.network_params_path_non_cn):
        encoder_model = Encoder(encoder=gconv).to(device)
        Loss = torch.nn.CrossEntropyLoss().to(device)
        optimizer = Adam(encoder_model.parameters(), lr=0.001)

        with tqdm(total=100, desc='(T)') as pbar:
            for epoch in range(1, 101):
                loss = train_4_ft(encoder_model, dataloader, optimizer, Loss)
                pbar.set_postfix({'loss': loss})
                pbar.update()

            torch.save({'model': encoder_model.state_dict()}, Config.network_params_path_non_cn)

        Evaluation.plot_loss(epoch_loss_list, "non_ContNet")

    encoder_model = Encoder(encoder=gconv).to(device)
    encoder_model.load_state_dict(torch.load(Config.network_params_path_non_cn, map_location=Config.device)["model"])
    test(encoder_model, dataloader, "non_ContNet")


def main():
    training_dataset = MyDataset4Train()
    testing_dataset = MyDataset4Test()

    training_data = training_dataset
    testing_data = testing_dataset

    training_dataloader = GraphDataLoader(training_data, batch_size=128, shuffle=True)
    testing_dataloader = GraphDataLoader(testing_data, batch_size=128)

    pretrain(training_dataloader, testing_dataloader)
    fine_tuning(testing_dataloader)
    non_ContNet_train(testing_dataloader)

    result_list = []
    for model_path in [Config.network_params_path, Config.network_params_path_ft, Config.network_params_path_non_cn]:
        encoder_model = Encoder(encoder=gconv).to(device)
        encoder_model.load_state_dict(torch.load(model_path, map_location=Config.device)["model"])
        result_list.append(test(encoder_model, testing_dataloader, None, True))

    Evaluation.plot_roc_4_three_model(result_list)


if __name__ == '__main__':
    main()
