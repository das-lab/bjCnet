import platform
import torch
import os

class Config:
    device = torch.device('cuda:0')

    product = ""
    raw_data_path = r"" # PROMISE dataset's dir
    processed_data_path = r"XXX\processed\source_code\whole_code"
    source_code_path = os.path.join(raw_data_path,r"\source code\{}".format(product))
    graph_path = r"XXX\processed\graph\msg\{}".format(product)
    dot_path = r"XXX\processed\graph\dot_ast\{}".format(product)
    graph_map_path = r"XXX\processed\graph_map.txt"
    processed_msg_path = r"XXX\processed\graph\processed_msg"
    network_params_path = r"XXX\network_param\bjCnet.pth"
    network_params_path_ft = r"XXX\network_param\bjCnet_ft.pth"
    network_params_path_non_cn = r"XXX\network_param\non_ContNet.pth"
    loss_path = r"XXX\loss_figure\loss_{}.png"
    roc_path = r"XXX\loss_figure\roc.png"