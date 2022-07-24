import torch

from torch_geometric.datasets import Planetoid

from models import GCN


def compute_acc(logits, label):
    # 获取每个节点的logits中最大数值对应的类别，即模型预测的概率最大的类别
    prediction = logits.max(dim=-1)[1]
    accuarcy = torch.eq(prediction, label).float().mean()
    return accuarcy


if __name__ == '__main__':
    # 定义相关的超参数
    learning_rate = 0.1
    weight_decay = 5e-4
    epochs = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 16
    # 从GCN，GAT和SAGE中选择一个
    conv_type = 'GCN'

    ################################################
    # 加载数据集，数据集可以从Cora，CiteSeer和PubMed中选择
    data_name = 'Cora'
    dataset = Planetoid('./data', data_name)
    # 一个数据集一般是一组图的集合，但许多场景下我们的数据仅需要一张大图进行表示
    data = dataset[0].to(device)
    feature_dim = dataset.num_features
    out_dim = dataset.num_classes

    ################################################
    # 加载模型
    model = GCN(feature_dim, hidden_dim, out_dim, conv_type=conv_type).to(device)

    # 损失函数使用交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #################################################
    # 开始训练
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for i in range(epochs):
        # 半监督模式，前向传播获得图上所有节点的分类结果，包括训练，验证和测试的节点。
        logits = model(data.x, data.edge_index)
        # 半监督模式，获得表示/预测结果时利用整图，但进行训练时仅选择训练节点构造loss。半监督模式训练时仅需要有限（这里是140）个训练节点的label。
        train_logits = logits[data.train_mask]
        # 获取训练节点的对应标签
        train_y = data.y[data.train_mask]
        # 计算loss
        loss = criterion(train_logits, train_y)
        # 清楚优化器中上一步的梯度
        optimizer.zero_grad()
        # 反向传播，计算参数当前步梯度
        loss.backward()
        # 根据梯度更新参数
        optimizer.step()
        # 计算acc
        train_acc = compute_acc(train_logits, train_y)

        val_logits, val_label = logits[data.val_mask], data.y[data.val_mask]
        val_acc = compute_acc(val_logits, val_label)

        loss_history.append(loss.item())
        train_acc_history.append(train_acc.item())
        val_acc_history.append(val_acc.item())
        print("Epoch: {:03d}: Loss {:.4f}, TrainAcc {:.4f}, ValAcc {:.4f}".format(i, loss.item(), train_acc.item(),
                                                                                  val_acc.item()))

    # 完成所有的训练epoch，计算最新模型下的准确率
    with torch.no_grad():
        # 获取当前模型下的所有logits
        logits = model(data.x, data.edge_index)
        # 计算训练样本的预测准确率
        train_logits, train_label = logits[data.train_mask], data.y[data.train_mask]
        train_acc = compute_acc(train_logits, train_label)
        print("Train accuarcy: {:.4f}".format(train_acc.item()))
        # 计算验证样本的预测准确率
        val_logits, val_label = logits[data.val_mask], data.y[data.val_mask]
        val_acc = compute_acc(val_logits, val_label)
        print("Val accuarcy: {:.4f}".format(val_acc.item()))
        # 计算测试样本的预测准确率
        test_logits, test_label = logits[data.test_mask], data.y[data.test_mask]
        test_acc = compute_acc(test_logits, test_label)
        print("Test accuarcy: {:.4f}".format(test_acc.item()))

    # 保存模型
    SAVE_MODEL = False
    if SAVE_MODEL:
        model_name = "{}_{}_{}.pt".format(conv_type, hidden_dim, data_name)
        torch.save(model, model_name)
