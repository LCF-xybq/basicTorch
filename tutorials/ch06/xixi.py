import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict
import matplotlib.pyplot as plt

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)

def preprocess():
    global t_u, t_c

    n_samples = t_u.shape[0]
    n_vals = int(0.2 * n_samples)
    shuffled_idx = torch.randperm(n_samples)
    train_idx = shuffled_idx[:-n_vals]
    val_idx = shuffled_idx[-n_vals:]
    train_u = t_u[train_idx]
    train_c = t_c[train_idx]
    val_u = t_u[val_idx]
    val_c = t_c[val_idx]

    return 0.1 * train_u, train_c, 0.1 * val_u, val_c

def model():
    return nn.Sequential(
        nn.Linear(1, 13),
        nn.Tanh(),
        nn.Linear(13, 1)
    )

def modelWithExpNames():
    return nn.Sequential(
        OrderedDict([
            ('hidden_linear', nn.Linear(1, 8)),
            ('hidden_activation', nn.Tanh()),
            ('output_linear', nn.Linear(8, 1))
        ])
    )

def printParams(model):
    for name, param in model.named_parameters():
        print(name, param.shape)

def train(n_epochs, model, optimizer, loss, train_u, train_c, val_u, val_c):
    for epoch in range(1, n_epochs + 1):
        train_p = model(train_u)
        train_loss = loss(train_p, train_c)

        val_p = model(val_u)
        val_loss = loss(val_p, val_c)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")

if __name__ == '__main__':
    train_u, train_c, val_u, val_c = preprocess()

    seq_model = modelWithExpNames()

    print(seq_model)
    print('-' * 80)
    printParams(seq_model)

    optimizer = optim.SGD(seq_model.parameters(), lr=1e-3)

    train(
        n_epochs=5000,
        model=seq_model,
        optimizer=optimizer,
        loss=nn.MSELoss(),
        train_u=train_u,
        train_c=train_c,
        val_u=val_u,
        val_c=val_c
    )

    print('-' * 80)
    print('output', seq_model(val_u))
    print('answer', val_c)
    print('hidden', seq_model.hidden_linear.weight.grad)

    # vis
    t_range = torch.arange(20., 90.).unsqueeze(1)
    fig = plt.figure(dpi=100)
    plt.xlabel("Fahrenheit")
    plt.ylabel("Celsius")
    plt.plot(t_u.numpy(), t_c.numpy(), 'o')
    # detach: requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad
    plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
    plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
    plt.show()