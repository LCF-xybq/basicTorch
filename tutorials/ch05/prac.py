import torch
import torch.optim as optim


t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

t_u = torch.tensor(t_u)
t_c = torch.tensor(t_c)

lr = 1e-2
params = torch.tensor([1.0, 0.0], requires_grad=True)
optimizer = optim.SGD([params], lr=lr)

# data
n_samples = t_u.shape[0]
shuffle_indexs = torch.randperm(n_samples)
n_val = int(0.2 * n_samples)

train_idx = shuffle_indexs[:-n_val]
val_idx = shuffle_indexs[-n_val:]

train_u = t_u[train_idx]
train_c = t_c[train_idx]
val_u = t_u[val_idx]
val_c = t_c[val_idx]

train_u = 0.1 * train_u
val_u = 0.1 * val_u


def model(t_u, w, b):
    res = t_u * w + b
    return res

def loss_fn(t_p, t_c):
    diff = (t_p - t_c) ** 2
    return diff.mean()

def train_loop(n_epochs, optimizer, params, train_u, val_u, train_c, val_c):
    for epoch in range(1, n_epochs + 1):
        train_p = model(train_u, *params)
        train_loss = loss_fn(train_p, train_c)

        with torch.no_grad():
            val_p = model(val_u, *params)
            val_loss = loss_fn(val_p, val_c)
            assert val_loss.requires_grad == False

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch < 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Val loss {val_loss.item():.4f}")

    return params

def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_p = model(train_t_u, *params)
        train_loss = loss_fn(train_p, train_t_c)

        with torch.no_grad():
            val_p = model(val_t_u, *params)
            val_loss = loss_fn(val_p, val_t_c)
            assert val_loss.requires_grad == False

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch < 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Val loss {val_loss.item():.4f}")

    return params

train_loop(
    n_epochs=5000,
    optimizer=optimizer,
    params = params,
    train_u=train_u,
    val_u=val_u,
    train_c=train_c,
    val_c=val_c
)
