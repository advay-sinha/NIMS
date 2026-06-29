import torch
import torch.nn as nn
import time

device = torch.device("cuda")

model = nn.Sequential(
    nn.Linear(1000, 2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 100)
).to(device)

x = torch.randn(4096, 1000, device=device)
y = torch.randn(4096, 100, device=device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

start = time.time()

for i in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

print(f"Training Time: {time.time()-start:.2f}s")
print("GPU:", torch.cuda.get_device_name(0))
print("GPU Memory Allocated:",
      round(torch.cuda.memory_allocated()/1024**2,2), "MB")