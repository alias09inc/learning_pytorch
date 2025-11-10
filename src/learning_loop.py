

for x, y in dataloader:
    pred = model(x)
    loss = criterion(pred, y)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
"""
optimzer.zero_grad()をしないと、loss.backward()で計算した勾配が累積されてしまう。
"""