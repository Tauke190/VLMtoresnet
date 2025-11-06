import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 11))

# Average losses per epoch
losses = [0.411485, 0.376155, 0.357978, 0.358970, 0.364873,
          0.356559, 0.323311, 0.333465, 0.313720, 0.328997]

# Plotting
plt.figure(figsize=(8,5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.xticks(epochs)
plt.grid(True)
plt.show()
