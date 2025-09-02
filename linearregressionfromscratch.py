import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LinearRegression():
    def __init__(self, lr=0.0001, max_iters=500):
        self.max_iters = max_iters
        self.lr = lr
        self.m = np.random.uniform(low=-20, high=20)
        self.b = np.random.uniform(low=-20, high=20)
    
    def forward(self, x):
        return self.m * x + self.b
    
    def cost(self, ytrue, ypred):
        return np.mean((ytrue-ypred)**2)
    
    def backprop(self, x, ytrue, ypred):
        df = (ypred - ytrue)
        db = 2 * np.mean(df)
        dm = 2 * np.mean(np.multiply(x, df))
        return db, dm

    def training_step(self, x, y):
        predictions = self.forward(x)
        loss = self.cost(y, predictions)
        db, dm = self.backprop(x, y, predictions)
        self.m = self.m - self.lr * dm
        self.b = self.b - self.lr * db
        return predictions, loss, db, dm

    def train(self, x, y):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.scatter(x,y,s=0.5)
        line, = ax1.plot(x, (self.m * x + self.b), 'r-', label="Prediction")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        
        loss_line, = ax2.plot([], [], 'b-', label="Loss")
        initial_preds = self.forward(x)
        initial_loss = self.cost(y, initial_preds)
        ax2.set_xlim(0, self.max_iters)
        ax2.set_ylim(0, initial_loss * 1.1) 
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss")

        losses = []
        grads_m = []
        grads_b = []
        pred_lines = []

        def update(frame):
            preds, loss, db, dm = self.training_step(x, y)
            losses.append(loss)
            grads_b.append(db)
            grads_m.append(dm)
            ax1.set_title(f"Iter {frame}: loss={loss:.2f}, dm={dm:.2f}, db={db:.2f}")
            loss_line.set_data(np.arange(len(losses)), losses)
            line.set_ydata(preds)
            return line, loss_line

        ani = FuncAnimation(fig, update, frames=self.max_iters, interval=100, blit=False, repeat=False)
        plt.legend() 
        plt.show()

x = np.linspace(-20, 20, 500)
y = x.copy()
noise = np.random.normal(scale=20.0,size=500)
y = y + noise

lin_reg = LinearRegression()
lin_reg.train(x,y)

