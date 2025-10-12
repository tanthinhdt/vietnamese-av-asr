import numpy as np
import matplotlib.pyplot as plt


# Define Data
class Data:
    def __init__(self):
        self.samples = (
            dict({'x': 1, 'y': 2}),
            dict({'x': 2, 'y': 4}),
            dict({'x': 3, 'y': 6})
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Model:
    def __init__(self):
        self.w1: float = np.random.rand()
        self.w2: float = np.random.rand()
        self.b: float = 0

    def __call__(
            self,
            x,
            *args,
            **kwargs
    ):
        y_hat = pow(x, 2)*self.w2 + x*self.w1 + self.b
        return y_hat

    def gradient(
            self,
            y_hat,
            y,
            x,
    ):
        return {
            'grad_w1': 2 * (y_hat - y) * x,
            'grad_w2': 2 * (y_hat - y) * pow(x, 2),
            'grad_b': 2 * (y_hat - y)
        }

    def learn(
            self,
            grad: dict,
            learning_rate: float = 0.001,
    ):
        self.w1 = self.w1 - learning_rate * grad['grad_w1']
        self.w2 = self.w2 - learning_rate * grad['grad_w2']
        self.b = self.b - learning_rate * grad['grad_b']

class Loss:
    def __init__(self):
        self.loss = None

    def __call__(
            self,
            y,
            y_hat,
    ):
        self.loss = pow(y_hat-y, 2)
        return self.loss


data = Data()
model = Model()
loss_fn = Loss()
epoch = 50
for i in range(epoch):
    for sample in data.samples:
        x, y = sample.values()
        y_hat = model(x)
        loss_value = loss_fn(y=y, y_hat=y_hat)
        grad = model.gradient(y_hat=y_hat, x=x, y=y)
        model.learn(grad=grad, learning_rate=0.012)

xs = [d['x'] for d in data.samples]
ys = [d['y'] for d in data.samples]
y_hats = [model(x=x_) for x_ in xs]

def plot(
        x,
        y,
        y_hat
):
    fig = plt.figure(figsize=(16,9))
    plt.plot(x, y, label='target')
    plt.plot(x, y_hat, label='prediction')
    plt.legend()
    plt.title("Plot targets and predictions.")
    plt.show()

plot(xs,ys,y_hats)