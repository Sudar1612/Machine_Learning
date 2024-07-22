import numpy as np
import matplotlib.pyplot as plt
import tqdm

# Our dataset is a sinusoid function turned into 200 samples each of 25-time steps.
def dataset(size = 250,timesteps = 25):
    x,y=[],[]
    sin_wave = np.sin(np.arange(size))
    for step in range(sin_wave.shape[0]-timesteps):
        x.append(sin_wave[step:step+timesteps])
        y.append(sin_wave[step+timesteps])
    return np.array(x).reshape(len(y),timesteps,1),np.array(y).reshape(len(y),1)

class RNN:
    def __init__(self,x,y,hidden_units):
        self.x = x  # inputs
        self.y = y  # output
        self.hidden_units = hidden_units
        self.Wx = np.random.randn(self.hidden_units,self.x.shape[2])
        self.Wh = np.random.randn(self.hidden_units, self.hidden_units)
        self.Wy = np.random.randn(self.y.shape[1],self.hidden_units)

    def cell(self,xt,ht_1):
        '''
        h_t = f(W^(hh) * h_(t-1) + W^(hx) * x_t)
        y_t = softmax(W^(S) * h_t)
        J^(t)(θ) = Σ_(i=1)^(|V|) y_i^(t) * log(y_i)
        :param xt: input x at time stamp t
        :param ht_1: hidden state or the word from previous state t-1
        :return: the embedding of current input -> yt , a hidden unit to next layer -> ht
        '''
        ht =np.tanh( np.dot(self.Wx,xt.reshape(1,1)) + np.dot(self.Wh,ht_1))
        yt = np.dot(self.Wy,ht)
        return ht,yt

    def forward(self,sample):
        sample_x,sample_y = self.x[sample],self.y[sample]
        ht = np.zeros((self.hidden_units,1) )#first hidden state is zeros vector
        self.hidden_states = [ht]
        self.inputs = []
        for step in range(len(sample_x)):
            ht , yt = self.cell(sample_x[step],ht)
            self.inputs.append(sample_x[step].reshape(1,1))
            self.hidden_states.append(ht)

        self.error = yt - sample_y
        self.loss = 0.5*self.error**2
        self.yt=yt

    def backward(self):
        n = len(self.inputs)
        dyt = self.error  # dL/dyt
        dWy = np.dot(dyt, self.hidden_states[-1].T)  # dyt/dWy
        dht = np.dot(dyt, self.Wy).T  # dL/dht = dL/dyt * dyt/dht ,where ht = tanh(Wx*xt + Wh*ht))
        dWx = np.zeros(self.Wx.shape)
        dWh = np.zeros(self.Wh.shape)
        # BPTT
        for step in reversed(range(n)):
            temp = (1 - self.hidden_states[
                step + 1] ** 2) * dht  # dL/dtanh = dL/dyt * dyt/dht * dht/dtanh, where dtanh = (1-ht**2)
            dWx += np.dot(temp, self.inputs[step].T)  # dL/dWx = dL/dyt * dyt/dht * dht/dtanh * dtanh/dWx
            dWh += np.dot(temp, self.hidden_states[step].T)  # dL/dWh = dL/dyt * dyt/dht * dht/dtanh * dtanh/dWh

            dht = np.dot(self.Wh, temp)  # dL/dht-1 = dL/dht * (1 - ht+1^2) * Whh
        dWy = np.clip(dWy, -1, 1)
        dWx = np.clip(dWx, -1, 1)
        dWh = np.clip(dWh, -1, 1)
        self.Wy -= self.lr * dWy
        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh

    def train(self , epochs , learning_rate):
        self.over_loss = []
        self.lr = learning_rate
        for epoch in tqdm.tqdm(range(epochs)):
            for sample in range(self.x.shape[0]):
                self.forward(sample)
                self.backward()
            self.over_loss.append(np.squeeze(self.loss/self.x.shape[0]))
            self.loss = 0

    def test(self,x,y):
        self.x = x
        self.y = y
        self.outputs = []
        for sample in range(len(x)):
            self.forward(sample)
            self.outputs.append(self.yt)

x,y=dataset()
x_test,y_test=dataset(300)
x_test = x_test[250:]
y_test = y_test[250:]

rnn=RNN(x,y,100)
rnn.train(25,1e-2)
rnn.test(x_test,y_test)

plt.tight_layout()
plt.figure(dpi=120)
plt.subplot(121)
plt.plot(rnn.over_loss)
plt.subplot(122)
plt.plot([i for i in range(len(x_test))],y_test,np.array(rnn.outputs).reshape(y_test.shape))
plt.show()


















