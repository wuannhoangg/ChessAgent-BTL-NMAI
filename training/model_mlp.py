# training/model_mlp.py
import numpy as np
try:
    import cupy as cp
    xp = cp
    print("[INFO] Đang sử dụng GPU (CuPy) cho tính toán thủ công.")
except ImportError:
    xp = np
    print("[INFO] Đang sử dụng CPU (NumPy). Cài cupy để chạy GPU.")


class LinearLayer:
    def __init__(self, input_size, output_size):
        self.W = xp.random.randn(output_size, input_size) * xp.sqrt(2. / input_size)
        
        self.b = xp.zeros((output_size, 1))
        
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        return xp.dot(self.W, x) + self.b

    def backward(self, dZ):
        m = self.x.shape[1]
        
        self.dW = (1 / m) * xp.dot(dZ, self.x.T)
        
        dX = xp.dot(self.W.T, dZ)
        return dX

    def update(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class ReLULayer:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return xp.maximum(0, Z)

    def backward(self, dA):
        dZ = xp.array(dA, copy=True)
        dZ[self.Z <= 0] = 0
        return dZ


class TanhLayer:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        self.A = xp.tanh(Z)
        return self.A

    def backward(self, dA):
        return dA * (1 - self.A**2)


class ChessMLP_Scratch:
    def __init__(self):
        self.layers = [
            LinearLayer(832, 1024),
            ReLULayer(),
            LinearLayer(1024, 512),
            ReLULayer(),
            LinearLayer(512, 1),
            TanhLayer()  # Tanh ở cuối để map giá trị về [-1, 1]
        ]

    def forward(self, X):
        if X.ndim == 4:
            batch_size = X.shape[0]
            # Flatten: (B, 13, 8, 8) -> (B, 832), rồi transpose -> (832, B)
            X_flat = X.reshape(batch_size, -1).T
        else:
            # Trường hợp X đã là (Features, Batch) thì giữ nguyên
            X_flat = X
        
        out = X_flat
        for layer in self.layers:
            out = layer.forward(out)
        
        # out shape: (1, Batch_Size)
        return out

    def backward(self, predictions, targets):
        # Đảm bảo targets có shape (1, Batch) để broadcast đúng
        targets = targets.reshape(1, -1)
        
        # Gradient ban đầu truyền vào từ Loss
        dA = (predictions - targets)
        
        grad = dA
        for layer in reversed(self.layers):
            # Chỉ các layer có backward mới được gọi (Linear, ReLU, Tanh)
            if isinstance(layer, (LinearLayer, ReLULayer, TanhLayer)):
                grad = layer.backward(grad)
    
    def update(self, lr):
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                layer.update(lr)
                
    def save_weights(self, path):
        weights = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                if xp.__name__ == 'cupy':
                    weights[f'W{i}'] = xp.asnumpy(layer.W)
                    weights[f'b{i}'] = xp.asnumpy(layer.b)
                else:
                    weights[f'W{i}'] = layer.W
                    weights[f'b{i}'] = layer.b
        np.savez(path, **weights)
        
    def load_weights(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                layer.W = xp.asarray(data[f'W{i}'])
                layer.b = xp.asarray(data[f'b{i}'])

def to_device(x):
    return xp.asarray(x)
