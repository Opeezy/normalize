import numpy as np
import logging

from numpy import ndarray

logging.basicConfig(level="INFO")

class Normalize():
    def __init__(self,
                 data: ndarray,
                 value: int = 1) -> ndarray:
        self.ORIGINAL = data
        self.data = data
        self.min = min(data)
        self.max = max(data)
        self.value = value

        self.is_normalized = False

    def revert(self) -> None:
        if self.is_normalized:
            for ikey, i in enumerate(self.data):
                self.data[ikey] = i * (self.max-self.min) + self.min
        else:
            logging.error("Data hasn't been normalized")
        self.is_normalized = False

    def to_normal(self) -> None:
        if not self.is_normalized:
            for ikey, i in enumerate(self.data):
                self.data[ikey] = (i - self.min) / (self.max - self.min) * self.value
        else:
            logging.error("Data is already normalized")
        self.is_normalized = True

if __name__ == "__main__":
    data = np.random.uniform(20000.00, 30000.00, (9000,3))
    x = data[:, 0]
    print(f"{x[0:10]}\n")

    n_x = Normalize(x, 100)
    n_x.to_normal()
    print(f"{n_x.data[0:10]}\n")
    n_x.revert()
    print(f"{n_x.data[0:10]}\n")
