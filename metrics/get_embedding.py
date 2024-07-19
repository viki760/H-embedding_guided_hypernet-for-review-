import importlib
from matplotlib import pyplot as plt
class embedding:
    def __init__(self, metric, mode="offline"):

        self.get_metric = getattr(importlib.import_module(f"metrics.{metric}"), f"get_{metric}")

        if mode == "offline":
            self.get_emb_fun = getattr(importlib.import_module(f"metrics.offline_embedding.get_embedding_offline"), "get_embedding")
        elif mode == "online":
            self.get_emb_fun = getattr(importlib.import_module(f"metrics.online_embedding.get_embedding_online"), "get_embedding")
        else:
            raise ValueError("mode must be either 'offline' or 'online'")        
    
    def get_embedding(self, *args, **kwargs):
        self.embedding = self.get_emb_fun(self.get_metric, *args, **kwargs)
        return self.embedding
    
    def display(self, compress_mode = None):
        print(self.embedding)
        if compress_mode == "PCA":
            pass
        elif compress_mode == "TSNE":
            pass
        else:
            plt.imshow(self.embedding)
            plt.show()

    def get_distance(self, *args, **kwargs):
        pass
    
if __name__ == "__main__":
    WTE_emb = embedding("WTE", mode="offline")
    WTE_emb.get_embedding()
    H_emb = embedding("Hscore", mode="online")
