import torch
import torch.optim as optim
from cifar import train_utils as tutils
from torch.utils.tensorboard import SummaryWriter

def get_Hembedding(dim_emb, cur_data, pre_embs, hnet, mnet, device, num_iter=1000, tensorboard = False):
    if tensorboard:
        writer = SummaryWriter("runs/Hembedding")
    cur_dist = [get_Hscore(hnet(emb), mnet, cur_data) for emb in pre_embs]
    cur_emb = torch.rand(dim_emb, requires_grad=True).to(device)
    dist_scaling = torch.tensor(1,requires_grad=True).to(device)

    optimizer = optim.Adam([cur_emb,dist_scaling], lr=0.001)

    for i in range(num_iter):
        optimizer.zero_grad()

        loss = torch.sum(torch.abs(dist_scaling * cur_dist - torch.sqrt(torch.sum((cur_emb-pre_embs)**2,dim=1))))

        loss.backward()
        optimizer.step()

        if i%20 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}, Dist_scaling: {dist_scaling.item()}")
            if tensorboard:
                writer.add_scalar("Loss",loss.item(),i)
                writer.add_scalar("Dist_scaling",dist_scaling.item(),i)
                writer.add_histogram("Cur_emb",cur_emb,i)
        
    writer.close() if tensorboard else None
    return cur_emb    


def get_Hscore(mnet_weights, mnet, cur_data):
    from Hscore import getDiffNN
    cur_X, cur_label = cur_data['X'], cur_data['T']
    mnet_kwargs = {'return_features': True}
    cur_feature = mnet.forward(cur_X, weights=mnet_weights, **mnet_kwargs)
    score = getDiffNN(cur_feature, cur_label)
    return score

if __name__ == "__main__":
    pass
