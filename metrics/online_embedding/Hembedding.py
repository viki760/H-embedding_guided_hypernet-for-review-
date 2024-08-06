import torch
import torch.optim as optim
import sys
sys.path.append('/mnt/d/task/research/codes/HyperNet/hypercl/metrics/online_embedding/')
from torch.utils.tensorboard import SummaryWriter

def get_Hembedding(dim_emb, cur_data, pre_embs, hnet, mnet, device, num_iter=1000, tensorboard = False, writer = None, lr = 0.001, epsilon = 0.1, mode = 'direct'):
    if tensorboard:
        assert writer is not None
    # get_Hscore(hnet.forward(task_emb=pre_embs[0]), mnet, cur_data)
    cur_dist = [get_Hscore(hnet.forward(task_emb=emb), mnet, cur_data) for emb in pre_embs]
    
    border1 = min(cur_dist)
    border2 = max(cur_dist)
    
    if mode == 'reciprocal':
        processed_cur_dist = [1/(x- border1 + epsilon) for x in cur_dist]
    elif mode == 'direct':
        processed_cur_dist = [(border2-x+epsilon+border1) for x in cur_dist]
    
    
    task_id = len(pre_embs)
    cur_emb = torch.nn.Parameter(torch.rand(dim_emb, requires_grad=True, device=device))
    dist_scaling = torch.tensor(1.0, requires_grad=True, device=device)

    if len(pre_embs) == 0:
        return cur_emb.detach()
    elif len(pre_embs) == 1:
        optimizer = optim.Adam([cur_emb], lr=lr)
    else:
        dist_scaling = torch.nn.Parameter(dist_scaling)
        optimizer = optim.Adam([cur_emb,dist_scaling], lr=lr)

    torch_cur_dist = torch.tensor(processed_cur_dist).to(device)
    pre_embs = torch.tensor(torch.stack(pre_embs, dim=0)).to(device) if pre_embs is list else pre_embs.to(device)

    for i in range(num_iter):
        optimizer.zero_grad()

        loss = torch.sum(torch.abs(dist_scaling * torch_cur_dist - torch.sqrt(torch.sum((cur_emb-pre_embs)**2,dim=1))))

        loss.backward()
        optimizer.step()

        if i%50 == 0:
            print(f"Epoch {i}, Loss: {loss.item()}, Dist_scaling: {dist_scaling.item()}")
            if tensorboard:
                writer.add_scalar(f"task_{task_id}\Loss",loss.item(),i)
                writer.add_scalar(f"task_{task_id}\Dist_scaling",dist_scaling.item(),i)
                writer.add_histogram(f"task_{task_id}\Cur_emb",cur_emb,i)
        
    print('Finished optimization for task',len(pre_embs))
    print(f"Final Loss: {loss.item()}, Dist_scaling: {dist_scaling.item()}")
    print("Cur_emb:",cur_emb)
    return cur_emb.detach()    


def get_Hscore(mnet_weights, mnet, cur_data):
    from Hscore import getDiffNN
    cur_X, cur_label = cur_data['X'], cur_data['T']
    mnet_kwargs = {'return_features': True}
    cur_feature = mnet.forward(cur_X, weights=mnet_weights, **mnet_kwargs)
    score = getDiffNN(cur_feature, cur_label)
    return score

if __name__ == "__main__":
    pass
