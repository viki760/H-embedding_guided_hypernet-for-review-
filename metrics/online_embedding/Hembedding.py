import torch
import torch.optim as optim
import sys
sys.path.append('/path/to/working/directory/metrics/online_embedding/')
from torch.utils.tensorboard import SummaryWriter
import gc

def get_Hembedding(config, cur_data, pre_embs, hnet, mnet, device, tensorboard = False, writer = None):
    if tensorboard:
        assert writer is not None
    # get_Hscore(hnet.forward(task_emb=pre_embs[0]), mnet, cur_data)
    cur_dist = []
    for emb in pre_embs:
        with torch.no_grad():
            hscore = get_Hscore(hnet.forward(task_emb=emb), mnet, cur_data)
        gc.collect()
        torch.cuda.empty_cache()
        cur_dist.append(hscore)
    # cur_dist = [get_Hscore(hnet.forward(task_emb=emb), mnet, cur_data) for emb in pre_embs]
    
    border1 = min(cur_dist)
    border2 = max(cur_dist)
    
    if config.emb_mode == 'reciprocal':
        processed_cur_dist = [1/(x- border1 + config.emb_epsilon) for x in cur_dist]
    elif config.emb_mode == 'direct':
        processed_cur_dist = [(border2-x+config.emb_epsilon+border1) for x in cur_dist]
    
    
    task_id = len(pre_embs)
    cur_emb = torch.nn.Parameter(torch.rand(config.temb_size, requires_grad=True, device=device))
    dist_scaling = torch.tensor(1.0, requires_grad=True, device=device)

    if len(pre_embs) == 0:
        return cur_emb.detach()
    elif len(pre_embs) == 1:
        optimizer = optim.Adam([cur_emb], lr=config.emb_lr)
    else:
        dist_scaling = torch.nn.Parameter(dist_scaling)
        optimizer = optim.Adam([cur_emb,dist_scaling], lr=config.emb_lr)

    torch_cur_dist = torch.tensor(processed_cur_dist).to(device)
    pre_embs = torch.tensor(torch.stack(pre_embs, dim=0)).to(device) if pre_embs is list else pre_embs.to(device)

    for i in range(config.emb_num_iter):
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
        
    print('Finished optimization for task',len(pre_embs)+1)
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
