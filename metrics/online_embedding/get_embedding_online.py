
import importlib

def get_cur_embedding(metric, dim_emb, cur_data, pre_embs, hnet, mnet, device):
    # get cur emb; need to be inserted into training process
    module = importlib.import_module(f"metrics.{metric}")

    get_metric = getattr(module, f"get_{metric}")

    cur_emb = get_metric(dim_emb, cur_data, pre_embs, hnet, mnet, device, num_iter=1000, tensorboard = False)
    return cur_emb

def get_embedding(metric, dim_emb, dhandlers, hnet, device,n_sample=1000):
    # get embedding for each task
    # get all embedding at once (without embedding update by hnet), for testing & baseline only
    # if n_sample == None, all data will be used for embedding measurement 


    # module = importlib.import_module(f"metrics.online_embedding.{metric}")
    # get_metric = getattr(module, f"get_{metric}")

    embeddings = []

    for dhandler in dhandlers:
        cur_data = [dhandler.get_train_inputs(), dhandler.get_train_outputs()] if n_sample is None else dhandler.next_train_batch(n_sample)
        X = dhandler.input_to_torch_tensor(cur_data[0], device, mode='train')
        T = dhandler.output_to_torch_tensor(cur_data[1], device, mode='train')
        cur_data = {'X':X,'T':T}
        dim_label = cur_data['T'].shape[-1]
        cur_emb = get_cur_embedding(metric, dim_emb, cur_data, embeddings, hnet, device=device)
        embeddings.append(cur_emb)

    return embeddings

if __name__ == "__main__":
    import sys
    sys.path.append("/mnt/d/task/research/codes/HyperNet/hypercl/")

    from cifar import train_utils as tutils
    from cifar import train_args
    from utils import sim_utils as sutils
    from argparse import Namespace
    
    DATA_DIR_CIFAR = r"/mnt/d/task/research/codes/MultiSource/wsl/2/multi-source/data/"

    ### Load datasets (i.e., create tasks).
    # Container for variables shared across function.
    shared = Namespace()
    experiment='resnet'
    shared.experiment = experiment
    config = train_args.parse_cmd_arguments(mode='resnet_cifar')
    device, writer, logger = sutils.setup_environment(config, logger_name='det_cl_cifar_%s' % experiment)
    dhandlers = tutils.load_datasets(config, shared, logger,
                                     data_dir=DATA_DIR_CIFAR)
    
    hnet = None
    dim_emb = 20
    get_embedding("Hembedding",dim_emb, dhandlers, hnet, device=device)