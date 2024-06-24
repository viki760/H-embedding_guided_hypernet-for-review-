import importlib

def sample_data(n_i,dhandlers,n_sample,device):
    '''
    sample `n_sample` data for each of the first `n_i` tasks from the `dhandlers` 
    '''
    sampled_data = []
    for i in range(n_i):
        data = dhandlers[i].next_train_batch(n_sample)
        X = dhandlers[i].input_to_torch_tensor(data[0], device, mode='train')
        T = dhandlers[i].output_to_torch_tensor(data[1], device, mode='train')
        sampled_data.append({'X':X,'T':T})
    return sampled_data



def get_embedding(metric, dhandlers, n_sample, device):

    module = importlib.import_module(f"metrics.{metric}")

    get_metric = getattr(module, f"get_{metric}")

    sampled_data = sample_data(n_i = len(dhandlers), dhandlers=dhandlers, n_sample=n_sample, device=device)
    dim_label = sampled_data[0]['T'].shape[-1]
    embeddings = get_metric(sampled_data, label_dim = dim_label, device=device)
    return embeddings



def get_distance(metric, dhandlers, n_sample, device):
    pass

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
    get_embedding("WTE",dhandlers,n_sample=100,device=device)