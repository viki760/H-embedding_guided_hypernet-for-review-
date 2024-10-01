
import importlib

def get_cur_embedding(config, metric, cur_data, pre_embs, hnet, mnet, device, tensorboard=False, writer=None):
    # get cur emb; need to be inserted into training process
    module = importlib.import_module(f"metrics.online_embedding.{metric}")

    get_metric = getattr(module, f"get_{metric}")

    cur_emb = get_metric(config, cur_data, pre_embs, hnet, mnet, device, tensorboard = tensorboard, writer=writer)
    return cur_emb

def get_embedding(config, metric, dhandlers, hnet, mnet, device, n_sample=1000, tensorboard=False, writer=None):
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
        # dim_label = cur_data['T'].shape[-1]
        cur_emb = get_cur_embedding(config, metric, cur_data, embeddings, hnet, mnet, device=device, tensorboard = tensorboard, writer=writer)
        embeddings.append(cur_emb)

    print(f"Embedding: {embeddings}")
    return embeddings

if __name__ == "__main__":
    import sys
    sys.path.append("/mnt/d/task/research/codes/HyperNet/hypercl/")

    from cifar import train_utils as tutils
    from cifar import train_args
    from utils import sim_utils as sutils
    from argparse import Namespace
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter
    
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
    
  
    mnet = tutils.get_main_model(config, shared, logger, device, no_weights=not config.mnet_only)
    hnet = tutils.get_hnet_model(config, mnet, logger, device)
    # with previous hnet weight
    # hnet.load_state_dict(torch.load(weights_path))

    metric = "Hembedding"
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/{metric}-{date_time_str}")

    embs = get_embedding(metric=metric, dim_emb=config, dhandlers=dhandlers, hnet=hnet, mnet=mnet, device=device, tensorboard=True, writer=writer)
    writer.close()
    