#!/usr/bin/env python3

"""
Training of a deterministic CL hypernetwork on CIFAR-10/100
-----------------------------------------------------------

The module :mod:`cifar.train` implements the training of a deterministic
hypernet to solve a CIFAR continual learning problem.

.. note::
    The module is not executable! Please refer to :mod:`cifar.train_resnet` or
    :mod:`cifar.train_zenke`.
"""
import sys
sys.path.append('/mnt/d/task/research/codes/HyperNet/hypercl/')

from argparse import Namespace
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from time import time

from cifar import train_utils as tutils
from mnets.classifier_interface import Classifier
from utils import sim_utils as sutils
import utils.optim_step as opstep
import utils.hnet_regularizer as hreg
from utils.torch_utils import get_optimizer
from metrics.online_embedding import Hembedding as Hemb
from train.fc_decoder import EmbDecoder


DATA_DIR_CIFAR = r"/mnt/d/task/research/codes/MultiSource/wsl/2/multi-source/data/"

def test(task_id, data, mnet, hnet, device, shared, config, writer, logger,
         train_iter=None, task_emb=None, cl_scenario=1, test_size=None):
    """Evaluate the current performance using the test set.

    Note:
        The hypernetwork ``hnet`` may be ``None``, in which case it is assumed
        that the main network ``mnet`` has internal weights.

    Args:
        task_id (int): The index of the task on which we train.
        data: The dataset handler.
        mnet: The model of the main network.
        hnet: The model of the hyper network. May be ``None``.
        device: Torch device (cpu or gpu).
        shared (argparse.Namespace): Set of variables shared between functions.
        config: The command line arguments.
        writer: The tensorboard summary writer.
        logger: The logger that should be used rather than the print method.
        train_iter (int, optional): The current training iteration. If given, it
            is used for tensorboard logging.
        task_emb (torch.Tensor, optional): Task embedding. If given, no task ID
            will be provided to the hypernetwork. This might be useful if the
            performance of other than the trained task embeddings should be
            tested.
            .. note::
                This option may only be used for ``cl_scenario=1``. It doesn't
                make sense if the task ID has to be inferred.
        cl_scenario (int, optional): In case the system should be tested on
            another CL scenario than the one user-defined in ``config``.
            .. note::
                It is up to the user to ensure that the CL scnearios are
                compatible in this implementation.
        test_size (int, optional): In case the testing shouldn't be performed
            on the entire test set, this option can be used to specify the
            number of test samples to be used.

    Returns:
        (tuple): Tuple containing:

        - **test_acc**: Test accuracy on classification task.
        - **task_acc**: Task prediction accuracy (always 100% for **CL1**).
    """

    mnet.eval()
    if hnet is not None:
        hnet.eval()

    logger.info('### Test run ...') if train_iter is None else logger.info(f'# Testing network before running training step {train_iter} ...')


    # Determine the batch statistics to use for the main network, especially when using batch normalization.
    mnet_kwargs = {}

    # Check if the model uses batch normalization layers
    if mnet.batchnorm_layers is not None:
        # If running stats and stats checkpointing are enabled, set the condition for selecting the correct stats
        if not config.bn_no_running_stats and not config.bn_no_stats_checkpointing:
            mnet_kwargs['condition'] = task_id  # Use current task ID to select correct running stats
            # Warn if task embeddings are used, which might not correspond to the task ID
            if task_emb is not None:
                logger.warning(f'Using batch statistics for task {task_id}, but testing with a specific task embedding.')

    with torch.no_grad():
        batch_size = config.val_batch_size
        #* Assuming all output heads have the same size.
        n_head = data.num_classes

        if test_size is None or test_size >= data.num_test_samples:
            test_size = data.num_test_samples
        else:
            # Make sure that we always use the same test samples.
            data.reset_batch_generator(train=False, test=True, val=False)
            logger.info(f'Only {test_size}/{data.num_test_samples} of test set is used for this test run.')

        test_loss = 0.0

        # Store all predicted labels while going over individual test batches.
        correct_labels = np.empty(test_size, np.int64)
        pred_labels = np.empty(test_size, np.int64)

        N_processed = 0

        # Sweep through the test set.
        #TODO Use torch dataloader？
        while N_processed < test_size:
            cur_batch_size = min(test_size - N_processed, batch_size)
            N_processed += cur_batch_size

            batch = data.next_test_batch(cur_batch_size)
            X = data.input_to_torch_tensor(batch[0], device)
            T = data.output_to_torch_tensor(batch[1], device)

            correct_labels[N_processed-cur_batch_size:N_processed] = T.argmax(dim=1, keepdim=False).cpu().numpy()

            ############################
            ### Get main net weights ###
            ############################
            if hnet is None:
                weights = None
            elif task_emb is not None:
                #* Fix Task Embedding
                weights = hnet.forward(task_emb=task_emb)
            else:
                weights = hnet.forward(task_id=task_id)

            #######################
            ### Get predictions ###
            #######################
            Y_hat_logits = mnet.forward(X, weights=weights, **mnet_kwargs)

            # Select current head
            task_out = [task_id*n_head, (task_id+1)*n_head]

            #? Any other solution to avoid this?
            #TODO scenario with unknown task_id
            Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]
            # Take softmax after the output neurons are chosen.
            Y_hat = F.softmax(Y_hat_logits, dim=1).cpu().numpy()
            pred_labels[N_processed-cur_batch_size:N_processed] = Y_hat.argmax(axis=1)

            # labels are 1-hot encoded.
            test_loss += Classifier.logit_cross_entropy_loss(Y_hat_logits, T,
                                                             reduction='sum')

        class_n_correct = (correct_labels == pred_labels).sum()
        test_acc = 100.0 * class_n_correct / test_size

        test_loss /= test_size

        task_info = f" (before training iteration {train_iter})" if train_iter is not None else ""
        task_emb_info = " (using a given task embedding)" if task_emb is not None else ""

        msg = f"### Test accuracy of task {task_id + 1}{task_info}: {test_acc:.3f}{task_emb_info}"

        logger.info(msg)

        if train_iter is not None:
            writer.add_scalar('test/task_%d/class_accuracy' % task_id, test_acc, train_iter)
            writer.add_scalar('test/task_%d/test_loss' % task_id, test_loss, train_iter)

        return test_acc

def train(task_id, data, mnet, hnet, device, config, shared, writer, logger):
    """Train the hyper network using the task-specific loss plus a regularizer
    that should overcome catastrophic forgetting.

    :code:`loss = task_loss + beta * regularizer`.

    Args:
        task_id: The index of the task on which we train.
        data: The dataset handler.
        mnet: The model of the main network.
        hnet: The model of the hyper network. May be ``None``.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        shared (argparse.Namespace): Set of variables shared between functions.
        writer: The tensorboard summary writer.
        logger: The logger that should be used rather than the print method.
    """
    start_time = time()

    logger.info('Training network ...')

    #################
    ### Optimizer ###
    #################
    # Define the optimizers used to train main network and hypernet.
    #TODO: check the forward of Hnet class
    #! TODO: add decoder for hidden space
    #* Add all previous task embedding to parameters but only get emb optimizer for current embedding
    mnet.train()
    if hnet is not None:
        hnet.train()
        theta_params = list(hnet.theta)

        if config.continue_emb_training:
            for i in range(task_id): # for all previous task embeddings
                theta_params.append(hnet.get_task_emb(i))

        # Only for the current task embedding.
        # Important that this embedding is in a different optimizer in case we use the lookahead.
        emb_optimizer = get_optimizer([hnet.get_task_emb(task_id)],
            config.lr, momentum=config.momentum,
            weight_decay=config.weight_decay, use_adam=config.use_adam,
            adam_beta1=config.adam_beta1, use_rmsprop=config.use_rmsprop)
    else:
        theta_params = mnet.weights
        emb_optimizer = None

    theta_optimizer = get_optimizer(theta_params, config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay,
        use_adam=config.use_adam, adam_beta1=config.adam_beta1,
        use_rmsprop=config.use_rmsprop)

    ################################
    ### Learning rate schedulers ###
    ################################
    if config.plateau_lr_scheduler:
        assert(config.epochs != -1)
        # The scheduler config has been taken from https://keras.io/examples/cifar10_resnet/
        # We use 'max' instead of 'min' as we look at accuracy rather than validation loss!
        plateau_scheduler_theta = optim.lr_scheduler.ReduceLROnPlateau( \
            theta_optimizer, 'max', factor=np.sqrt(0.1), patience=5,
            min_lr=0.5e-6, cooldown=0)
        plateau_scheduler_emb = None
        if emb_optimizer is not None:
            plateau_scheduler_emb = optim.lr_scheduler.ReduceLROnPlateau( \
                emb_optimizer, 'max', factor=np.sqrt(0.1), patience=5,
                min_lr=0.5e-6, cooldown=0)

    if config.lambda_lr_scheduler:
        assert(config.epochs != -1)

        def lambda_lr(epoch):
            """Multiplicative Factor for Learning Rate Schedule.

            Computes a multiplicative factor for the initial learning rate based
            on the current epoch. This method can be used as argument
            ``lr_lambda`` of class :class:`torch.optim.lr_scheduler.LambdaLR`.

            The schedule is inspired by the Resnet CIFAR-10 schedule suggested
            here https://keras.io/examples/cifar10_resnet/.

            In: epoch (int): The number of epochs -> Out: lr_scale (float32): learning rate scale
            """
            lr_scale = 1.
            if epoch > 180:
                lr_scale = 0.5e-3
            elif epoch > 160:
                lr_scale = 1e-3
            elif epoch > 120:
                lr_scale = 1e-2
            elif epoch > 80:
                lr_scale = 1e-1
            return lr_scale

        lambda_scheduler_theta = optim.lr_scheduler.LambdaLR(theta_optimizer, lambda_lr)
        lambda_scheduler_emb = None
        if emb_optimizer is not None:
            lambda_scheduler_emb = optim.lr_scheduler.LambdaLR(emb_optimizer, lambda_lr)

    ##############################
    ### Prepare CL Regularizer ###
    ##############################
    # Whether to calculate the regularizer
    cl_reg = task_id > 0 and not config.mnet_only and config.beta > 0 and not config.train_from_scratch
    emb_reg = task_id > 1 and config.emb_reg

    # Compute targets when the reg is activated
    if cl_reg:
        if config.online_target_computation: #* default False
            # Compute targets for the regularizer whenever they are needed. -> Computationally expensive.
            targets_hypernet = None
            prev_theta = [p.detach().clone() for p in hnet.theta]
            prev_task_embs = [p.detach().clone() for p in hnet.get_task_embs()]
        else:
            # Compute targets for the regularizer once and keep them all in memory -> Memory expensive.
            targets_hypernet = hreg.get_current_targets(task_id, hnet)
            prev_theta = None
            prev_task_embs = None

        # If we do not want to regularize all outputs (in a multi-head setup).
        # Note, we don't care whether output heads other than the current one change.
        n_y = data.num_classes
        regged_outputs = [list(range(i*n_y, (i+1)*n_y)) for i in
                            range(task_id)]
    
    ###########################################################
    ### Prepare Task Embedding for Embedding Regularization ###
    ###########################################################
    if emb_reg:
        emb_data_np = data.next_train_batch(config.emb_data_size)
        emb_data = {'X':torch.from_numpy(emb_data_np[0]).float().to(device), 'T':torch.from_numpy(emb_data_np[1]).float().to(device)}
        data.reset_batch_generator(train=True,val=False,test=False)

        if config.emb_metric == 'Hembedding':
            #! not tested
            prev_emb = torch.stack([hnet.get_task_emb(i).detach() for i in range(task_id)])
            guide_emb = Hemb.get_Hembedding(dim_emb=config.emb_size, cur_data=emb_data, pre_embs=prev_emb, hnet=hnet, mnet=mnet, device=device, num_iter=config.emb_num_iter, tensorboard=False)

        hidden_dim = hnet.get_hidden_dim()
        logger.info('Hidden dim for task %d: %s' % (task_id, str(hnet.get_hidden_dim(size_only=False))))
        decoder = EmbDecoder(hidden_dim=hidden_dim, emb_dim=config.emb_size).to(device)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr)
        

    # We need to tell the main network, which batch statistics to use, in case batchnorm is used and we checkpoint the batchnorm stats.
    mnet_kwargs = {}
    if mnet.batchnorm_layers is not None:
        if not config.bn_no_running_stats and not config.bn_no_stats_checkpointing:
            # Specify current task as condition to select correct running stats.
            mnet_kwargs['condition'] = task_id

    ######################
    ### Start training ###
    ######################

    iter_per_epoch = -1
    if config.epochs == -1:
        training_iterations = config.n_iter
    else:
        assert(config.epochs > 0)
        iter_per_epoch = int(np.ceil(data.num_train_samples / config.batch_size))
        training_iterations = config.epochs * iter_per_epoch

    logger.info('Epochs per task: %d ...' % config.epochs)
    logger.info('Iters per task: %d ...' % training_iterations)

    summed_iter_runtime = 0

    for i in range(training_iterations):

        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            test(task_id, data, mnet, hnet, device, shared, config, writer,
                 logger, train_iter=i)
            mnet.train()
            if hnet is not None:
                hnet.train()

        if i % 200 == 0:
            logger.info('Training step: %d ...' % i)

        ### Start training ###
        iter_start_time = time()

        theta_optimizer.zero_grad()
        if emb_optimizer is not None:
            emb_optimizer.zero_grad()

        #######################################
        ### Data for current task and batch ###
        #######################################
        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        # Get the output neurons depending on the continual learning scenario.
        n_y = data.num_classes

        ########################
        ### Loss computation ###
        ########################
        if config.mnet_only:
            weights = None
        else:
            weights = hnet.forward(task_id=task_id)
        Y_hat_logits = mnet.forward(X, weights, **mnet_kwargs)

        # Choose current head and restrict output neurons
        task_out = [task_id*n_y, (task_id+1)*n_y]
        Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]
        assert(T.shape[1] == Y_hat_logits.shape[1]) # Testing for output shape

        # Compute loss on task and compute gradients
        if config.soft_targets:
            soft_label = 0.95
            num_classes = data.num_classes
            soft_targets = torch.where(T == 1,
                torch.Tensor([soft_label]),
                torch.Tensor([(1 - soft_label) / (num_classes-1)]))
            soft_targets = soft_targets.to(device)
            loss_task = Classifier.softmax_and_cross_entropy(Y_hat_logits, soft_targets)
        else:
            loss_task = Classifier.logit_cross_entropy_loss(Y_hat_logits, T)

        # Compute gradients based on task loss (those used in the CL regularizer).
        loss_task.backward(retain_graph=cl_reg, create_graph=cl_reg and config.backprop_dt)

        # The current task embedding only depends in the task & emb loss, so we can update it already.
        #! TODO add decoder loss
        if emb_reg:
            decoder.train()
            decoder_optimizer.zero_grad()
            hidden_emb = hnet.forward(task_id=task_id, emb_reg=True).view(-1) #! flattened to fit into FC (may be modified in later trials)
            decode_emb = decoder.forward(hidden_emb)

            cosine_sim = F.cosine_similarity(decode_emb, guide_emb, dim=0)
            loss_emb = 1 - cosine_sim

            loss_emb.backward()
            decoder_optimizer.step()

        if emb_optimizer is not None:
            emb_optimizer.step()

        #############################
        ### CL (HNET) Regularizer ###
        #############################
        loss_reg = 0
        dTheta = None

        if cl_reg:
            if config.no_lookahead:
                dTembs = None
                dTheta = None
            else:
                dTheta = opstep.calc_delta_theta(theta_optimizer, False,
                    lr=config.lr, detach_dt=not config.backprop_dt)

                if config.continue_emb_training:
                    dTembs = dTheta[-task_id:]
                    dTheta = dTheta[:-task_id]
                else:
                    dTembs = None

            loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                targets=targets_hypernet, dTheta=dTheta, dTembs=dTembs,
                mnet=mnet, inds_of_out_heads=regged_outputs,
                prev_theta=prev_theta, prev_task_embs=prev_task_embs,
                batch_size=config.cl_reg_batch_size)

            loss_reg *= config.beta

            loss_reg.backward()

        # Now, that we computed the regularizer, we can use the accumulated
        # gradients and update the hnet (or mnet) parameters.
        theta_optimizer.step()

        Y_hat = F.softmax(Y_hat_logits, dim=1)
        classifier_accuracy = Classifier.accuracy(Y_hat, T) * 100.0

        #########################
        # Learning rate scheduler
        #########################
        if config.plateau_lr_scheduler:
            assert(iter_per_epoch != -1)
            if i % iter_per_epoch == 0 and i > 0:
                curr_epoch = i // iter_per_epoch
                logger.info('Computing test accuracy for plateau LR ' +
                            'scheduler (epoch %d).' % curr_epoch)
                # We need a validation quantity for the plateau LR scheduler.
                # FIXME we should use an actual validation set rather than the
                # test set.
                # Note, https://keras.io/examples/cifar10_resnet/ uses the test
                # set to compute the validation loss. We use the "validation"
                # accuracy instead.
                # FIXME We increase `train_iter` as the print messages in the
                # test method suggest that the testing has been executed before
                test_acc, _ = test(task_id, data, mnet, hnet, device, shared,
                                   config, writer, logger, train_iter=i+1)
                mnet.train()
                if hnet is not None:
                    hnet.train()

                plateau_scheduler_theta.step(test_acc)
                if plateau_scheduler_emb is not None:
                    plateau_scheduler_emb.step(test_acc)

        if config.lambda_lr_scheduler:
            assert(iter_per_epoch != -1)
            if i % iter_per_epoch == 0 and i > 0:
                curr_epoch = i // iter_per_epoch
                logger.info('Applying Lambda LR scheduler (epoch %d).'
                            % curr_epoch)

                lambda_scheduler_theta.step()
                if lambda_scheduler_emb is not None:
                        lambda_scheduler_emb.step()

        ###########################
        ### Tensorboard summary ###
        ###########################
        # We don't wanna slow down training by having too much output.
        if i % 50 == 0:
            writer.add_scalar('train/task_%d/class_accuracy' % task_id, classifier_accuracy, i)
            writer.add_scalar('train/task_%d/loss_task' % task_id, loss_task, i)
            writer.add_scalar('train/task_%d/loss_reg' % task_id, loss_reg, i)

        ### Show the current training progress to the user.
        if i % config.val_iter == 0:
            msg = 'Training step {}: Classifier Accuracy: {:.3f} (on current training batch).'
            logger.debug(msg.format(i, classifier_accuracy))

        iter_end_time = time()
        summed_iter_runtime += (iter_end_time - iter_start_time)

        if i % 200 == 0:
            logger.info('Training step: %d ... Done -- (runtime: %f sec)' % \
                        (i, iter_end_time - iter_start_time))

    if mnet.batchnorm_layers is not None:
        if not config.bn_distill_stats and \
                not config.bn_no_running_stats and \
                not config.bn_no_stats_checkpointing:
            # Checkpoint the current running statistics (that have been estimated while training the current task).
            for bn_layer in mnet.batchnorm_layers:
                assert(bn_layer.num_stats == task_id+1)
                bn_layer.checkpoint_stats()

    avg_iter_time = summed_iter_runtime / config.n_iter
    logger.info('Average runtime per training iteration: %f sec.' % avg_iter_time)

    logger.info('Elapsed time for training task %d: %f sec.' % \
                (task_id+1, time()-start_time))

def test_multiple(dhandlers, mnet, hnet, device, config, shared, writer,
                  logger):
    """Method to test continual learning experiment accuracy

    Args:
        (....): See docstring of function :func:`train`.
        dhandlers (list): List of data handlers. The accuracy of each task in
            this list will be computed using function :func:`test`. The index
            within the list will be considered as task ID.
    """
    class_accs = []
    task_accs = []

    num_tasks = len(dhandlers)

    ### Task-incremental learning
    logger.info('### Testing task-incremental learning scenario')
    # Iterate through learned embeddings and tasks and compute test acc.
    for j in range(num_tasks):
        data = dhandlers[j]

        test_acc = test(j, data, mnet, hnet, device, shared,
                            config, writer, logger)
        
        writer.add_scalar('final/task_incremental',test_acc, j)

        class_accs.append(test_acc)
        shared.summary['acc_final'][j] = test_acc

    shared.summary['acc_avg_final'] = np.mean(class_accs)
    logger.info('### Task-incremental learning scenario accuracies: %s ' \
                % (str(class_accs)) + '(avg: %.3f)'
                % (shared.summary['acc_avg_final']))

    writer.add_scalar('final/task_incremental_avg',
                        shared.summary['acc_avg_final'])

    return task_accs, class_accs

def analysis(dhandlers, mnet, hnet, device, config, shared, writer, logger,
             during_weights):
    """A function to do some post-hoc analysis on the hypernetwork.

    Specifically, this function does the following:
        - Computing and logging statistics on how the weights changed since a
          task has been learned.
        - Assessing the diversity of ``hnet`` outputs, i.e., how close are the
          ``hnet`` outputs for different tasks.

    Args:
        (....): See docstring of function :func:`test_multiple`.
        during_weights (list): List of flattened ``hnet`` outputs right after
            training on each task.
    """
    assert hnet is not None
    mnet.eval()
    hnet.eval()

    num_tasks = len(dhandlers)

    # Test how much the weights of each task have changed during training the remaining tasks
    for j in range(num_tasks):
        cur_weights = hnet.forward(j)
        cur_weights = torch.cat([a.detach().clone().cpu().flatten() for a in cur_weights])
        aft_weights = torch.cat([a.flatten() for a in during_weights[j]])

        logger.info('### Euclidean distance of current hnet output to original one for task %d: %f' % (j, torch.sqrt(torch.sum((aft_weights - cur_weights)**2))))

    # FIXME Inefficient, we already computed all hnet outputs above.
    for j in range(num_tasks):
        for i in range(num_tasks):
            if i <= j:
                continue
            weights_1 = hnet.forward(j)
            weights_2 = hnet.forward(i)
            weights_1 = torch.cat([a.detach().clone().flatten() for a in weights_1])
            weights_2 = torch.cat([a.detach().clone().flatten() for a in weights_2])
            logger.info('### Euclidean distance between task %d and task %d: %f' % (j, i, torch.sqrt(torch.sum((weights_1 - weights_2)**2))))

def run(config, experiment='resnet'):
    """Run the training.

    Args:
        config (argparse.Namespace): Command-line arguments.
        experiment (str): Which kind of experiment should be performed?

            - ``resnet``: CIFAR-10/100 with Resnet-32.
            - ``zenke``: CIFAR-10/100 with Zenkenet.
    """
    assert(experiment in ['resnet', 'zenke'])

    script_start = time()

    device, writer, logger = sutils.setup_environment(config,
        logger_name='det_cl_cifar_%s' % experiment)
    # TODO Adapt script to allow checkpointing of models using
    # `utils.torch_ckpts` (i.e., we should be able to continue training or just
    # test an existing checkpoint).
    #config.ckpt_dir = os.path.join(config.out_dir, 'checkpoints')

    # Container for variables shared across function.
    shared = Namespace()
    shared.experiment = experiment

    ### Load datasets (i.e., create tasks).
    dhandlers = tutils.load_datasets(config, shared, logger,
                                     data_dir=DATA_DIR_CIFAR)

    ### Create main network.
    # TODO Allow main net only training.
    mnet = tutils.get_main_model(config, shared, logger, device,
                                 no_weights=not config.mnet_only)

    ### Create the hypernetwork.
    if config.mnet_only:
        hnet = None
    else:
        hnet = tutils.get_hnet_model(config, mnet, logger, device)

    ### Initialize the performance measures, that should be tracked during
    ### training.
    tutils.setup_summary_dict(config, shared, mnet, hnet=hnet)

    # Add hparams to tensorboard, such that the identification of runs is
    # easier.
    writer.add_hparams(hparam_dict={**vars(config), **{
        'num_weights_main': shared.summary['num_weights_main'],
        'num_weights_hyper': shared.summary['num_weights_hyper'],
        'num_weights_ratio': shared.summary['num_weights_ratio'],
    }}, metric_dict={})

    # FIXME: Method "calc_fix_target_reg" expects a None value.
    # But `writer.add_hparams` can't deal with `None` values.
    if config.cl_reg_batch_size == -1:
        config.cl_reg_batch_size = None

    # We keep the hnet output right after training to measure forgetting.
    weights_after_training = []

    ######################
    ### Start Training ###
    ######################
    for j in range(config.num_tasks):
        logger.info('Starting training of task %d ...' % (j+1))

        data = dhandlers[j]

        # It might be that tasks are very similar and we can transfer knowledge
        # form the previous solution.
        if hnet is not None and config.init_with_prev_emb and j > 0:
            last_emb = hnet.get_task_emb(j-1).detach().clone()
            hnet.get_task_emb(j).data = last_emb

        # Training from scratch -- create new network instance!
        # -> No transfer possible.
        if j > 0 and config.train_from_scratch:
            # FIXME Since we simply override the current network, future testing
            # on this new network for old tasks doesn't make sense. So we
            # shouldn't report `final` accuracies.
            if config.mnet_only:
                logger.info('From scratch training: Creating new main network.')
                mnet = tutils.get_main_model(config, shared, logger, device,
                                             no_weights=not config.mnet_only)
            else:
                logger.info('From scratch training: Creating new hypernetwork.')
                hnet = tutils.get_hnet_model(config, mnet, logger, device)

        ################################
        ### Train and test on task j ###
        ################################
        train(j, data, mnet, hnet, device, config, shared, writer, logger)

        ### Final test run.
        if hnet is not None:
            weights = hnet.forward(j)
            # Push to CPU to avoid growing GPU memory when solving very long
            # task sequences.
            weights = [w.detach().clone().cpu() for w in weights]
            weights_after_training.append(weights)

        test_acc = test(j, data, mnet, hnet, device, shared, config, writer,
                           logger)

        logger.info('### Accuracy of task %d / %d:  %.3f' % (j+1, config.num_tasks, test_acc))
        logger.info('### Finished training task: %d' % (j+1))
        shared.summary['acc_during'][j] = test_acc

        # Backup results so far.
        tutils.save_summary_dict(config, shared, experiment)

    shared.summary['acc_avg_during'] = np.mean(shared.summary['acc_during'])

    logger.info('### Accuracy of individual tasks after training %s' % \
                (str(shared.summary['acc_during'])))
    logger.info('### Average of these accuracies  %.2f' % (shared.summary['acc_avg_during']))
    writer.add_scalar('final/during_acc_avg', shared.summary['acc_avg_during'])

    #########################################
    ### Test continual learning scenarios ###
    #########################################
    test_multiple(dhandlers, mnet, hnet, device, config, shared, writer,
                  logger)

    #########################
    ### Run some analysis ###
    #########################
    if not config.mnet_only:
        analysis(dhandlers, mnet, hnet, device, config, shared, writer, logger,
                 weights_after_training)

    ### Write final summary.
    shared.summary['finished'] = 1
    tutils.save_summary_dict(config, shared, experiment)
    # save hnet
    if hnet is not None:
        torch.save(hnet.state_dict(), config.out_dir + '/hnet.pth')

    writer.close()

    logger.info('Program finished successfully in %f sec.' % (time() - script_start))

if __name__ == '__main__':
    raise Exception('Script is not executable!')

