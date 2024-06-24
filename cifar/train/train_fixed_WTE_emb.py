#!/usr/bin/env python3
# Copyright 2019 Johannes von Oswald
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title           :cifar/train.py
# @author          :jvo, ch
# @contact         :oswald@ini.ethz.ch
# @created         :04/10/2019
# @version         :1.0
# @python_version  :3.6.5
"""
Training of a deterministic CL hypernetwork on CIFAR-10/100
-----------------------------------------------------------

The module :mod:`cifar.train` implements the training of a deterministic
hypernet to solve a CIFAR continual learning problem.

.. note::
    The module is not executable! Please refer to :mod:`cifar.train_resnet` or
    :mod:`cifar.train_zenke`.
"""
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
from metrics.WTE import WTE
from hypercl.cifar.train.train_vanilla import *

DATA_DIR_CIFAR = r"/mnt/d/task/research/codes/MultiSource/wsl/2/multi-source/data/"

def train_with_emb(task_id, data, mnet, hnet, device, config, shared, writer, logger, with_emb = True, task_embs=None):
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

    mnet.train()
    if hnet is not None:
        hnet.train()

    #################
    ### Optimizer ###
    #################
    # Define the optimizers used to train main network and hypernet.
    #TODO: check the definition of Hnet class and add conditional generalization
    if hnet is not None:
        theta_params = list(hnet.theta)
        if with_emb:
            print("Training with given task embedding (fixed)")
        # else:
        #     if config.continue_emb_training:
        #         for i in range(task_id): # for all previous task embeddings
        #             theta_params.append(hnet.get_task_emb(i))

        #     # Only for the current task embedding.
        #     # Important that this embedding is in a different optimizer in case
        #     # we use the lookahead.
        #     emb_optimizer = get_optimizer([hnet.get_task_emb(task_id)],
        #         config.lr, momentum=config.momentum,
        #         weight_decay=config.weight_decay, use_adam=config.use_adam,
        #         adam_beta1=config.adam_beta1, use_rmsprop=config.use_rmsprop)

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
        # The scheduler config has been taken from here:
        # https://keras.io/examples/cifar10_resnet/
        # Note, we use 'max' instead of 'min' as we look at accuracy rather
        # than validation loss!
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

            Args:
                epoch (int): The number of epochs

            Returns:
                lr_scale (float32): learning rate scale
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

        lambda_scheduler_theta = optim.lr_scheduler.LambdaLR(theta_optimizer,
                                                             lambda_lr)
        lambda_scheduler_emb = None
        if emb_optimizer is not None:
            lambda_scheduler_emb = optim.lr_scheduler.LambdaLR(emb_optimizer,
                                                               lambda_lr)

    ##############################
    ### Prepare CL Regularizer ###
    ##############################
    # Whether we will calculate the regularizer.
    calc_reg = task_id > 0 and not config.mnet_only and config.beta > 0 and \
        not config.train_from_scratch

    # Compute targets when the reg is activated and we are not training
    # the first task
    if calc_reg:
        if config.online_target_computation:
            # Compute targets for the regularizer whenever they are needed.
            # -> Computationally expensive.
            targets_hypernet = None
            prev_theta = [p.detach().clone() for p in hnet.theta]
            prev_task_embs = [p.detach().clone() for p in hnet.get_task_embs()]
        else:
            # Compute targets for the regularizer once and keep them all in
            # memory -> Memory expensive.
            targets_hypernet = hreg.get_current_targets(task_id, hnet)
            prev_theta = None
            prev_task_embs = None

        # If we do not want to regularize all outputs (in a multi-head setup).
        # Note, we don't care whether output heads other than the current one
        # change.
        regged_outputs = None
        if config.cl_scenario != 2:
            # FIXME We assume here that all tasks have the same output size.
            n_y = data.num_classes
            regged_outputs = [list(range(i*n_y, (i+1)*n_y)) for i in
                              range(task_id)]

    # We need to tell the main network, which batch statistics to use, in case
    # batchnorm is used and we checkpoint the batchnorm stats.
    mnet_kwargs = {}
    if mnet.batchnorm_layers is not None:
        if config.bn_distill_stats:
            raise NotImplementedError()
        elif not config.bn_no_running_stats and \
                not config.bn_no_stats_checkpointing:
            # Specify current task as condition to select correct
            # running stats.
            mnet_kwargs['condition'] = task_id

    ######################
    ### Start training ###
    ######################

    iter_per_epoch = -1
    if config.epochs == -1:
        training_iterations = config.n_iter
    else:
        assert(config.epochs > 0)
        iter_per_epoch = int(np.ceil(data.num_train_samples / \
                                     config.batch_size))
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
                 logger, train_iter=i, task_emb=task_embs[task_id])
            mnet.train()
            if hnet is not None:
                hnet.train()

        if i % 200 == 0:
            logger.info('Training step: %d ...' % i)

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
        if config.cl_scenario == 1:
            # Choose current head.
            task_out = [task_id*n_y, (task_id+1)*n_y]
        elif config.cl_scenario == 2:
            # Always all output neurons, only one head is used.
            task_out = [0, n_y]
        else:
            # Choose current head, which will be inferred during inference.
            task_out = [task_id*n_y, (task_id+1)*n_y]

        ########################
        ### Loss computation ###
        ########################
        if config.mnet_only:
            weights = None
        else:
            weights = hnet.forward(task_emb=task_embs[task_id])
        Y_hat_logits = mnet.forward(X, weights, **mnet_kwargs)

        # Restrict output neurons
        Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]
        assert(T.shape[1] == Y_hat_logits.shape[1])
        # compute loss on task and compute gradients
        if config.soft_targets:
            soft_label = 0.95
            num_classes = data.num_classes
            soft_targets = torch.where(T == 1,
                torch.Tensor([soft_label]),
                torch.Tensor([(1 - soft_label) / (num_classes-1)]))
            soft_targets = soft_targets.to(device)
            loss_task = Classifier.softmax_and_cross_entropy(Y_hat_logits,
                                                             soft_targets)
        else:
            loss_task = Classifier.logit_cross_entropy_loss(Y_hat_logits, T)

        # Compute gradients based on task loss (those might be used in the CL
        # regularizer).
        loss_task.backward(retain_graph=calc_reg, create_graph=calc_reg and \
                           config.backprop_dt)

        # The current task embedding only depends in the task loss, so we can
        # update it already.
        if emb_optimizer is not None:
            emb_optimizer.step()

        #############################
        ### CL (HNET) Regularizer ###
        #############################
        loss_reg = 0
        dTheta = None

        if calc_reg:
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
            writer.add_scalar('train/task_%d/class_accuracy' % task_id,
                              classifier_accuracy, i)
            writer.add_scalar('train/task_%d/loss_task' % task_id, loss_task, i)
            writer.add_scalar('train/task_%d/loss_reg' % task_id, loss_reg, i)

        ### Show the current training progress to the user.
        if i % config.val_iter == 0:
            msg = 'Training step {}: Classifier Accuracy: {:.3f} ' + \
                  '(on current training batch).'
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
            # Checkpoint the current running statistics (that have been
            # estimated while training the current task).
            for bn_layer in mnet.batchnorm_layers:
                assert(bn_layer.num_stats == task_id+1)
                bn_layer.checkpoint_stats()

    avg_iter_time = summed_iter_runtime / config.n_iter
    logger.info('Average runtime per training iteration: %f sec.' % \
                avg_iter_time)

    logger.info('Elapsed time for training task %d: %f sec.' % \
                (task_id+1, time()-start_time))


def analysis(dhandlers, mnet, hnet, device, config, shared, writer, logger,
             during_weights, task_embs):
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

    # Test how much the weights of each task have changed during training the
    # remaining tasks.
    for j in range(len(task_embs)):
        cur_weights = hnet.forward(task_emb=task_embs[j])
        cur_weights = torch.cat([a.detach().clone().cpu().flatten()
                                                        for a in cur_weights])
        aft_weights = torch.cat([a.flatten() for a in during_weights[j]])

        logger.info('### Euclidean distance of current hnet output to ' +
                    'original one for task %d: %f' % \
                    (j, torch.sqrt(torch.sum((aft_weights - cur_weights)**2))))

    # FIXME Inefficient, we already computed all hnet outputs above.
    for j in range(len(task_embs)):
        for i in range(len(task_embs)):
            if i <= j:
                continue
            weights_1 = hnet.forward(task_emb=task_embs[j])
            weights_2 = hnet.forward(task_emb=task_embs[i])
            weights_1 = torch.cat([a.detach().clone().flatten() \
                                   for a in weights_1])
            weights_2 = torch.cat([a.detach().clone().flatten() \
                                   for a in weights_2])
            logger.info('### Euclidean distance between ' +
                        'task %d and task %d: %f' % (j, i,
                        torch.sqrt(torch.sum((weights_1 - weights_2)**2))))

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

        #* WTE task embedding
        #TODO
        rand_task_embs = torch.tensor()
        # WTE_task_embs = WTE(data)

        ################################
        ### Train and test on task j ###
        ################################
        train_with_emb(j, data, mnet, hnet, device, config, shared, writer, logger,task_embs=rand_task_embs)

        ### Final test run.
        if hnet is not None:
            weights = hnet.forward(task_emb=rand_task_embs[j])
            # Push to CPU to avoid growing GPU memory when solving very long
            # task sequences.
            weights = [w.detach().clone().cpu() for w in weights]
            weights_after_training.append(weights)

        test_acc, _ = test(j, data, mnet, hnet, device, shared, config, writer,
                           logger)

        logger.info('### Accuracy of task %d / %d:  %.3f' % \
                    (j+1, config.num_tasks, test_acc))
        logger.info('### Finished training task: %d' % (j+1))
        shared.summary['acc_during'][j] = test_acc

        # Backup results so far.
        tutils.save_summary_dict(config, shared, experiment)

    shared.summary['acc_avg_during'] = np.mean(shared.summary['acc_during'])

    logger.info('### Accuracy of individual tasks after training %s' % \
                (str(shared.summary['acc_during'])))
    logger.info('### Average of these accuracies  %.2f' % \
                (shared.summary['acc_avg_during']))
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

    writer.close()

    logger.info('Program finished successfully in %f sec.'
                % (time() - script_start))

if __name__ == '__main__':
    raise Exception('Script is not executable!')

