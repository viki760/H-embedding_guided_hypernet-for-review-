#!/usr/bin/env python3

# 导入必要的库和模块
import sys
import numpy as np
import torch.nn as nn

from argparse import Namespace
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as nps
from time import time
import copy
# 导入自定义模块
from cifar import train_utils as tutils
from mnets.classifier_interface import Classifier
from utils import sim_utils as sutils
import utils.optim_step as opstep
import utils.hnet_regularizer as hreg
from utils.torch_utils import get_optimizer
from metrics.online_embedding import Hembedding as Hemb
# from hypernet_exp1.cifar.train.fc_decoder import EmbDecoder

# 定义CIFAR数据集的路径
DATA_DIR_CIFAR = r"/mnt/d/task/research/codes/MultiSource/wsl/2/multi-source/data/"

# 定义测试函数
def test(task_id, data, mnet, device, shared, config, writer, logger,
         train_iter=None, task_emb=None, cl_scenario=1, test_size=None):
    """
    对指定任务进行测试
    
    参数:
    - task_id: 当前任务的ID
    - data: 数据处理器
    - mnet: 主网络模型
    - device: 计算设备(CPU/GPU)
    - shared: 共享量
    - config: 配置参数
    - writer: TensorBoard写入器
    - logger: 日志记录器
    - train_iter: 当前训练迭代次数
    - task_emb: 任务嵌入
    - cl_scenario: 持续学习场景
    - test_size: 测试集大小
    
    返回:
    - test_acc: 测试准确率
    """
    
    # 将网络设置为评估模式
    mnet.eval()
    
    # 记录日志
    logger.info('### Test run ...') if train_iter is None else logger.info(f'# Testing network before running training step {train_iter} ...')

    # 设置主网络的参数
    mnet_kwargs = {}
    if mnet.batchnorm_layers is not None:
        if not config.bn_no_running_stats and not config.bn_no_stats_checkpointing:
            mnet_kwargs['condition'] = min(task_id, config.num_tasks - 1)  # 确保不超过最大任务数
            if task_emb is not None:
                logger.warning(f'Using batch statistics for task {task_id}, but testing with a specific task embedding.')

    # 开始测试过程
    with torch.no_grad():
        batch_size = config.val_batch_size
        n_head = data.num_classes

        if test_size is None or test_size >= data.num_test_samples:
            test_size = data.num_test_samples
            
        # 重置批次生成器
        data.reset_batch_generator(train=False, test=True, val=False)
        logger.info(f'{test_size}/{data.num_test_samples} of test set is used for this test run.')

        test_loss = 0.0

        # 初始化存储预测结果的数组
        correct_labels = np.empty(test_size, np.int64)
        pred_labels = np.empty(test_size, np.int64)

        N_processed = 0

        # 遍历测试集
        while N_processed < test_size:
            cur_batch_size = min(test_size - N_processed, batch_size)
            N_processed += cur_batch_size

            # 获取一个批次的数据
            batch = data.next_test_batch(cur_batch_size)
            X = data.input_to_torch_tensor(batch[0], device)
            T = data.output_to_torch_tensor(batch[1], device)

            correct_labels[N_processed-cur_batch_size:N_processed] = T.argmax(dim=1, keepdim=False).cpu().numpy()

            # 获取模型预测结果
            Y_hat_logits = mnet.forward(X, weights=None, **mnet_kwargs)

            # 选择当前任务的输出头
            task_out = [task_id*n_head, (task_id+1)*n_head]
            Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]
            
            # 计算softmax并获取预测标签
            Y_hat = F.softmax(Y_hat_logits, dim=1).cpu().numpy()
            pred_labels[N_processed-cur_batch_size:N_processed] = Y_hat.argmax(axis=1)

            # 计算损失
            test_loss += Classifier.logit_cross_entropy_loss(Y_hat_logits, T,
                                                             reduction='sum')

        # 计算准确
        class_n_correct = (correct_labels == pred_labels).sum()
        test_acc = 100.0 * class_n_correct / test_size

        test_loss /= test_size

        # 记录测试结果
        task_info = f" (before training iteration {train_iter})" if train_iter is not None else ""
        task_emb_info = " (using a given task embedding)" if task_emb is not None else ""

        msg = f"### Test accuracy of task {task_id + 1}{task_info}: {test_acc:.3f}{task_emb_info}"

        logger.info(msg)

        if train_iter is not None:
            writer.add_scalar('test/task_%d/class_accuracy' % task_id, test_acc, train_iter)
            writer.add_scalar('test/task_%d/test_loss' % task_id, test_loss, train_iter)

        return test_acc
    
# 定义计算PredKD损失的函数
def compute_predkd_loss(config, current_model, previous_model, data, device, task_id, n_y):
    predkd_loss = 0
    current_model.eval()
    previous_model.eval()
    
    with torch.no_grad():
        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        
        # 获取当前模型对所有先前任务的预测
        current_preds = current_model(X)[:, :task_id*n_y]
        
        # 获取先前模型对所有先前任务的预测
        previous_preds = previous_model(X)[:, :task_id*n_y]
        
    # 计算 CE
    predkd_loss = F.cross_entropy(
        current_preds / 2.0,
        F.log_softmax(previous_preds / 2.0, dim=1),
        reduction='batchmean'
    ) * (2.0 ** 2)
    
    return predkd_loss
# 

# 定义训练函数
def train(task_id, data, mnet, device, config, shared, writer, logger):
    logger.info('Training network ...')
    start_time = time()

    mnet.train()
    theta_optimizer = get_optimizer(mnet.parameters(), config.lr,
        momentum=config.momentum, weight_decay=config.weight_decay,
        use_adam=config.use_adam, adam_beta1=config.adam_beta1,
        use_rmsprop=config.use_rmsprop)

    # 设置学习率调度器
    if config.plateau_lr_scheduler:
        plateau_scheduler_theta = optim.lr_scheduler.ReduceLROnPlateau(
            theta_optimizer, 'max', factor=np.sqrt(0.1), patience=5,
            min_lr=0.5e-6, cooldown=0)

    if config.lambda_lr_scheduler:
        def lambda_lr(epoch):
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

    mnet_kwargs = {}
    if mnet.batchnorm_layers is not None:
        if not config.bn_no_running_stats and not config.bn_no_stats_checkpointing:
            mnet_kwargs['condition'] = task_id

    iter_per_epoch = int(np.ceil(data.num_train_samples / config.batch_size))
    training_iterations = config.epochs * iter_per_epoch

    logger.info('Epochs per task: %d ...' % config.epochs)
    logger.info('Iters per task: %d ...' % training_iterations)

    # 初始化旧模型
    if task_id > 0:
        old_model = copy.deepcopy(mnet)
        old_model.eval()
    else:
        old_model = None

    summed_iter_runtime = 0
    
    # 早停相关变量
    best_val_acc = 0
    patience = config.early_stopping_patience  # 假设这个参数在config中定义
    patience_counter = 0
    early_stop = False
    
    # for i in range(100):        
    for i in range(training_iterations):        
        if (i+1) % iter_per_epoch == 0:
            current_epoch = (i+1) // iter_per_epoch
           
            val_acc = test(task_id, data, mnet, device, shared, config, writer, logger, train_iter=i+1)
            mnet.train()

            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 如果需要，可以在这里保存最佳模型
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f'在第 {current_epoch} 轮触发早停')
                early_stop = True
                break

            # 检查是否达到50轮
            if current_epoch >= 100:
                break

        if early_stop:
            break

        if i % 200 == 0:
            logger.info('训练步骤: %d ...' % i)

        iter_start_time = time()

        theta_optimizer.zero_grad()

        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        n_y = data.num_classes

        Y_hat_logits = mnet.forward(X, weights=None, **mnet_kwargs)

        # 选择当前任务的输出头
        task_out = [task_id*n_y, (task_id+1)*n_y]
        Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]
        assert(T.shape[1] == Y_hat_logits.shape[1])

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

        if task_id > 0:
            predkd_loss = compute_predkd_loss(config, mnet, old_model, data, device, task_id, n_y)
            loss = loss_task + config.predkd_lambda * predkd_loss
        else:
            loss = loss_task
            predkd_loss = torch.tensor(0.0)  # 为了记录和打印

        loss.backward()
        theta_optimizer.step()

        Y_hat = F.softmax(Y_hat_logits, dim=1)
        classifier_accuracy = Classifier.accuracy(Y_hat, T) * 100.0

        # 学习率调度
        if config.plateau_lr_scheduler:
            if i % iter_per_epoch == 0 and i > 0:
                curr_epoch = i // iter_per_epoch
                logger.info('Computing test accuracy for plateau LR ' +
                            'scheduler (epoch %d).' % curr_epoch)
                test_acc = test(task_id, data, mnet, device, shared,
                                   config, writer, logger, train_iter=i+1)
                mnet.train()

                plateau_scheduler_theta.step(test_acc)

        if config.lambda_lr_scheduler:
            if i % iter_per_epoch == 0 and i > 0:
                curr_epoch = i // iter_per_epoch
                logger.info('Applying Lambda LR scheduler (epoch %d).'
                            % curr_epoch)

                lambda_scheduler_theta.step()

        # TensorBoard记录
        if i % 50 == 0:
            writer.add_scalar('train/task_%d/class_accuracy' % task_id, classifier_accuracy, i)
            writer.add_scalar('train/task_%d/loss_task' % task_id, loss_task, i)
            writer.add_scalar('train/task_%d/predkd_loss' % task_id, predkd_loss, i)

        if i % config.val_iter == 0:
            msg = 'Training step {}: Classifier Accuracy: {:.3f} (on current training batch).'
            logger.debug(msg.format(i, classifier_accuracy))

        iter_end_time = time()
        summed_iter_runtime += (iter_end_time - iter_start_time)
    # Update batch normalization statistics
    if mnet.batchnorm_layers is not None:
        if not config.bn_distill_stats and not config.bn_no_running_stats and not config.bn_no_stats_checkpointing:
            for bn_layer in mnet.batchnorm_layers:
                assert bn_layer.num_stats == task_id+1
                bn_layer.checkpoint_stats()

    avg_iter_time = summed_iter_runtime / training_iterations
    logger.info('Average runtime per training iteration: %f sec.' % avg_iter_time)

    logger.info('Elapsed time for training task %d: %f sec.' % \
                (task_id+1, time()-start_time))

import numpy as np

def test_multiple(dhandlers, mnet, device, config, shared, writer, logger, current_task):
    """
    测试持续学习实验的准确率
    
    参数:
    - dhandlers: 数据处理器列表
    - mnet: 主网络模型
    - device: 计算设备(CPU/GPU)
    - config: 配置参数
    - shared: 共享变量
    - writer: TensorBoard写入器
    - logger: 日志记录器
    - current_task: 当前任务的索引
    
    返回:
    - accuracy_matrix: 任务准确率矩阵
    """
    num_tasks = len(dhandlers)
    
    # 如果是第一个任务，初始化准确率矩阵
    if current_task == 0:
        shared.accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    logger.info(f'### 测试任务增量学习场景 (当前任务: {current_task + 1})')
    
    for j in range(current_task + 1):
        data = dhandlers[j]
        
        test_acc = test(j, data, mnet, device, shared, config, writer, logger)
        
        shared.accuracy_matrix[current_task, j] = test_acc
        
        writer.add_scalar(f'task_incremental/task_{j+1}_accuracy', test_acc, current_task)
        
        logger.info(f'任务 {j+1} 在任务 {current_task+1} 之后的准确率: {test_acc:.2f}%')
    
    # 计算到目前为止的平均准确率
    current_avg_acc = np.mean(shared.accuracy_matrix[current_task, :current_task+1])
    writer.add_scalar('task_incremental/average_accuracy', current_avg_acc, current_task)
    
    logger.info(f'### 任务 {current_task+1} 结束后的平均准确率: {current_avg_acc:.2f}%')
    
    # 更新shared.summary字典
    shared.summary['acc_final'] = shared.accuracy_matrix[current_task].tolist()
    shared.summary['acc_during'] = shared.accuracy_matrix.diagonal().tolist()
    shared.summary['acc_avg_final'] = np.mean(shared.accuracy_matrix[current_task])
    shared.summary['acc_avg_during'] = np.mean(shared.accuracy_matrix.diagonal())

    return shared.accuracy_matrix

# 定义主运行函数
def run(config, experiment='resnet'):
    """
    运行训练过程
    
    参数:
    - config: 配置参数
    - experiment: 实验类型 ('resnet' 或 'zenke')
    """
    script_start = time()

    # 设置环境
    device, writer, logger = sutils.setup_environment(config,
        logger_name='det_cl_cifar_%s' % experiment)

    # 创建共享变量容器
    shared = Namespace()
    shared.experiment = experiment

    # 加载数据集
    dhandlers = tutils.load_datasets(config, shared, logger,
                                     data_dir=DATA_DIR_CIFAR)

    # 创建主网络
    mnet = tutils.get_main_model(config, shared, logger, device,
                                 no_weights=False)

    # 初始化性能指标
    tutils.setup_summary_dict(config, shared, mnet)

    # 添加超参数到TensorBoard
    writer.add_hparams(hparam_dict={**vars(config), **{
        'num_weights_main': shared.summary['num_weights_main'],
        'num_weights_hyper': shared.summary['num_weights_hyper'],
        'num_weights_ratio': shared.summary['num_weights_ratio'],
    }}, metric_dict={})

    if config.cl_reg_batch_size == -1:
        config.cl_reg_batch_size = None

    # Start training loop
    for j, data in enumerate(dhandlers):
        logger.info(f'开始训练任务 {j+1} ...')

        # If training from scratch is required
        if j > 0 and config.train_from_scratch:
            logger.info('From scratch training: Creating new main network.')
            mnet = tutils.get_main_model(config, shared, logger, device, no_weights=False)

        # Train on current task
        train(j, data, mnet, device, config, shared, writer, logger)

        # Test on current task and all previous tasks
        accuracy_matrix = test_multiple(dhandlers, mnet, device, config, shared, writer, logger, j)

        # 更新shared.summary
        shared.summary['acc_final'] = accuracy_matrix[j].tolist()
        shared.summary['acc_during'] = accuracy_matrix.diagonal().tolist()
        shared.summary['acc_avg_final'] = np.mean(accuracy_matrix[j])
        shared.summary['acc_avg_during'] = np.mean(accuracy_matrix.diagonal())
        shared.summary['accuracy_matrix'] = shared.accuracy_matrix.tolist()  # 添加这行

        # Save intermediate results
        tutils.save_summary_dict(config, shared, experiment)

        # Record task completion in TensorBoard
        writer.add_scalar('task_completion', j+1, j)

    # 训练结束后，打印最终的准确率矩阵
    logger.info("最终准确率矩阵:")
    logger.info(shared.accuracy_matrix)

    # 计算并记录最终的平均准确率
    final_avg_acc = np.mean(shared.accuracy_matrix[config.num_tasks-1, :])
    writer.add_scalar('final/overall_accuracy', final_avg_acc)
    logger.info(f'最终平均准确率: {final_avg_acc:.2f}%')

    # 更新最终的shared.summary
    shared.summary['acc_final'] = shared.accuracy_matrix[-1].tolist()
    shared.summary['acc_during'] = shared.accuracy_matrix.diagonal().tolist()
    shared.summary['acc_avg_final'] = final_avg_acc
    shared.summary['acc_avg_during'] = np.mean(shared.accuracy_matrix.diagonal())
    shared.summary['accuracy_matrix'] = shared.accuracy_matrix.tolist()  # 添加这行
    shared.summary['finished'] = 1

    # 保存最终结果
    tutils.save_summary_dict(config, shared, experiment)

    writer.close()

    logger.info(f'Program finished successfully in {time() - script_start:.2f} sec.')

if __name__ == '__main__':
    raise Exception('Script is not executable!')
