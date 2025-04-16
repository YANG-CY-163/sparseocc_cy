import os
import sys
import glob
import tempfile
import mmcv
import torch
import shutil
import logging
import datetime
import socket
import wandb
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger import LoggerHook, TextLoggerHook
from mmcv.runner.dist_utils import master_only
from torch.utils.tensorboard import SummaryWriter
import time
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
import bisect
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import _calc_dynamic_intervals
import os.path as osp
import pickle

def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def backup_code(work_dir, verbose=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for pattern in ['*.py', 'configs/*.py', 'models/*.py', 'loaders/*.py', 'loaders/pipelines/*.py']:
        for file in glob.glob(pattern):
            src = os.path.join(base_dir, file)
            dst = os.path.join(work_dir, 'backup', os.path.dirname(file))

            if verbose:
                logging.info('Copying %s -> %s' % (os.path.relpath(src), os.path.relpath(dst)))
            
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, dst)


@HOOKS.register_module()
class MyTextLoggerHook(TextLoggerHook):
    def _log_info(self, log_dict, runner):
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        # by epoch: Epoch [4][100/1000]
        # by iter:  Iter [100/100000]
        if self.by_epoch:
            log_str = f'Epoch [{log_dict["epoch"]}/{runner.max_epochs}]' \
                        f'[{log_dict["iter"]}/{len(runner.data_loader)}] '
        else:
            log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}] '

        log_str += 'loss: %.2f, ' % log_dict['loss']

        if 'time' in log_dict.keys():
            # MOD: skip the first iteration since it's not accurate
            if runner.iter == self.start_iter:
                time_sec_avg = log_dict['time']
            else:
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter)

            eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_str += f'eta: {eta_str}, '
            log_str += f'time: {log_dict["time"]:.2f}s, ' \
                        f'data: {log_dict["data_time"] * 1000:.0f}ms, '
            # statistic memory
            if torch.cuda.is_available():
                log_str += f'mem: {log_dict["memory"]}M'

        runner.logger.info(log_str)

    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = {
            'mode': self.get_mode(runner),
            'epoch': self.get_epoch(runner),
            'iter': cur_iter
        }

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)

        # MOD: disable writing to files
        # self._dump_log(log_dict, runner)
        self._log_info(log_dict, runner)

        return log_dict

    def after_train_epoch(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            runner.log_buffer.output.pop('eval_iter_num')

        if runner.log_buffer.ready:
            metrics = self.get_loggable_tags(runner)
            runner.logger.info('--- Evaluation Results ---')
            runner.logger.info('RayIoU: %.4f' % metrics['val/RayIoU'])


@HOOKS.register_module()
class MyTensorboardLoggerHook(LoggerHook):
    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=False, by_epoch=True):
        super(MyTensorboardLoggerHook, self).__init__(
            interval, ignore_last, reset_flag, by_epoch)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        super(MyTensorboardLoggerHook, self).before_run(runner)
        if self.log_dir is None:
            self.log_dir = runner.work_dir
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)

        for key, value in tags.items():
            # MOD: merge into the 'train' group
            if key == 'learning_rate':
                key = 'train/learning_rate'

            # MOD: skip momentum
            ignore = False
            if key == 'momentum':
                ignore = True

            # MOD: skip intermediate losses
            for i in range(5):
                if key[:13] == 'train/d%d.loss' % i:
                    ignore = True

            if self.get_mode(runner) == 'train' and key[:5] != 'train':
                ignore = True

            if self.get_mode(runner) != 'train' and key[:3] != 'val':
                ignore = True

            if ignore:
                continue

            if key[:5] == 'train':
                self.writer.add_scalar(key, value, self.get_iter(runner))
            elif key[:3] == 'val':
                self.writer.add_scalar(key, value, self.get_epoch(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()


# modified from mmcv.runner.hooks.logger.wandb
@HOOKS.register_module()
class MyWandbLoggerHook(LoggerHook):
    """Class to log metrics with wandb.

    It requires `wandb`_ to be installed.


    Args:
        log_dir (str): directory for saving logs
            Default None.
        project_name (str): name for your project (mainly used to specify saving path on wandb server)
            Default None.
        team_name (str): name for your team (mainly used to specify saving path on wandb server)
            Default None.
        experiment_name (str): name for your run, if not specified, use the last part of log_dir
            Default None.
        interval (int): Logging interval (every k iterations).
            Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
            Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        commit (bool): Save the metrics dict to the wandb server and increment
            the step. If false ``wandb.log`` just updates the current metrics
            dict with the row argument and metrics won't be saved until
            ``wandb.log`` is called with ``commit=True``.
            Default: True.
        by_epoch (bool): Whether EpochBasedRunner is used.
            Default: True.
        with_step (bool): If True, the step will be logged from
            ``self.get_iters``. Otherwise, step will not be logged.
            Default: True.
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be uploaded to wandb.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.4.3.`

    .. _wandb:
        https://docs.wandb.ai
    """
    def __init__(self, log_dir=None, project_name=None, team_name=None, experiment_name=None, 
                 interval=10, ignore_last=True, reset_flag=False, by_epoch=True, commit=True, 
                 with_step=True, out_suffix = ('.log.json', '.log', '.py')):
        
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.import_wandb()
        self.commit = commit
        self.with_step = with_step
        self.out_suffix = out_suffix
        
        self.log_dir = log_dir
        self.project_name = project_name
        self.team_name = team_name
        self.experiment_name = experiment_name
        if commit:
            os.system('wandb online')
        else:
            os.system('wandb offline')
            
    def import_wandb(self) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb
        
    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        if self.log_dir is None:
            self.log_dir = runner.work_dir
        if self.experiment_name is None:
            self.experiment_name = os.path.basename(self.log_dir)
        init_kwargs = dict(
            project=self.project_name,
            entity=self.team_name,
            notes=socket.gethostname(),
            name=self.experiment_name,
            dir=self.log_dir,
            reinit=True
        )
            
        if self.wandb is None:
            self.import_wandb()
        if init_kwargs:
            self.wandb.init(**init_kwargs)  # type: ignore
        else:
            self.wandb.init()  # type: ignore
    
    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        mode = self.get_mode(runner)
        if not tags:
            return
        if 'learning_rate' in tags.keys():
            tags['train/learning_rate'] = tags['learning_rate']
            del tags['learning_rate']
        if 'momentum' in tags.keys():
            del tags['momentum']
        tags = {k: v for k, v in tags.items() if k.startswith(mode)}
        
        if self.with_step:
            self.wandb.log(
                tags, step=self.get_iter(runner), commit=self.commit)
        else:
            tags['global_step'] = self.get_iter(runner)
            self.wandb.log(tags, commit=self.commit)

    @master_only
    def after_run(self, runner) -> None:
        self.wandb.join()

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    return results

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        NOTE bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        '''
        NOTE bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

# Modify from DistEvalHook
class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.

        # NOTE  use custom_multi_gpu_test, since sampler of val_loader is changed, 
        # result collect should also be changed 
        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)