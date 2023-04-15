import os
import logging
from collections import defaultdict
import json
import sys
import time
import datetime
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def progress_report(phase, count, start_time, batchsize, whole_sample, is_last=False):
    """Display the progress report of training and inference.

    Args:
        phase (str): Identifier of this step (E.g. 'train-source')
        count (int): Step count
        start_time (float): Timestamp
        batchsize (int): Batchsize
        whole_sample (int): Total size of data
        is_last (bool): If this is the last step is_last = True.
    """
    duration = time.time() - start_time
    interval = str(datetime.timedelta(seconds=int(duration)))
    throughput = count * batchsize / duration
    sys.stdout.write(
        f'\r{phase}: {count} updates ({count * batchsize} / {whole_sample} samples)'
        f'time: {interval} ({throughput:.2f} samples/sec)')
    if is_last:
        sys.stdout.write('\r\n')


def init_model(net, device, restore):
    """Initialize models

    Args:
        net : Model class
        device : torch.device
        restore (str): Path of trained model

    Returns:
        net : Initialized model
    """
    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        logger.info(f"Load existing model from: {os.path.abspath(restore)}")

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True

    return net.to(device)


def evaluate(plotter, epoch, data_name, pred_list, gt_list):
    """Evaluate model prediction and save it in plotter.

    Args:
        plotter : Plotter class
        epoch (int): Current epoch
        data_name (str): Log identifier
        pred_list : Numpy array of predicted scores
        gt_list : Numpy array of ture labels

    Returns:
        pr_auc: PR-AUC score
    """
    acc = np.mean(np.array(gt_list) == np.argmax(pred_list, axis=1))
    pr_auc = average_precision_score(gt_list, pred_list[:, 1])
    plotter.record(epoch, f'{data_name}_accuracy', acc)
    plotter.record(epoch, f'{data_name}_pr_auc', pr_auc)

    return pr_auc


def create_output_dir(base_dir, cur_dir, args):
    """Create a directory to save models and logs.
    If the directory already exists, it asks if you want to overwrite it.

    Args:
        base_dir (str): Base directory name to create current log directory
        cur_dir (str): Current log directory name
        args : Parameters to save

    Returns:
        out_dir: Full path of Current log directory
    """
    os.makedirs(os.path.join(os.path.curdir, base_dir), exist_ok=True)
    # Create or Overwrite the log directory
    for past_log in os.listdir(os.path.join(os.path.curdir, base_dir)):
        if past_log[7:] == cur_dir or past_log[9:] == cur_dir:
            ans = input(f'overwrite "{past_log}" (y/n)')
            if ans == 'y' or ans == 'Y':
                print('\nmove existing directory to dump')
                past_dir = os.path.join(os.path.curdir, base_dir, past_log)
                shutil.rmtree(past_dir)
            else:
                print('\ntry again')
                exit()
    else:
        out = datetime.datetime.now().strftime('%m%d%H%M') + '_' + cur_dir

    out_dir = os.path.abspath(os.path.join(os.path.curdir, base_dir, out))
    os.makedirs(out_dir, exist_ok=True)

    # Save args
    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return out_dir


class Plotter:
    """Generic class for logging and plotting fugures

        Args:
            out_dir (str): Output directory name
        """
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.logs = defaultdict(lambda: {})
        self.best_score = 0.0

        os.makedirs(os.path.join(out_dir, 'plot'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'feature'), exist_ok=True)

    def save_files(self, files, score):
        if self.best_score <= score:
            for file_name, file in files.items():
                np.save(os.path.join(self.out_dir, 'feature', file_name), file)

    def save_models(self, models, score):
        if self.best_score <= score:
            logger.info(f"Best score {score}. Models are saved.")
            self.best_score = score
            for model_name, model in models.items():
                torch.save(model.state_dict(), os.path.join(self.out_dir, 'model', model_name))

    def record(self, epoch, name, value):
        self.logs[name][epoch] = value

    def refresh(self):
        self.logs = defaultdict(lambda: {})

    def flush(self, epoch, plot_flag=True):
        """Plot fugires and save in files
        """
        log = f"epoch {epoch}\n"

        keys = sorted(self.logs.keys())
        for name in keys:
            vals = self.logs[name]
            log += f" {name}\t{vals[epoch]:.5f}\n"

            if plot_flag:
                x_vals = np.sort(list(vals.keys()))
                y_vals = [vals[x] for x in x_vals]

                plt.clf()
                plt.plot(x_vals, y_vals)
                plt.xlabel('iteration')
                plt.ylabel(name)
                plt.savefig(os.path.join(self.out_dir, 'plot', name.replace(' ', '_') + '.jpg'))

        with open(os.path.join(self.out_dir, 'log'), 'a+') as f:
            f.write(log + '\n')
