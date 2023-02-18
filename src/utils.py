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


def progress_report(phase, count, start_time, batchsize, whole_sample, is_last):
    duration = time.time() - start_time
    interval = str(datetime.timedelta(seconds=int(duration)))
    throughput = count * batchsize / duration
    sys.stdout.write(
        f'\r{phase}: {count} updates ({count * batchsize} / {whole_sample} samples)'
        f'time: {interval} ({throughput:.2f} samples/sec)')
    if is_last:
        sys.stdout.write('\r\n')


def init_model(net, device, restore):
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
    acc = np.mean(np.array(gt_list) == np.argmax(pred_list, axis=1))
    pr_auc = average_precision_score(gt_list, pred_list[:, 1])
    plotter.record(epoch, f'{data_name}_accuracy', acc)
    plotter.record(epoch, f'{data_name}_pr_auc', pr_auc)

    return pr_auc


def create_output_dir(args):
    os.makedirs(os.path.join(os.path.curdir, args.log_dir), exist_ok=True)
    # Create or Overwrite the log directory
    for past_log in os.listdir(os.path.join(os.path.curdir, args.log_dir)):
        if past_log[7:] == args.out or past_log[9:] == args.out:
            ans = input(f'overwrite "{past_log}" (y/n)')
            if ans == 'y' or ans == 'Y':
                print('\nmove existing directory to dump')
                past_dir = os.path.join(os.path.curdir, args.log_dir, past_log)
                shutil.rmtree(past_dir)
            else:
                print('\ntry again')
                exit()
    else:
        out = datetime.datetime.now().strftime('%m%d%H%M') + '_' + args.out

    out_dir = os.path.abspath(os.path.join(os.path.curdir, args.log_dir, out))
    os.makedirs(out_dir, exist_ok=True)

    # save args
    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return out_dir


class Plotter:
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
