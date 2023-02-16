import os
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
from sklearn.metrics import roc_auc_score, average_precision_score


def progress_report(epoch, count, start_time, batchsize, whole_sample):
    duration = time.time() - start_time
    throughput = count * batchsize / duration
    sys.stderr.write(
        '\r epoch {}: {} updates ({} / {} samples) time: {} ({:.2f} samples/sec)'.format(
        epoch, count, count * batchsize, whole_sample, str(datetime.timedelta(seconds=int(duration))), throughput))


def init_model(net, device, restore):
    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Load existing model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True

    return net.to(device)


def evaluate(plotter, epoch, data_name, pred_list, gt_list, args_mode):
    acc = np.mean(np.array(gt_list) == np.argmax(pred_list, axis=1))
    roc_auc = roc_auc_score(gt_list, pred_list[:, 1])
    pr_auc = average_precision_score(gt_list, pred_list[:, 1])
    plotter.record(epoch, f'{data_name}_accuracy', acc)
    plotter.record(epoch, f'{data_name}_pr_auc', pr_auc)
    if "target" in data_name and (("bag" in data_name) or (args_mode != "mil" and "classifier" in data_name) or ("instance" in data_name)):
        plotter.record(epoch, f'{data_name}_roc_auc', roc_auc)

    return pr_auc


def create_output_dir(args):
    # Create or Overwrite the log directory
    for past_log in os.listdir(os.path.join(os.path.curdir, args.log_dir)):
        if past_log[7:] == args.out or past_log[9:] == args.out:
            ans = input('overwrite "{}" (y/n)'.format(past_log))
            if ans == 'y' or ans == 'Y':
                print('move existing directory to dump')
                past_dir = os.path.join(os.path.curdir, args.log_dir, past_log)
                shutil.rmtree(past_dir)
            else:
                print('try again')
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

        os.makedirs(os.path.join(out_dir, 'plot'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'feature'), exist_ok=True)

    def save_files(self, files):
        for file_name, file in files.items():
            np.save(os.path.join(self.out_dir, 'feature', file_name), file)

    def save_models(self, models):
        for model_name, model in models.items():
            torch.save(model.state_dict(), os.path.join(self.out_dir, 'model', model_name))

    def record(self, epoch, name, value):
        self.logs[name][epoch] = value

    def refresh(self):
        self.logs = defaultdict(lambda: {})

    def flush(self, epoch):
        log = f"epoch {epoch}\n"

        for name, vals in self.logs.items():
            log += " {}\t{:.5f}\n".format(name, vals[epoch])

            x_vals = np.sort(list(vals.keys()))
            y_vals = [vals[x] for x in x_vals]

            plt.clf()
            plt.plot(x_vals, y_vals)
            plt.xlabel('iteration')
            plt.ylabel(name)
            plt.savefig(os.path.join(self.out_dir, 'plot', name.replace(' ', '_') + '.jpg'))

        with open(os.path.join(self.out_dir, 'log'), 'a+') as f:
            f.write(log + '\n')
