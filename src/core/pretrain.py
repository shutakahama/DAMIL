import os
import time
import inspect
import numpy as np
import logging
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from utils import evaluate, progress_report, Plotter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PretrainDAMIL:
    def __init__(self, out_dir, encoder, classifier, attention, data_loaders, args):

        self.plotter = Plotter(os.path.join(out_dir, "pretrain"))
        self.encoder = encoder
        self.classifier = classifier
        self.attention = attention
        self.data_loaders = data_loaders
        self.args = args

        if "get_params" in [obj[0] for obj in inspect.getmembers(self.encoder, inspect.ismethod)]:
            self.optimizer_e = optim.Adam(
                self.encoder.get_params(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
        else:
            self.optimizer_e = optim.Adam(
                self.encoder.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
        self.optimizer_c = optim.Adam(
            self.classifier.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
        self.optimizer_a = optim.Adam(
            self.attention.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)

        self.len_data_loader = min(
            len(self.data_loaders["source_train"].dataset),
            len(self.data_loaders["target_train"].dataset)
        )

    def reset_grad(self):
        self.optimizer_e.zero_grad()
        self.optimizer_c.zero_grad()
        self.optimizer_a.zero_grad()

    def train(self):
        self.encoder.train()
        self.classifier.train()
        self.attention.train()

        criterion = nn.CrossEntropyLoss().to(self.args.device)

        for epoch in range(self.args.num_epochs_pre):
            logger.info(f"epoch: {epoch}")
            train_loss_s = train_loss_t = 0
            pred_cls_list = np.empty((0, 2), np.float32)
            pred_bag_list = np.empty((0, 2), np.float32)
            pred_ins_list = np.empty((0, 2), np.float32)
            gt_cls_list = np.empty(0, np.float32)
            gt_bag_list = np.empty(0, np.float32)
            gt_ins_list = np.empty(0, np.float32)
            start_time = time.time()

            data_zip = enumerate(zip(self.data_loaders["source_train"], self.data_loaders["target_train"]))

            for step, (source_batch, target_batch) in data_zip:
                source_data, source_label, _ = source_batch
                target_data, target_label, _ = target_batch

                source_data = source_data.squeeze(0)
                target_data = target_data.squeeze(0)

                source_bag_label = source_label[0].long()
                source_instance_label = source_label[1].reshape(-1).long()
                target_bag_label = target_label[0].long()
                target_instance_label = target_label[1].reshape(-1).long()

                source_data = source_data.requires_grad_().to(self.args.device)
                target_data = target_data.requires_grad_().to(self.args.device)
                source_bag_label = source_bag_label.to(self.args.device)
                source_instance_label = source_instance_label.to(self.args.device)
                target_bag_label = target_bag_label.to(self.args.device)
                target_instance_label = target_instance_label.to(self.args.device)

                # Optimize classifier and attention
                self.reset_grad()
                output_s = self.classifier(self.encoder(source_data))
                pred_tgt, _ = self.attention(self.encoder(target_data))

                loss_s = criterion(output_s, source_instance_label)
                loss_t = criterion(pred_tgt, target_bag_label)
                loss = loss_s + loss_t
                loss.backward()
                self.optimizer_a.step()
                self.optimizer_e.step()
                self.optimizer_c.step()
                train_loss_s += loss_s.data
                train_loss_t += loss_t.data

                # calculate accuracy
                output_cls = self.classifier(self.encoder(source_data))
                output_bag, _output_att = self.attention(self.encoder(target_data))
                _output_att = torch.unsqueeze(_output_att.reshape(-1), dim=0)
                output_att = torch.cat((1 - _output_att, _output_att), dim=0).T

                pred_cls_list = np.append(pred_cls_list, np.array(F.softmax(output_cls.data.cpu(), dim=1)), axis=0)
                pred_bag_list = np.append(pred_bag_list, np.array(F.softmax(output_bag.data.cpu(), dim=1)), axis=0)
                pred_ins_list = np.append(pred_ins_list, np.array(F.softmax(output_att.data.cpu(), dim=1)), axis=0)
                gt_cls_list = np.append(gt_cls_list, np.array(source_instance_label.data.cpu()), axis=0)
                gt_bag_list = np.append(gt_bag_list, np.max(target_instance_label.data.cpu().numpy(), keepdims=True), axis=0)
                gt_ins_list = np.append(gt_ins_list, np.array(target_instance_label.data.cpu()), axis=0)

                is_last = (step + 1 == self.len_data_loader)
                progress_report("train", step, start_time, 1, self.len_data_loader, is_last)

            self.plotter.record(epoch, 'train_source_loss', train_loss_s/self.len_data_loader)
            self.plotter.record(epoch, 'train_target_loss', train_loss_t/self.len_data_loader)
            evaluate(self.plotter, epoch, 'train_source_classifier', pred_cls_list, gt_cls_list)
            evaluate(self.plotter, epoch, 'train_target_bag', pred_bag_list, gt_bag_list)
            evaluate(self.plotter, epoch, 'train_target_instance', pred_ins_list, gt_ins_list)

            self.predict(epoch, self.data_loaders["source_valid"], split_type='valid', data_category='source')
            self.predict(epoch, self.data_loaders["target_valid"], split_type='valid', data_category='target')
            self.plotter.flush(epoch)

    def predict(self, epoch, data_loader, split_type='test', data_category='source'):
        self.encoder.eval()
        self.classifier.eval()
        self.attention.eval()
        criterion = nn.CrossEntropyLoss().to(self.args.device)

        # init loss and accuracy
        len_data = len(data_loader.dataset)
        loss_cls = loss_bag = loss_ins = 0
        pred_cls_list = np.empty((0, self.args.num_class), np.float32)
        pred_bag_list = np.empty((0, self.args.num_class), np.float32)
        pred_ins_list = np.empty((0, self.args.num_class), np.float32)
        gt_cls_list = np.empty(0, np.float32)
        gt_bag_list = np.empty(0, np.float32)
        feature_list = np.empty((0, self.args.feat_dim), np.float32)
        start_time = time.time()
        phase = f"{split_type} {data_category}"

        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                data, label, _ = batch
                bag_label = label[0].long()
                instance_label = label[1].reshape(-1).long()

                data = data.squeeze(0)
                data = data.requires_grad_().to(self.args.device)
                bag_label = bag_label.to(self.args.device)
                instance_label = instance_label.to(self.args.device)

                mid_feature = self.encoder(data)
                pred_cls = self.classifier(mid_feature)
                pred_bag, _pred_att = self.attention(mid_feature)
                _pred_att = torch.unsqueeze(_pred_att.reshape(-1), dim=0)
                pred_att = torch.cat((1 - _pred_att, _pred_att), dim=0).T

                loss_cls += criterion(pred_cls, instance_label).data
                pred_cls_list = np.append(pred_cls_list, np.array(F.softmax(pred_cls.data.cpu(), dim=1)), axis=0)
                gt_cls_list = np.append(gt_cls_list, np.array(instance_label.data.cpu()), axis=0)
                feature_list = np.append(feature_list, np.array(mid_feature.data.cpu()), axis=0)
                if data_category == "target":
                    loss_bag += criterion(pred_bag, bag_label).data
                    loss_ins += criterion(pred_att, instance_label).data
                    pred_bag_list = np.append(pred_bag_list, np.array(F.softmax(pred_bag.data.cpu(), dim=1)), axis=0)
                    pred_ins_list = np.append(pred_ins_list, np.array(F.softmax(pred_att.data.cpu(), dim=1)), axis=0)
                    gt_bag_list = np.append(gt_bag_list, np.max(instance_label.data.cpu().numpy(), keepdims=True), axis=0)

                is_last = (step + 1 == len_data)
                progress_report(phase, step, start_time, 1, len_data, is_last)

        self.plotter.record(epoch, f'{split_type}_{data_category}_classifier_loss', loss_cls / len_data)
        score = evaluate(self.plotter, epoch, f'{split_type}_{data_category}_classifier', pred_cls_list, gt_cls_list)
        if data_category == "target":
            self.plotter.record(epoch, f'{split_type}_{data_category}_bag_loss', loss_bag / len_data)
            self.plotter.record(epoch, f'{split_type}_{data_category}_instance_loss', loss_ins / len_data)
            evaluate(self.plotter, epoch, f'{split_type}_{data_category}_bag', pred_bag_list, gt_bag_list)
            evaluate(self.plotter, epoch, f'{split_type}_{data_category}_instance', pred_ins_list, gt_cls_list)

        if data_category == "target" and split_type == 'valid':
            self.plotter.save_files({
                f"feature_{data_category}.npy": feature_list,
                f"label_{data_category}.npy": gt_cls_list
            }, score)
            self.plotter.save_models({
                "encoder.model": self.encoder,
                "classifier.model": self.classifier,
                "attention.model": self.attention
            }, score)
