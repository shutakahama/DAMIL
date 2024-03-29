import os
import time
import numpy as np
import logging
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

from utils import evaluate, progress_report, Plotter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptDAMIL:
    """Model Traning Step 2 and 3.
    - Step 2 uses update().
    Train models with source data and pseudo-labeled target data.
    In addition, the feature matching loss of MCD-DA is included.
    - Step 3: uses labeling().
    Give pseudo-labels to the target data with high confidence scores.

        Args:
            out_dir (str): logging output directory name.
            encoder : Encoder model class.
            classifier1 : Instance classifier model class.
            classifier2 : Instance classifier model class.
            attention : Bag classifier model class
            data_loaders (dict): Dictionary of data loader class.
                This must include "source_train", "target_train",
                "source_valid", and "taret_valid".
            args : Other parameters.
        """
    def __init__(self, out_dir, encoder, classifier1, classifier2, attention, data_loaders, args):
        self.out_dir = out_dir
        self.plotter = Plotter(os.path.join(out_dir, "adapt"))
        self.encoder = encoder
        self.classifier1 = classifier1
        self.classifier2 = classifier2
        self.attention = attention
        self.data_loaders = data_loaders
        self.pseudo_label_dict = dict()
        self.args = args

        self.optimizer_e = optim.Adam(
            self.encoder.parameters(), lr=args.lr_adapt, weight_decay=args.weight_decay)
        self.optimizer_c1 = optim.Adam(
            self.classifier1.parameters(), lr=args.lr_adapt, weight_decay=args.weight_decay)
        self.optimizer_c2 = optim.Adam(
            self.classifier2.parameters(), lr=args.lr_adapt, weight_decay=args.weight_decay)
        self.optimizer_a = optim.Adam(
            self.attention.parameters(), lr=args.lr_adapt, weight_decay=args.weight_decay)

        self.len_data_loader = min(
            len(self.data_loaders["source_train"].dataset),
            len(self.data_loaders["target_train"].dataset)
        )

    def reset_grad(self):
        self.optimizer_e.zero_grad()
        self.optimizer_c1.zero_grad()
        self.optimizer_c2.zero_grad()
        self.optimizer_a.zero_grad()

    @staticmethod
    def discrepancy(out1, out2):
        return torch.mean(
            torch.abs(torch.softmax(out1, dim=1) - torch.softmax(out2, dim=1)))

    def update(self, epoch, labeled_num):
        """Train models with 4 steps.

        Args:
            epoch (int): Parameter to indicate the current epoch.
            labeled_num (int): The number of labeled target instances
                in the previous epoch.
        """
        train_loss_c = train_loss_a = train_loss_d1 = train_loss_d2 = 0
        pred_cls_list = np.empty((0, 2), np.float32)
        gt_cls_list = np.empty(0, np.float32)
        start_time = time.time()
        labeled_sample_sum = [0]*int(self.args.bag_size_mean * 2)
        max_sample_num = labeled_num // len(self.data_loaders["target_train"]) + 1

        criterion = nn.CrossEntropyLoss().to(self.args.device)

        data_zip = enumerate(zip(
            self.data_loaders["source_train"], self.data_loaders["target_train"]))

        for step, (source_batch, target_batch) in data_zip:
            source_data, source_label, _ = source_batch
            target_data, target_label, target_idx = target_batch

            source_bag_label = source_label[0].long()
            source_instance_label = source_label[1].reshape(-1).long()
            target_bag_label = target_label[0].long()
            target_instance_label = target_label[1].reshape(-1).long()
            target_idx = target_idx.reshape(-1)

            source_data = source_data.squeeze(0)
            target_data = target_data.squeeze(0)

            # Create mix data by combining source data and pseudo-labeled target data.
            # If target_bag_label = 0, all data have negative labels so we sample
            # {sample_num} data randomly.
            # If target_bag_label = 1, select labeled data with the pseudo_label_dict.
            if target_bag_label == 0:
                sample_num = min(len(target_data), max_sample_num)
                perm = np.random.permutation(len(target_data))[:sample_num]
                labeled_target_data = target_data[perm]
                labeled_target_label = torch.zeros(sample_num).long()
            else:
                labeled_target_idx_list, labeled_target_label_list = [], []
                for i in range(len(target_idx)):
                    if int(target_idx[i]) in self.pseudo_label_dict:
                        labeled_target_idx_list.append(i)
                        labeled_target_label_list.append(self.pseudo_label_dict[int(target_idx[i])])

                labeled_sample_sum[len(labeled_target_idx_list)] += 1
                if labeled_target_idx_list:
                    labeled_target_data = target_data[labeled_target_idx_list]
                    labeled_target_label = torch.Tensor(labeled_target_label_list).long()

            if target_bag_label == 1 and len(labeled_target_idx_list) == 0:
                # No labeled target data
                mix_data = source_data
                mix_label = source_instance_label
            else:
                mix_data = torch.cat((source_data, labeled_target_data), dim=0)
                mix_label = torch.cat((source_instance_label, labeled_target_label))
                # Mix source and target
                perm = np.random.permutation(len(mix_data))
                mix_data = mix_data[perm]
                mix_label = mix_label[perm]

            mix_data = mix_data.requires_grad_().to(self.args.device)
            mix_label = mix_label.long().to(self.args.device)
            source_data = source_data.requires_grad_().to(self.args.device)
            source_bag_label = source_bag_label.to(self.args.device)
            source_instance_label = source_instance_label.to(self.args.device)
            target_data = target_data.requires_grad_().to(self.args.device)
            target_bag_label = target_bag_label.to(self.args.device)
            target_instance_label = target_instance_label.to(self.args.device)

            # Training steps
            # Step 1. Optimize classifier models with mix data
            self.reset_grad()
            feat_mix = self.encoder(mix_data)
            output_m1 = self.classifier1(feat_mix)
            output_m2 = self.classifier2(feat_mix)

            loss_m1 = criterion(output_m1, mix_label)
            loss_m2 = criterion(output_m2, mix_label)
            loss_c = loss_m1 + loss_m2
            loss_c.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()
            self.optimizer_e.step()
            train_loss_c += loss_c.data

            # Step 2. Maximize discrepancy (MCDDA) for mix data
            self.reset_grad()
            feat_mix = self.encoder(mix_data)
            output_m1 = self.classifier1(feat_mix)
            output_m2 = self.classifier2(feat_mix)
            feat_tgt = self.encoder(target_data)
            output_t1 = self.classifier1(feat_tgt)
            output_t2 = self.classifier2(feat_tgt)

            loss_m1 = criterion(output_m1, mix_label)
            loss_m2 = criterion(output_m2, mix_label)
            loss_m = loss_m1 + loss_m2
            loss_dis = self.discrepancy(output_t1, output_t2)
            loss = loss_m - loss_dis
            loss.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()
            train_loss_d1 += loss.data

            # Step 3. Minimize discrepancy with MCD-DA loss
            for i in range(3):
                self.reset_grad()
                feat_tgt = self.encoder(target_data)
                output_t1 = self.classifier1(feat_tgt)
                output_t2 = self.classifier2(feat_tgt)

                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.optimizer_e.step()
                train_loss_d2 += loss_dis.data

            # Step 4. Optimize Attention model with source and target data
            self.reset_grad()
            feat_src = self.encoder(source_data)
            pred_src, _ = self.attention(feat_src)
            feat_tgt = self.encoder(target_data)
            pred_tgt, _ = self.attention(feat_tgt)

            loss_as = criterion(pred_src, source_bag_label)
            loss_at = criterion(pred_tgt, target_bag_label)
            loss_a = loss_as + loss_at
            loss_a.backward()
            self.optimizer_a.step()
            self.optimizer_e.step()
            train_loss_a += loss_a.data

            # Calculate accuracy
            feat_tgt = self.encoder(target_data)
            output_t1 = self.classifier1(feat_tgt)
            output_t2 = self.classifier2(feat_tgt)
            output_t = (output_t1 + output_t2)
            pred_cls_list = np.append(pred_cls_list, np.array(F.softmax(output_t.data.cpu(), dim=1)), axis=0)
            gt_cls_list = np.append(gt_cls_list, np.array(target_instance_label.data.cpu()), axis=0)

            is_last = (step + 1 == self.len_data_loader)
            progress_report("train", step, start_time, 1, self.len_data_loader, is_last)

        # Record metrics
        self.plotter.record(epoch, 'train_classifier_loss', train_loss_c / self.len_data_loader)
        self.plotter.record(epoch, 'train_attention_loss', train_loss_a / self.len_data_loader)
        self.plotter.record(epoch, 'train_max_dis_loss', train_loss_d1 / self.len_data_loader)
        self.plotter.record(epoch, 'train_min_dis_loss', train_loss_d2 / self.len_data_loader)
        evaluate(self.plotter, epoch, 'train_target_classifier', pred_cls_list, gt_cls_list)
        # logger.info(f"Distribution of the number of labeled samples: {labeled_sample_sum}")

    def train(self):
        """Core function to train models
        """
        self.encoder.train()
        self.classifier1.train()
        self.classifier2.train()
        self.attention.train()

        labeled_num = 0

        for epoch in range(self.args.num_epochs_adapt):
            logger.info(f"epoch: {epoch}")
            self.update(epoch, labeled_num)
            cls_score, ins_score = self.predict(
                epoch, self.data_loaders["source_valid"], split_type='valid', data_category='source')
            self.predict(
                epoch, self.data_loaders["target_valid"], split_type='valid', data_category='target')
            labeled_num = self.labeling(epoch, cls_score, ins_score)

            self.plotter.flush(epoch)

    def predict(self, epoch, data_loader, split_type, data_category):
        """Calculate prediction scores of trained models

        Args:
            epoch (int): Parameter to indicate the current epoch
            data_loader : Data loader class for prediction
            split_type (str): 'valid' or 'test'
            data_category (str): 'source' or 'target'

        Returns:
            cls_score : PR-AUC score of predictions by the instance classifier
            ins_score : PR-AUC score of predictions by the bag classifier
        """
        self.encoder.eval()
        self.classifier1.eval()
        self.classifier2.eval()
        self.attention.eval()
        criterion = nn.CrossEntropyLoss().to(self.args.device)

        # Initialize loss and accuracy scores
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
                pred_cls1 = self.classifier1(mid_feature)
                pred_cls2 = self.classifier2(mid_feature)
                pred_cls = (pred_cls1 + pred_cls2)
                pred_bag, _pred_att = self.attention(mid_feature)
                _pred_att = torch.unsqueeze(_pred_att.reshape(-1), dim=0)
                pred_att = torch.cat((1 - _pred_att, _pred_att), dim=0).T

                loss_cls += criterion(pred_cls, instance_label).data
                loss_bag += criterion(pred_bag, bag_label).data
                loss_ins += criterion(pred_att, instance_label).data
                pred_cls_list = np.append(pred_cls_list, np.array(F.softmax(pred_cls.data.cpu(), dim=1)), axis=0)
                pred_bag_list = np.append(pred_bag_list, np.array(F.softmax(pred_bag.data.cpu(), dim=1)), axis=0)
                pred_ins_list = np.append(pred_ins_list, np.array(pred_att.data.cpu()), axis=0)
                gt_cls_list = np.append(gt_cls_list, np.array(instance_label.data.cpu()), axis=0)
                gt_bag_list = np.append(gt_bag_list, np.array(bag_label.data.cpu()), axis=0)
                feature_list = np.append(feature_list, np.array(mid_feature.data.cpu()), axis=0)

                is_last = (step + 1 == len_data)
                progress_report(phase, step, start_time, 1, len_data, is_last)

        # Record metrics
        self.plotter.record(epoch, f'{split_type}_{data_category}_classifier_loss', loss_cls / len_data)
        self.plotter.record(epoch, f'{split_type}_{data_category}_instance_loss', loss_ins / len_data)
        self.plotter.record(epoch, f'{split_type}_{data_category}_bag_loss', loss_bag / len_data)
        evaluate(self.plotter, epoch, f'{split_type}_{data_category}_bag', pred_bag_list, gt_bag_list)
        cls_score = evaluate(
            self.plotter, epoch, f'{split_type}_{data_category}_classifier', pred_cls_list, gt_cls_list)
        ins_score = evaluate(
            self.plotter, epoch, f'{split_type}_{data_category}_instance', pred_ins_list, gt_cls_list)

        if data_category == "target" and split_type == 'valid':
            self.plotter.save_files({
                f'feature_{data_category}.npy': feature_list,
                f'label_{data_category}.npy': gt_cls_list,
            }, cls_score)
            self.plotter.save_models({
                "encoder.model": self.encoder,
                "classifier1.model": self.classifier1,
                "classifier2.model": self.classifier2,
                "attention.model": self.attention
            }, cls_score)

        return cls_score, ins_score

    def labeling(self, epoch, cls_score, ins_score):
        """Give pseudo-labels to target instances in the train set.

        Args:
            epoch (int): Parameter to indicate the current epoch
            cls_score (float): PR-AUC score of the instance classifier prediction
            ins_score (float): PR-AUC score of by the bag classifier prediction

        Returns:
            labeling_num (int): The number of labeled target instances
        """
        self.encoder.eval()
        self.classifier1.eval()
        self.classifier2.eval()
        self.attention.eval()

        # init loss and accuracy
        len_data = len(self.data_loaders["target_train"].dataset)
        pred1_list = np.empty((0, self.args.num_class), np.float32)
        pred2_list = np.empty((0, self.args.num_class), np.float32)
        gt_list = np.empty(0, np.int32)
        idx_list = np.empty(0, np.int32)
        start_time = time.time()

        with torch.no_grad():
            for step, batch in enumerate(self.data_loaders["target_train"]):
                data, label, index = batch
                bag_label = label[0].long()
                instance_label = label[1].reshape(-1).long()
                index = index.reshape(-1)
                data = data.squeeze(0)
                data = data.requires_grad_().to(self.args.device)

                # give labels only to positive bags
                if bag_label == 1:
                    mid_feature = self.encoder(data)
                    pred_cls1 = self.classifier1(mid_feature)
                    pred_cls2 = self.classifier2(mid_feature)
                    pred_cls = (pred_cls1 + pred_cls2) / 2
                    _, _pred_att = self.attention(mid_feature)
                    _pred_att = torch.unsqueeze(_pred_att.reshape(-1), dim=0)
                    pred_att = torch.cat((1 - _pred_att, _pred_att), dim=0).T
                    pred1_list = np.append(pred1_list, np.array(F.softmax(pred_cls.data.cpu(), dim=1)), axis=0)
                    pred2_list = np.append(pred2_list, np.array(F.softmax(pred_att.data.cpu(), dim=1)), axis=0)
                    gt_list = np.append(gt_list, np.array(instance_label), axis=0)
                    idx_list = np.append(idx_list, np.array(index), axis=0)

                is_last = (step + 1 == len_data)
                progress_report("pseudo labeling", step, start_time, 1, len_data, is_last)

        labeling_num = self.give_pseudo_label(
            epoch, pred1_list, pred2_list, gt_list, idx_list, cls_score, ins_score)

        return labeling_num

    def give_pseudo_label(self, epoch, pred1, pred2, gt_list, idx_list, cls_score, ins_score):
        """Core function to choose instances to give pseudo-labels

        Args:
            epoch (int): Parameter to indicate the current epoch
            pred1 : Numpy array of prediction scores by the instance classifier
            pred2 : Numpy array of prediction scores by the bag classifier
            gt_list : Numpy array of true labels (Only for experimental evaluation)
            idx_list : Numpy array of data indexes
            cls_score (float): PR-AUC score of the instance classifier prediction
            ins_score (float): PR-AUC score of by the bag classifier prediction

        Returns:
            labeling_num (int): The number of labeled target instances
        """
        # Calculate candidate index with weighted confidence score
        weighted_val = pred1 * cls_score + pred2 * ins_score
        max_idx = np.argmax(weighted_val, axis=1)
        pos_idx_cand = np.where((weighted_val[:, 1] >= self.args.labeling_thre) & (max_idx == 1))[0]
        neg_idx_cand = np.where((weighted_val[:, 0] >= self.args.labeling_thre) & (max_idx == 0))[0]
        pos_val_cand, neg_val_cand = weighted_val[:, 1][pos_idx_cand], weighted_val[:, 0][neg_idx_cand]

        # Give labels to items with high confidence scores
        n_max, n_min = len(pred1) // 10, len(pred1) // 30
        num = min(n_min + ((n_max - n_min) // 20) * epoch, n_max)
        num = min(num, len(pos_idx_cand), len(neg_idx_cand))
        num_pos, num_neg = num, 3*num
        _pos_idx, _neg_idx = np.argsort(-pos_val_cand)[:num_pos], np.argsort(-neg_val_cand)[:num_neg]
        pos_idx, neg_idx = pos_idx_cand[_pos_idx], neg_idx_cand[_neg_idx]

        logger.info(f"Number of labeled instances: pos -> {len(pos_idx)}, neg -> {len(neg_idx)}")
        if num > 0:
            pos_score_min = -np.sort(-pos_val_cand)[num_pos - 1]
            neg_score_min = -np.sort(-pos_val_cand)[num_pos - 1]
            logger.info(f"Munimum labeling score: pos -> {pos_score_min} neg -> {neg_score_min}")

        # Evaluate label accuracy
        true_label = np.concatenate((gt_list[pos_idx], gt_list[neg_idx]))
        new_label = np.concatenate((np.ones(len(pos_idx)), np.zeros(len(neg_idx))))
        pseudo_label_acc = np.mean(new_label == true_label) if len(new_label) > 0 else 0.0
        self.plotter.record(epoch, "pseudo_label_accuracy", pseudo_label_acc)
        self.plotter.record(epoch, "pseudo_label_number", len(new_label))

        # Update pseudo labels
        self.pseudo_label_dict.clear()
        for pi in pos_idx:
            self.pseudo_label_dict[idx_list[pi]] = 1
        for ni in neg_idx:
            self.pseudo_label_dict[idx_list[ni]] = 0

        return len(new_label)
