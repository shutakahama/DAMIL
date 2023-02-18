import os
import inspect
import time
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from utils import evaluate, progress_report, Plotter


class AdaptPMCDDA:
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

        if "get_params" in [obj[0] for obj in inspect.getmembers(self.encoder, inspect.ismethod)]:
            self.optimizer_e = optim.Adam(
                self.encoder.get_params(), lr=args.lr_adapt, weight_decay=args.weight_decay)
        else:
            self.optimizer_e = optim.Adam(
                self.encoder.parameters(), lr=args.lr_adapt, weight_decay=args.weight_decay)
        self.optimizer_c1 = optim.Adam(
            self.classifier1.parameters(), lr=args.lr_adapt, weight_decay=args.weight_decay)
        self.optimizer_c2 = optim.Adam(
            self.classifier2.parameters(), lr=args.lr_adapt, weight_decay=args.weight_decay)
        self.optimizer_a = optim.Adam(
            self.attention.parameters(), lr=args.lr_adapt, weight_decay=args.weight_decay)

        self.scheduler_e = StepLR(
            self.optimizer_e, step_size=args.lr_interval, gamma=args.lr_dim)
        self.scheduler_c1 = StepLR(
            self.optimizer_c1, step_size=args.lr_interval, gamma=args.lr_dim)
        self.scheduler_c2 = StepLR(
            self.optimizer_c2, step_size=args.lr_interval, gamma=args.lr_dim)
        self.scheduler_a = StepLR(
            self.optimizer_a, step_size=args.lr_interval, gamma=args.lr_dim)

        self.len_data_loader = min(
            len(self.data_loaders["source_train"].dataset),
            len(self.data_loaders["target_train"].dataset)
        )

    def reset_grad(self):
        self.optimizer_e.zero_grad()
        self.optimizer_c1.zero_grad()
        self.optimizer_c2.zero_grad()
        self.optimizer_a.zero_grad()

    def scheduler_step(self):
        self.scheduler_e.step()
        self.scheduler_c1.step()
        self.scheduler_c2.step()
        self.scheduler_a.step()

    @staticmethod
    def discrepancy(out1, out2):
        return torch.mean(
            torch.abs(torch.softmax(out1, dim=1) - torch.softmax(out2, dim=1)))

    def update(self, epoch, labeled_num, criterion):
        train_loss_m = train_loss_a = train_loss_d1 = train_loss_d2 = 0
        pred_cls_list = np.empty((0, 2), np.float32)
        gt_cls_list = np.empty(0, np.float32)
        start_time = time.time()
        labeled_sample_sum = [0]*int(self.args.bag_size_mean * 2)
        max_sample_num = labeled_num // len(self.data_loaders["target_train"]) + 1

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

            # for mix
            # 1. Optimize Classifier model with source data
            self.reset_grad()
            feat_m = self.encoder(mix_data)
            output_m1 = self.classifier1(feat_m)
            output_m2 = self.classifier2(feat_m)

            loss_m1 = criterion(output_m1, mix_label)
            loss_m2 = criterion(output_m2, mix_label)
            loss_m = loss_m1 + loss_m2
            loss_m.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()
            self.optimizer_e.step()
            train_loss_m += loss_m.data

            # 2. maximize discrepancy (MCDDA) for mix data
            self.reset_grad()
            feat_m = self.encoder(mix_data)
            output_m1 = self.classifier1(feat_m)
            output_m2 = self.classifier2(feat_m)
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

            # 3. minimize discrepancy (MCDDA)
            for i in range(3):
                self.reset_grad()
                feat_tgt = self.encoder(target_data)
                output_t1 = self.classifier1(feat_tgt)
                output_t2 = self.classifier2(feat_tgt)

                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.optimizer_e.step()
                train_loss_d2 += loss_dis.data

            # 4. Optimize Attention model (target data supervised learning)
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

            # calculate accuracy
            feat_tgt = self.encoder(target_data)
            output_t1 = self.classifier1(feat_tgt)
            output_t2 = self.classifier2(feat_tgt)
            output_t = (output_t1 + output_t2)
            pred_cls_list = np.append(pred_cls_list, np.array(F.softmax(output_t.data.cpu(), dim=1)), axis=0)
            gt_cls_list = np.append(gt_cls_list, np.array(target_instance_label.data.cpu()), axis=0)

            progress_report(epoch, step, start_time, self.args.batch_size, self.len_data_loader)

        self.plotter.record(epoch, 'train_mix_classifier_loss', train_loss_m / self.len_data_loader)
        self.plotter.record(epoch, 'train_attention_loss', train_loss_a / self.len_data_loader)
        self.plotter.record(epoch, 'train_max_dis_loss', train_loss_d1 / self.len_data_loader)
        self.plotter.record(epoch, 'train_min_dis_loss', train_loss_d2 / self.len_data_loader)
        evaluate(self.plotter, epoch, 'train_target_classifier', pred_cls_list, gt_cls_list)
        print(f"labeled sample num: {labeled_sample_sum}")

    def train(self):
        print("train")
        self.encoder.train()
        self.classifier1.train()
        self.classifier2.train()
        self.attention.train()

        criterion = nn.CrossEntropyLoss().to(self.args.device)
        labeled_num = 0

        for epoch in range(self.args.num_epochs_adapt):
            self.update(epoch, labeled_num, criterion)
            cls_score, ins_score = self.predict(
                epoch, self.data_loaders["source_valid"], split_type='valid', data_category='source')
            self.predict(epoch, self.data_loaders["target_valid"], split_type='valid', data_category='target')
            labeled_num = self.labeling(epoch, cls_score, ins_score)

            self.scheduler_step()
            self.plotter.flush(epoch)

    def predict(self, epoch, data_loader, split_type='test', data_category=None):
        print(f"{split_type} {data_category}")
        self.encoder.eval()
        self.classifier1.eval()
        self.classifier2.eval()
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

                progress_report(split_type, step, start_time, self.args.batch_size, len_data)

        self.plotter.record(epoch, f'{split_type}_{data_category}_classifier_loss', loss_cls / len_data)
        self.plotter.record(epoch, f'{split_type}_{data_category}_instance_loss', loss_ins / len_data)
        self.plotter.record(epoch, f'{split_type}_{data_category}_bag_loss', loss_bag / len_data)
        evaluate(self.plotter, epoch, f'{split_type}_{data_category}_bag', pred_bag_list, gt_bag_list)
        cls_score = evaluate(self.plotter, epoch, f'{split_type}_{data_category}_classifier', pred_cls_list, gt_cls_list)
        ins_score = evaluate(self.plotter, epoch, f'{split_type}_{data_category}_instance', pred_ins_list, gt_cls_list)

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
        print("pseudo labeling")
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

                if bag_label == 0:  # give labels only to positive bags
                    continue

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

                progress_report('labeling', step, start_time, self.args.batch_size, len_data)

        labeling_num = self.give_pseudo_label(
            epoch, pred1_list, pred2_list, gt_list, idx_list, cls_score, ins_score)

        return labeling_num

    def give_pseudo_label(self, epoch, pred1, pred2, gt_list, idx_list, cls_score, ins_score):
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

        print(f"Labeled Number: pos -> {len(pos_idx)}, neg -> {len(neg_idx)}")
        if num > 0:
            pos_score_min = -np.sort(-pos_val_cand)[num_pos - 1]
            neg_score_min = -np.sort(-pos_val_cand)[num_pos - 1]
            print(f"Minimum labeling score: pos -> {pos_score_min} neg -> {neg_score_min}")

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
