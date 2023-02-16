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


class AdaptPMCDDA(object):
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
            print("fixed model")
            self.optimizer_e = optim.Adam(
                self.encoder.get_params(), lr=args.lr_e, weight_decay=args.wd)
        else:
            self.optimizer_e = optim.Adam(
                self.encoder.parameters(), lr=args.lr_e, weight_decay=args.wd)
        self.optimizer_c1 = optim.Adam(
            self.classifier1.parameters(), lr=args.lr_c, weight_decay=args.wd)
        self.optimizer_c2 = optim.Adam(
            self.classifier2.parameters(), lr=args.lr_c, weight_decay=args.wd)
        self.optimizer_a = optim.Adam(
            self.attention.parameters(), lr=args.lr_a, weight_decay=args.wd)

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
        # return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))
        return torch.mean(
            torch.abs(torch.softmax(out1, dim=1) - torch.softmax(out2, dim=1)))

    def update(self, epoch, labeing_num, criterion):
        train_loss_m = train_loss_a = train_loss_d1 = train_loss_d2 = 0
        pred_cls_list = np.empty((0, 2), np.float32)
        gt_cls_list = np.empty(0, np.float32)
        start_time = time.time()
        labeled_sample_sum = [0]*int(self.args.mean * 2)

        print("sample_num", min(self.args.mean, 2*labeing_num//len(self.data_loaders["target_train"]) + 1))

        data_zip = enumerate(zip(
            self.data_loaders["source_train"], self.data_loaders["target_train"]))
        for step, (source_batch, target_batch) in data_zip:
            source_data, source_label, source_idx = source_batch
            target_data, target_label, target_idx = target_batch

            source_bag_label = source_label[0].long()
            source_instance_label = source_label[1].reshape(-1).long()
            target_bag_label = target_label[0].long()
            target_instance_label = target_label[1].reshape(-1).long()
            target_idx = target_idx.reshape(-1)

            source_data = source_data.squeeze(0)
            target_data = target_data.squeeze(0)

            mix_mode = "mix"

            if target_bag_label == 0:
                # sample_num = 5
                # dynamic sampling number
                sample_num = min(
                    len(target_data),
                    labeing_num//len(self.data_loaders["target_train"]) + 1
                )
                perm = np.random.permutation(len(target_data))[:sample_num]
                labeled_target_data = target_data[perm]
                labeled_target_label = torch.zeros(sample_num).long()
            else:
                labeled_target_idx_list, labeled_target_label_list = [], []
                for i in range(len(target_idx)):
                    if int(target_idx[i]) in self.pseudo_label_dict:
                        labeled_target_idx_list.append(i)
                        labeled_target_label_list.append(self.pseudo_label_dict[int(target_idx[i])])

                if labeled_target_idx_list:
                    labeled_target_data = target_data[labeled_target_idx_list]
                    labeled_target_label = torch.Tensor(labeled_target_label_list).long()

            target_exist = True if (target_bag_label == 0 or labeled_target_idx_list) else False

            if target_bag_label == 1:
                labeled_sample_sum[len(labeled_target_idx_list)] += 1

            if mix_mode == "sep" and target_exist:
                labeled_target_data = labeled_target_data
                labeled_target_data = labeled_target_data.requires_grad_().to(self.args.device)
                labeled_target_label = labeled_target_label.to(self.args.device)
            elif mix_mode == "mix":
                if self.args.mode == "wolabel" or not target_exist:
                    mix_data = source_data
                    mix_label = source_instance_label
                else:
                    mix_data = torch.cat((source_data, labeled_target_data), dim=0)
                    mix_label = torch.cat((source_instance_label, labeled_target_label))
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
            if mix_mode == "mix":
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

            # for sep
            else:
                # 1. Optimize Classifier model with source data
                self.reset_grad()
                feat_s = self.encoder(source_data)
                output_m1 = self.classifier1(feat_s)
                output_m2 = self.classifier2(feat_s)

                loss_m1 = criterion(output_m1, source_instance_label)
                loss_m2 = criterion(output_m2, source_instance_label)
                loss_m = loss_m1 + loss_m2
                loss_m.backward()
                self.optimizer_c1.step()
                self.optimizer_c2.step()
                self.optimizer_e.step()
                train_loss_m += loss_m.data

                # 2. Optimize Classifier model with labeled target data
                if target_exist:
                    self.reset_grad()
                    feat_t = self.encoder(labeled_target_data)
                    output_t1 = self.classifier1(feat_t)
                    output_t2 = self.classifier2(feat_t)

                    loss_t1 = criterion(output_t1, labeled_target_label)
                    loss_t2 = criterion(output_t2, labeled_target_label)
                    loss_t = loss_t1 + loss_t2
                    loss_t.backward()
                    self.optimizer_c1.step()
                    self.optimizer_c2.step()
                    self.optimizer_e.step()
                    train_loss_m += loss_t.data

            # 2. maximize discrepancy (MCDDA) for mix data
            if self.args.mode != "womcd":
                # for i in range(self.num_k):
                for i in range(1):
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
                # for i in range(self.num_k):
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
            for i in range(1):
                # source
                self.reset_grad()
                feat_src = self.encoder(source_data)
                pred_src, _ = self.attention(feat_src)
                loss_a = criterion(pred_src, source_bag_label)
                loss_a.backward()
                self.optimizer_a.step()
                self.optimizer_e.step()
                train_loss_a += loss_a.data

                # target
                self.reset_grad()
                feat_tgt = self.encoder(target_data)
                pred_tgt, _ = self.attention(feat_tgt)
                loss_a = criterion(pred_tgt, target_bag_label)
                loss_a.backward()
                self.optimizer_a.step()
                self.optimizer_e.step()
                train_loss_a += loss_a.data

            # calculate accuracy
            feat_tgt = self.encoder(target_data)
            output_t1 = self.classifier1(feat_tgt)
            output_t2 = self.classifier2(feat_tgt)
            output_t = (output_t1 + output_t2)
            # output_t = output_t1
            pred_cls_list = np.append(pred_cls_list, np.array(output_t.data.cpu()), axis=0)  # softmax
            gt_cls_list = np.append(gt_cls_list, np.array(target_instance_label.data.cpu()), axis=0)

            progress_report(epoch, step, start_time, self.args.batch_size, self.len_data_loader)

        self.plotter.record(epoch, 'train_mix_classifier_loss', train_loss_m / self.len_data_loader)
        self.plotter.record(epoch, 'train_attention_loss', train_loss_a / self.len_data_loader)
        self.plotter.record(epoch, 'train_max_dis_loss', train_loss_d1 / self.len_data_loader)
        self.plotter.record(epoch, 'train_min_dis_loss', train_loss_d2 / self.len_data_loader)
        evaluate(self.plotter, epoch, 'train_target_classifier', pred_cls_list, gt_cls_list, self.args.mode)
        print(f"labeled sample num: {labeled_sample_sum}")

    def train(self):
        print("train")
        self.encoder.train()
        self.classifier1.train()
        self.classifier2.train()
        self.attention.train()

        criterion = nn.CrossEntropyLoss().to(self.args.device)
        labeling_num = cls_acc = ins_acc = 0

        for epoch in range(self.args.num_epochs):
            self.update(epoch, labeling_num, criterion)
            cls_acc, ins_acc = self.test(epoch, self.data_loaders["source_test"], data_category='source')
            self.test(epoch, self.data_loaders["target_test"], data_category='target')
            labeling_num = self.labeling(epoch, cls_acc, ins_acc)

            self.scheduler_step()
            self.plotter.flush(epoch)


    def test(self, epoch, test_data_loader, data_category=None):
        print(f"{data_category} test")
        self.encoder.eval()
        self.classifier1.eval()
        self.classifier2.eval()
        self.attention.eval()
        criterion = nn.CrossEntropyLoss().to(self.args.device)

        # init loss and accuracy
        len_data = len(test_data_loader.dataset)
        loss_cls = loss_bag = loss_ins = 0
        pred_cls_list = np.empty((0, self.args.num_class), np.float32)
        pred_bag_list = np.empty((0, self.args.num_class), np.float32)
        pred_ins_list = np.empty((0, self.args.num_class), np.float32)
        gt_cls_list = np.empty(0, np.float32)
        gt_bag_list = np.empty(0, np.float32)
        feature_list = np.empty((0, self.args.feat_dim), np.float32)
        start_time = time.time()

        # evaluate network
        with torch.no_grad():
            for step, batch in enumerate(test_data_loader):
                data, label = batch

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
                # pred_cls = pred_cls1
                pred_bag, _pred_att = self.attention(mid_feature)
                _pred_att = torch.unsqueeze(_pred_att.reshape(-1), dim=0)
                pred_att = torch.cat((1 - _pred_att, _pred_att), dim=0).T

                loss_cls += criterion(pred_cls, instance_label).data
                loss_bag += criterion(pred_bag, bag_label).data
                loss_ins += criterion(pred_att, instance_label).data
                # 流石にsoftmaxかけるべきでは？
                pred_cls_list = np.append(pred_cls_list, np.array(pred_cls.data.cpu()), axis=0)  # softmax
                pred_bag_list = np.append(pred_bag_list, np.array(pred_bag.data.cpu()), axis=0)
                pred_ins_list = np.append(pred_ins_list, np.array(pred_att.data.cpu()), axis=0)
                gt_cls_list = np.append(gt_cls_list, np.array(instance_label.data.cpu()), axis=0)
                gt_bag_list = np.append(gt_bag_list, np.array(bag_label.data.cpu()), axis=0)
                feature_list = np.append(feature_list, np.array(mid_feature.data.cpu()), axis=0)

                progress_report('test', step, start_time, self.args.batch_size, len_data)

        self.plotter.record(epoch, f'test_{data_category}_classifier_loss', loss_cls / len_data)
        cls_acc = evaluate(self.plotter, epoch, f'test_{data_category}_classifier', pred_cls_list, gt_cls_list, self.args.mode)
        ins_acc = evaluate(self.plotter, epoch, f'test_{data_category}_instance', pred_ins_list, gt_cls_list, self.args.mode)
        self.plotter.record(epoch, f'test_{data_category}_instance_loss', loss_ins / len_data)
        self.plotter.record(epoch, f'test_{data_category}_bag_loss', loss_bag / len_data)
        evaluate(self.plotter, epoch, f'test_{data_category}_bag', pred_bag_list, gt_bag_list, self.args.mode)

        self.plotter.save_files({
            f'adapt_feature_{data_category}.npy': feature_list,
            f'adapt_label_{data_category}.npy': gt_cls_list,
        })
        self.plotter.save_models({
            "encoder.model": self.encoder,
            "classifier1.model": self.classifier1,
            "classifier2.model": self.classifier2,
            "attention.model": self.attention
        })

        return cls_acc, ins_acc

    def labeling(self, epoch, cls_acc, ins_acc):
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
        count = 0

        # evaluate network
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
                pred_bag, _pred_att = self.attention(mid_feature)
                _pred_att = torch.unsqueeze(_pred_att.reshape(-1), dim=0)
                pred_att = torch.cat((1 - _pred_att, _pred_att), dim=0).T
                pred1_list = np.append(pred1_list, np.array(F.softmax(pred_cls.data.cpu(), dim=1)), axis=0)
                pred2_list = np.append(pred2_list, np.array(F.softmax(pred_att.data.cpu(), dim=1)), axis=0)
                gt_list = np.append(gt_list, np.array(instance_label), axis=0)
                idx_list = np.append(idx_list, np.array(index), axis=0)

                progress_report('labeling', step, start_time, self.args.batch_size, len_data)

        new_label, true_label = self.give_pseudo_label(
            epoch, pred1_list, pred2_list, gt_list, idx_list, cls_acc, ins_acc)
        self.plotter.record(epoch, "pseudo_label_number", len(new_label))
        pseudo_label_acc = np.mean(new_label == true_label)
        self.plotter.record(epoch, "pseudo_label_accuracy", pseudo_label_acc)

        return len(new_label)

    def give_pseudo_label(self, epoch, pred1, pred2, gt_list, idx_list, cls_acc, ins_acc):
        # if epoch == -1:
        #     rate = max(int(1 / 20.0 * len(pred1)), 500)
        # else:
        #     rate = min(max(int((epoch + 1) / 20.0 * len(pred1)), 500), len(pred1))
        rate = len(pred1)

        perm = np.random.permutation(len(pred1))
        pred1, pred2 = pred1[perm[:rate]], pred2[perm[:rate]]
        gt_list, idx_list = gt_list[perm[:rate]], idx_list[perm[:rate]]

        thre = 0.2
        # thre = min(0.6 + 0.01*epoch, 0.8)

        if self.args.labeling_method == "cls":
            pos_idx_cand = np.where(pred1[:, 1] >= thre)[0]
            neg_idx_cand = np.where(pred1[:, 0] >= thre)[0]
            pos_val_cand, neg_val_cand = pred1[pos_idx_cand][:, 1], pred1[neg_idx_cand][:, 0]

        elif self.args.labeling_method == "ins":
            pos_idx_cand = np.where(pred2[:, 1] >= thre)[0]
            neg_idx_cand = np.where(pred2[:, 0] >= thre)[0]
            pos_val_cand, neg_val_cand = pred2[pos_idx_cand][:, 1], pred2[neg_idx_cand][:, 0]

        else:
            # weighted_val_pos = pred1[:, 1] * cls_acc + pred2[:, 1] * ins_acc
            # weighted_val_neg = pred1[:, 0] * cls_acc + pred2[:, 0] * ins_acc
            # pos_idx_cand = np.where(weighted_val_pos >= thre)[0]
            # neg_idx_cand = np.where(weighted_val_neg >= thre)[0]
            # pos_val_cand, neg_val_cand = weighted_val_pos[pos_idx_cand], weighted_val_neg[neg_idx_cand]

            if self.args.labeling_method == "simplesum":
                weighted_val = pred1 + pred2
            else:
                weighted_val = pred1 * cls_acc + pred2 * ins_acc

            # print(-np.sort(-weighted_val[:, 1])[:20])
            # pos_idx_cand = np.where(weighted_val[:, 1] >= thre)[0]
            # neg_idx_cand = np.where(weighted_val[:, 0] >= 1 - thre)[0]

            max_idx = np.argmax(weighted_val, axis=1)
            if self.args.labeling_method in ["random", "full"]:
                pos_idx_cand = np.where(max_idx == 1)[0]
                neg_idx_cand = np.where(max_idx == 0)[0]
            else:
                pos_idx_cand = np.where((weighted_val[:, 1] >= thre) & (max_idx == 1))[0]
                neg_idx_cand = np.where((weighted_val[:, 0] >= thre) & (max_idx == 0))[0]
                # pos_idx_cand = np.where(weighted_val[:, 1] >= thre)[0]
                # neg_idx_cand = np.where(weighted_val[:, 0] >= thre)[0]

            pos_val_cand, neg_val_cand = weighted_val[:, 1][pos_idx_cand], weighted_val[:, 0][neg_idx_cand]

        # num = min(len(pos_idx_cand), len(neg_idx_cand))
        # num = rate // 4
        # num = min(int(rate * ((epoch+4)//4) / 20), rate // 4)
        # n_max, n_min = rate // 4, rate // 20
        if self.args.dataset_name == "pathology":
            n_max, n_min = rate // 4, rate // 20
        elif self.args.dataset_name == "digit":
            n_max, n_min = rate // 10, rate // 30
        else:
            # n_max, n_min = rate // 5, rate // 30
            n_max, n_min = rate // 10, rate // 30

        num = min(n_min + ((n_max - n_min) // 20) * epoch, n_max)
        num = min(num, len(pos_idx_cand), len(neg_idx_cand))
        if self.args.dataset_name == "pathology":
            num_pos, num_neg = num, num
        else:
            num_pos, num_neg = num, 3*num
        print(rate, n_max, n_min, num, num_pos, num_neg)

        if self.args.labeling_method == "random":
            pos_idx = np.random.permutation(pos_idx_cand)[:num_pos]
            neg_idx = np.random.permutation(neg_idx_cand)[:num_neg]
        elif self.args.labeling_method == "full":
            pos_idx, neg_idx = pos_idx_cand, neg_idx_cand
        else:
            _pos_idx, _neg_idx = np.argsort(-pos_val_cand)[:num_pos], np.argsort(-neg_val_cand)[:num_neg]
            pos_idx, neg_idx = pos_idx_cand[_pos_idx], neg_idx_cand[_neg_idx]

        if num > 0:
            print(f"dynamic thre pos: {-np.sort(-pos_val_cand)[num_pos - 1]} neg: {-np.sort(-neg_val_cand)[num_neg - 1]}")

        print(f"pos: {len(pos_idx)}, neg: {len(neg_idx)}")

        true_label = np.concatenate((gt_list[pos_idx], gt_list[neg_idx]))
        new_label = np.concatenate((np.ones(len(pos_idx)), np.zeros(len(neg_idx))))

        self.pseudo_label_dict.clear()
        for pi in pos_idx:
            self.pseudo_label_dict[idx_list[pi]] = 1
        for ni in neg_idx:
            self.pseudo_label_dict[idx_list[ni]] = 0

        return new_label, true_label
