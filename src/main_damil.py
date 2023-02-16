import os
import torch
from torch.optim.lr_scheduler import StepLR
import argparse

from dataset_digit import DigitDataLoader
from dataset_visda import VisdaDataLoader
from core import AdaptPMCDDA, PretrainParallel
from models import EncoderVisda, EncoderDigit, Classifier, Attention
from utils import init_model, create_output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument('--log_dir', default='result_logs')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--lr_pe", type=float, default=1.0E-5)
    parser.add_argument("--lr_pc", type=float, default=1.0E-5)
    parser.add_argument("--lr_pa", type=float, default=1.0E-5)
    parser.add_argument("--lr_e", type=float, default=1.0E-6)
    parser.add_argument("--lr_c", type=float, default=1.0E-6)
    parser.add_argument("--lr_a", type=float, default=1.0E-6)
    parser.add_argument("--lr_interval", type=int, default=100)
    parser.add_argument("--lr_dim", type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=10e-5)
    # parser.add_argument("--num_k", type=int, default=2)
    parser.add_argument("--mean", type=float, default=10)
    parser.add_argument("--var", type=float, default=2)
    # parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--num_epochs_pre", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument('--feat_dim', type=int, default=500)
    parser.add_argument("--train_length", type=int, default=500)
    parser.add_argument("--test_length", type=int, default=200)
    parser.add_argument("--positive_class", type=int, default=-1)
    parser.add_argument("--negative_class", type=int, default=-1)
    parser.add_argument("--wda", type=float, default=1.0)
    parser.add_argument("--wmil", type=float, default=1.0)
    parser.add_argument('--att_func', default='sigmoid', choices=['softmax', 'sigmoid'])
    parser.add_argument('--load_method', default='random', choices=['random', 'slide', 'block', 'slide_random', 'cluster'])
    parser.add_argument('--labeling_method', default='both', choices=['both', 'cls', 'ins', 'simplesum', 'random', 'full', 'feature'])
    parser.add_argument("--dataset_name", type=str, default='visda')
    parser.add_argument("--source_dataset", type=str, default='train')
    parser.add_argument("--target_dataset", type=str, default='validation')
    parser.add_argument("--mode", type=str, default="full", choices=["mil", "mida", "full", "womcd", "wolabel"])
    parser.add_argument("--restore_path", type=str, default='')
    options, _ = parser.parse_known_args()
    return options


def main():
    args = parse_args()
    out_dir = create_output_dir(args)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.gpu >= 0:
        args.device = torch.device(args.gpu)
    else:
        args.device = torch.device('cpu')

    print(f"=== Loading datasets {args.dataset_name} ===")
    dataloader_class = {
        "visda": VisdaDataLoader,
        "digit": DigitDataLoader,
    }[args.dataset_name]
    dataloader = dataloader_class(
        args.positive_class, args.negative_class, args.batch_size,
        args.mean, args.var, args.num_class, args.seed)

    source_data, source_label = dataloader.get_data_bags(
        args.source_dataset, args.train_length)
    source_data_test, source_label_test = dataloader.get_data_bags(
        args.source_dataset, args.test_length, train=False)
    target_data, target_label = dataloader.get_data_bags(
        args.target_dataset, args.train_length)
    target_data_test, target_label_test = dataloader.get_data_bags(
        args.target_dataset, args.test_length, train=False)

    pretrain_data_loaders = {
        "source_train": dataloader.data_to_dataloader(
            source_data, source_label),
        "source_test": dataloader.data_to_dataloader(
            source_data_test, source_label_test, train=False),
        "target_train": dataloader.data_to_dataloader(
            target_data, target_label),
        "target_test": dataloader.data_to_dataloader(
            target_data_test, target_label_test, train=False),
    }

    # load models
    print("=== Loading models ===")
    encoder_model = {
        "visda": EncoderVisda,
        "digit": EncoderDigit
    }[args.dataset_name]
    encoder = init_model(
        net=encoder_model(args.feat_dim), device=args.device,
        restore=os.path.join(args.restore_path, 'pre_encoder.model'))
    classifier_pre = init_model(
        net=Classifier(args.feat_dim, args.num_class), device=args.device,
        restore=os.path.join(args.restore_path, 'pre_classifier.model'))
    classifier1 = init_model(
        net=Classifier(args.feat_dim, args.num_class), device=args.device,
        restore=None)
    classifier2 = init_model(
        net=Classifier(args.feat_dim, args.num_class), device=args.device,
        restore=None)
    attention = init_model(
        net=Attention(args.feat_dim, args.num_class, args.att_func), device=args.device,
        restore=os.path.join(args.restore_path, 'pre_attention.model'))

    # Pretrain model
    use_pretrained = (encoder.restored and classifier_pre.restored and attention.restored)
    if not use_pretrained:
        print("=== Pre-training classifier ===")
        pretrain = PretrainParallel(
            out_dir, encoder, classifier_pre, attention, pretrain_data_loaders, args)
        pretrain.train()
        pretrain.plotter.refresh()

        pretrain.test('test', pretrain_data_loaders["source_test"], data_category='source')
        pretrain.test('test', pretrain_data_loaders["target_test"], data_category='target')
        pretrain.plotter.flush('test')

        encoder = pretrain.encoder
        attention = pretrain.attention
        classifier1 = pretrain.classifier  # not initialize classifier ???
        classifier2 = pretrain.classifier
    else:
        print("== Use existing model ==")
        # pretrain_parallel.test(0, source_data_loader_test, data_category='source')
        # pretrain_parallel.test(0, target_data_loader_test, data_category='target')
        # logger.flush()

    adapt_data_loaders = {
        "source_train": dataloader.data_to_dataloader(
            source_data, source_label, dataloader_type="index"),
        "source_test": dataloader.data_to_dataloader(
            source_data_test, source_label_test, train=False),
        "target_train": dataloader.data_to_dataloader(
            target_data, target_label, dataloader_type="index"),
        "target_test": dataloader.data_to_dataloader(
            target_data_test, target_label_test, train=False),
    }

    # domain adaptation
    print("=== Training with Domain Adaptaion ===")
    adapt = AdaptPMCDDA(out_dir, encoder, classifier1, classifier2, attention,
                        adapt_data_loaders, args)
    # labeling_num = adapt.labeling(0, cls_acc, ins_acc)
    # labeling_num = adapt.labeling_feature(0)
    # result_logger.flush()

    adapt.train()
    adapt.plotter.refresh()

    adapt.test('test', adapt_data_loaders["source_test"], data_category='source')
    adapt.test('test', adapt_data_loaders["target_test"], data_category='target')
    adapt.plotter.flush('test')


if __name__ == '__main__':
    main()

