import os
import torch
import argparse
import logging

from dataset_digit import DigitDataFactory
from core import AdaptPMCDDA, PretrainParallel
from models import EncoderDigit, Classifier, Attention
from utils import init_model, create_output_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    # Basic settings
    parser.add_argument('--log_dir', default='result_logs')
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    # Dataset settings
    parser.add_argument("--dataset_name", type=str, default='digit')
    parser.add_argument("--source_dataset", type=str, default='mnist')
    parser.add_argument("--target_dataset", type=str, default='svhn')
    parser.add_argument("--positive_class", type=int, default=9)
    parser.add_argument("--negative_class", type=int, default=-1)
    parser.add_argument("--train_length", type=int, default=2000)
    parser.add_argument("--test_length", type=int, default=500)
    parser.add_argument("--bag_size_mean", type=float, default=10)
    parser.add_argument("--bag_size_var", type=float, default=2)
    parser.add_argument("--valid_rate", type=int, default=0.2)
    # Training settings
    parser.add_argument("--num_epochs_pre", type=int, default=50)
    parser.add_argument("--num_epochs_adapt", type=int, default=100)
    parser.add_argument("--lr_pretrain", type=float, default=1.0E-4)
    parser.add_argument("--lr_adapt", type=float, default=1.0E-4)
    parser.add_argument('--weight_decay', type=float, default=1.0E-4)
    # Model settings
    parser.add_argument("--restore_path", type=str, default='')
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument('--feat_dim', type=int, default=500)
    # Labeling settings
    parser.add_argument('--labeling_thre', type=float, default=0.5)
    options, _ = parser.parse_known_args()
    return options


def main():
    args = parse_args()
    out_dir = create_output_dir(args)

    torch.manual_seed(args.random_seed)

    if torch.cuda.is_available() and args.gpu >= 0:
        args.device = torch.device(args.gpu)
    else:
        args.device = torch.device('cpu')

    logger.info(f"=== Loading datasets {args.dataset_name} ===")
    data_factory = DigitDataFactory(
        args.positive_class, args.negative_class, args.bag_size_mean, args.bag_size_var,
        args.valid_rate, args.random_seed)

    dl_train_s, dl_valid_s = data_factory.get_train_data_loader(
        args.source_dataset, args.train_length)
    dl_test_s = data_factory.get_test_data_loader(
        args.source_dataset, args.test_length)
    dl_train_t, dl_valid_t = data_factory.get_train_data_loader(
        args.target_dataset, args.train_length)
    dl_test_t = data_factory.get_test_data_loader(
        args.target_dataset, args.test_length)

    data_loaders = {
        "source_train": dl_train_s,
        "source_valid": dl_valid_s,
        "source_test": dl_test_s,
        "target_train": dl_train_t,
        "target_valid": dl_valid_t,
        "target_test": dl_test_t,
    }

    logger.info("=== Loading models ===")
    encoder = init_model(
        net=EncoderDigit(args.feat_dim), device=args.device,
        restore=os.path.join(args.restore_path, 'encoder.model'))
    classifier_pre = init_model(
        net=Classifier(args.feat_dim, args.num_class), device=args.device,
        restore=os.path.join(args.restore_path, 'classifier.model'))
    classifier1 = init_model(
        net=Classifier(args.feat_dim, args.num_class), device=args.device,
        restore=os.path.join(args.restore_path, 'classifier.model'))
    classifier2 = init_model(
        net=Classifier(args.feat_dim, args.num_class), device=args.device,
        restore=os.path.join(args.restore_path, 'classifier.model'))
    attention = init_model(
        net=Attention(args.feat_dim, args.num_class), device=args.device,
        restore=os.path.join(args.restore_path, 'attention.model'))

    use_pretrained = (encoder.restored and classifier_pre.restored and attention.restored)
    if not use_pretrained:
        logger.info("=== Pre-training classifier ===")
        pretrain = PretrainParallel(
            out_dir, encoder, classifier_pre, attention, data_loaders, args)
        pretrain.train()
        pretrain.plotter.refresh()

        logger.info("=== Pre-training evaluation ===")
        pretrain.predict('test', data_loaders["source_test"], split_type='test', data_category='source')
        pretrain.predict('test', data_loaders["target_test"], split_type='test', data_category='target')
        pretrain.plotter.flush('test', plot_flag=False)

        encoder = pretrain.encoder
        attention = pretrain.attention
        classifier1 = pretrain.classifier
        classifier2 = pretrain.classifier

    logger.info("=== Adaptation Training ===")
    adapt = AdaptPMCDDA(
        out_dir, encoder, classifier1, classifier2, attention, data_loaders, args)

    adapt.train()
    adapt.plotter.refresh()

    logger.info("=== Adaptation evaluation ===")
    adapt.predict('test', data_loaders["source_test"], split_type='test', data_category='source')
    adapt.predict('test', data_loaders["target_test"], split_type='test', data_category='target')
    adapt.plotter.flush('test', plot_flag=False)


if __name__ == '__main__':
    main()

