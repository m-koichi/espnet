#!/usr/bin/env python3

from egs2.dcase20t4.sed1.pyscripts.sed_main import SEDTask


def get_parser():
    parser = SEDTask.get_parser()
    return parser


def main(cmd=None):
    """SED training
    Example:
        % python sed_train.py sed --print_config --optim adadelta
        % python sed_train.py --config conf/train_sed.yaml
    """
    SEDTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
