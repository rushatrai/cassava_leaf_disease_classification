import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from ray import tune
from data_module import DataModule
from vgg16 import VGG16
from resnet50 import ResNet50
from population_based_training import run_pbt


def main():
    # available compute resources
    resources = {
        'cpu': 4,
        'gpu': 1,
    }

    # configurable hyperparameters
    config = {
        'img_dims': (256, 256),
        'batch_size': 32,
        'lr': 1e-4,
        'wd': 1e-2,
    }

    model_class = ResNet50
    model_name = str(model_class.__name__)
    dm_class = DataModule

    search_mode = False
    if not search_mode:
        # viz logger
        logger = TensorBoardLogger(
            save_dir=f'{os.getcwd()}/runs', name=model_name)

        # early stopping callback
        # early_stop_callback = EarlyStopping(
        #     monitor='val_acc_epoch', mode='max', patience=5)

        # learning rate monitor callback
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # trainer object
        trainer = pl.Trainer(
            gpus=resources['gpu'],
            logger=logger,
            max_epochs=30,
            # callbacks=[early_stop_callback],
            callbacks = [lr_monitor],
            precision=16,
        )

        # fits model
        data_module = dm_class(config)
        model = model_class(config)
        trainer.fit(model, datamodule=data_module)

        # saves model + version number
        if not os.path.exists(f'{os.getcwd()}/saved_models'):
            os.makedirs('saved_models')

        ver_num = max([run_dir for run_dir in os.listdir(f'{os.getcwd()}/runs/{model_name}')])
        torch.save(model.state_dict(),
                   f'{os.getcwd()}/saved_models/{model_name}_{ver_num}.pt')

    else:
        # pbt config
        pbt_config = {
            # search settings
            'num_epochs': 15,
            'num_samples': 8,
            'resources': resources,
            'mutations': {
                'batch_size': [16, 32, 64],
                'lr': tune.loguniform(1e-4, 1e-1),
                'wd': tune.loguniform(1e-4, 1e-1),
            },
            'metrics_to_report': {'loss': 'val_loss_epoch', 'accuracy': 'val_acc_epoch'},

            # cli reporter settings
            'parameter_columns': ['batch_size', 'lr', 'wd'],
            'metric_columns': ['loss', 'accuracy', 'training_iteration'],

            # run settings
            'metric_to_optimize': 'accuracy',
            'metric_optimize_mode': 'max',
        }

        # runs pbt
        search_results = run_pbt(
            model_name, config, model_class, dm_class, pbt_config)
        print('Best hyperparameters found were: ', search_results)


if __name__ == "__main__":
    main()
