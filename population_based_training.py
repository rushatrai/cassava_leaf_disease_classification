import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter


def run_pbt(model_name, search_space, network, dm, pbt_config):
    def train_tune(config, epochs, resources, checkpoint_dir=None):
        # viz logger
        logger = TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name=model_name)

        # metric reporter + checkpoint callback
        callback = TuneReportCheckpointCallback(
            metrics=pbt_config['metrics_to_report'])

        # search trainer object
        trainer = pl.Trainer(
            max_epochs=epochs,
            gpus=resources['gpu'],
            logger=logger,
            callbacks=[callback],
            progress_bar_refresh_rate=50,
            precision=16,
        )

        # checkpointing system
        if checkpoint_dir:
            model = network.load_from_checkpoint(
                os.path.join(checkpoint_dir, 'checkpoint'))
        else:
            model = network(config)

        # fits model/data module with current hyperparameter set
        data_module = dm(config)
        trainer.fit(model, datamodule=data_module)

    # search params
    trainable = tune.with_parameters(
        train_tune,
        epochs=pbt_config['num_epochs'],
        resources=pbt_config['resources'],
    )

    # pbt object
    scheduler = PopulationBasedTraining(
        hyperparam_mutations=pbt_config['mutations'],
    )

    # cli reporter
    reporter = CLIReporter(
        parameter_columns=pbt_config['parameter_columns'],
        metric_columns=pbt_config['metric_columns'],
    )

    # runs pbt
    analysis = tune.run(
        trainable,
        metric=pbt_config['metric_to_optimize'],
        mode=pbt_config['metric_optimize_mode'],
        config=search_space,
        resources_per_trial=pbt_config['resources'],
        num_samples=pbt_config['num_samples'],
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=1,
    )

    # returns results of search
    return analysis.best_config
