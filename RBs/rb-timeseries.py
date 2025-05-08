import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import random
import os

# imports for training
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
# import dataset, network to train and metric to optimize
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, RMSE
from lightning.pytorch.tuner import Tuner

class Forecaster:

    def __init__(self, seed=42):
        self.model = None
        self.val_dataloader = None
        self.seed = seed

    def set_seed(self, seed=None):
        """Set seeds for reproducibility."""
        if seed is None:
            seed = self.seed
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set deterministic behavior with allowance for operations without deterministic implementation
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Allow operations that don't have deterministic implementations
        # This addresses the upsample_linear1d_backward_out_cuda error
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        pl.seed_everything(seed)

    def fit(self, data, group_size):
        # Set seeds for reproducibility
        self.set_seed()
        
        # define the dataset, i.e. add metadata to pandas dataframe for the model to understand it
        max_encoder_length = group_size-1
        max_prediction_length = 1

        training = TimeSeriesDataSet(
            data,
            time_idx= 'time_idx',  # column name of time of observation
            target= 'Next Week Fantasy Points',  # column name of target to predict
            group_ids=[ 'player_name' ],  # column name(s) for timeseries IDs
            max_encoder_length=max_encoder_length,  # how much history to use
            max_prediction_length=max_prediction_length,  # how far to predict into future
            # covariates static for a timeseries ID
            static_categoricals=[  ],
            static_reals=[  ],
            # covariates known and unknown in the future to inform prediction
            time_varying_known_categoricals=[ ],
            time_varying_known_reals=[ 'Rushing Yards', 'Receptions', 'Receiving Yards', 'Receiving TD',
                                        'Rushing TD', 'Fumbles Lost',],
            time_varying_unknown_categoricals=[  ],
            time_varying_unknown_reals=[ 'Next Week Fantasy Points' ],
        )

        # create validation dataset using the same normalization techniques as for the training dataset
        validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)

        # convert datasets to dataloaders for training
        batch_size = 64
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

        # create PyTorch Lighning Trainer with early stopping
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="auto",  # run on CPU, if on multiple GPUs, use strategy="ddp"
            callbacks=[lr_logger, early_stop_callback],
            logger=TensorBoardLogger("lightning_logs"),
            # Use deterministic algorithms where possible, but allow exceptions
            deterministic="warn"  # Changed from True to "warn"
        )

        # define network to train - the architecture is mostly inferred from the dataset, so that only a few hyperparameters have to be set by the user
        tft = TemporalFusionTransformer.from_dataset(
            # dataset
            training,
            # architecture hyperparameters
            hidden_size=32,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=16,
            # loss metric to optimize
            loss=RMSE(),
            # logging frequency
            log_interval=2,
            # optimizer parameters
            learning_rate=0.03,
            reduce_on_plateau_patience=4
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

        # find the optimal learning rate
        res = Tuner(trainer).lr_find(
            tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
        )
        # and plot the result - always visually confirm that the suggested learning rate makes sense
        print(f"suggested learning rate: {res.suggestion()}")
        # fig = res.plot(show=True, suggest=True)
        # fig.show()

        # fit the model on the data - redefine the model with the correct learning rate if necessary
        trainer.fit(
            tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
        )

        self.model = tft
        self.val_dataloader = val_dataloader

    def get_rmse(self):
        actuals = torch.cat([y[0] for x, y in iter(self.val_dataloader)])
        predictions = self.model.predict(self.val_dataloader, mode="prediction")

        # calculate the final RMSE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # send actuals to the same device as predictions
        actuals = actuals.to(device)

        # calculate RMSEs
        rmse = ((actuals - predictions) ** 2).mean().sqrt()

        return rmse

if __name__ == "__main__":
    sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    best_rmse = 1000
    best_size = 0

    for group_size in sizes:
        
        df = pd.read_csv('runningbacks.csv')

        players = df.groupby('Player')

        clean_players = pd.DataFrame()

        # loop through the players
        for name, group in players:
            # only keep players with more than 10 games
            if len(group) >= group_size:
                # loop through the player group and add 6 rows at a time per player
                for i in range(0, len(group)-group_size):
                    rows_to_add = group.iloc[i:i+group_size]
                    rows_to_add['player_name'] = name + '_' + str(i)
                    clean_players = pd.concat([clean_players, rows_to_add])

        # reset the index
        clean_players.reset_index(drop=True, inplace=True)
        # drop the original player column
        clean_players.drop(columns=['Player'], inplace=True)

        df = clean_players.copy()

        # add a time_idx to the dataframe by player
        df['time_idx'] = df.groupby('player_name').cumcount()

        df = df[['player_name', 'time_idx', 'Rushing Yards', 'Receptions', 'Receiving Yards', 'Receiving TD',
                'Rushing TD', 'Fumbles Lost', 'Next Week Fantasy Points']]

        model = Forecaster()

        model.fit(df, group_size)

        rmse = model.get_rmse()
        print(f'RMSE: {rmse}')
        
        if best_rmse > rmse:
            best_rmse = rmse
            best_size = group_size

    print(f'Best RMSE: {best_rmse} with group size: {best_size}')