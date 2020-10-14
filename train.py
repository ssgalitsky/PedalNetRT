import pytorch_lightning as pl
import argparse
from model import PedalNet

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor='val_loss',
    mode='min',
    period=100
)

def main(args):
    model = PedalNet(args)
    trainer = pl.Trainer(
        # max_epochs=args.max_epochs, gpus=args.gpus, row_log_interval=100
        # The following line is for use with the Colab notebook when training on TPUs.
        # Comment out the above line and uncomment the below line to use.
        max_epochs=args.max_epochs, tpu_cores=args.tpu_cores, row_log_interval=100, checkpoint_callback=ModelCheckpoint()
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels", type=int, default=5)
    parser.add_argument("--dilation_depth", type=int, default=10)
    parser.add_argument("--num_repeat", type=int, default=1)
    
    # filter_width=kernel_size
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tpu_cores", default="8")

    parser.add_argument("--data", default="data.pickle")
    args = parser.parse_args()
    main(args)
