import sys
from rich import print
import pytorch_lightning as pl
from argparse import ArgumentParser
from data_modules import get_data_module
from networks_modules import Conv2, Conv4, Conv6
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr",type=float,default=50, help="Learning Rate [default 50]")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum [default 0.9]")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size [default 256]")
    parser.add_argument("--strategy", type=str, default="ASLP", help="Which method to use [default ASLP]")
    parser.add_argument("--weight-rescale",action="store_true", help="Whether to use smart rescale or not [default False]")
    parser.add_argument("--signed-constant", action="store_true", help="Whether to use Signed Constant (SC) or not [default False]")
    parser.add_argument("--network", type=str, default="Conv4", help="Select network architecture (Conv2, 4 or 6) [default Conv4]")
    parser.add_argument("--data-path", type=str, default=".", help="Specify where to download dataset [default '.' (current directory)]")
    parser.add_argument("--name", type=str, default="Experiment", help="Name of the experiment for tensorboard logger [default Experiment]")
    parser.add_argument("--prune-and-test", action="store_true", help="Whether to prune and evaluate network on the test dataset [default False]")
    parser.add_argument("--no-data-augment", action="store_true", help="Whether to enable or disable data augmentation [default False]")

    args = parser.parse_args()

    config = {
    "dataset": "cifar10",                       # Use CIFAR10 dataset
    "val_split":0.1,                            # Use 10% of images from train split as a validation split
    "max_epochs": 1000,                         # Train during 1000 epochs maximum
    "target_sparsity":0.5,                      # Only applicable if strategy is edge-popup
    "init_scheme": "kaiming_uniform_fan_in",    # Which weight initialization (Pytorch Standard Init)   
    "init_value":0,                             # Initial value for the masks
    "sp_lr":1e-3,                               # Learning rate for the scaling parameters (Smart Rescale)
    }

    # Update config with values form the CLI arguments
    config.update(vars(args))

    # Print config dict
    print(config)

    # Get DataModule with our parameters
    dm = get_data_module(config)

    if config["network"] not in ["Conv2", "Conv4", "Conv6"]:
        print("##### The network {} is not defined".format(config["network"]),file=sys.stderr)
        sys.exit()
    else:
        Model = eval(config["network"])

    # Create model
    net = Model(config)
    print(net)

    # Create logger
    logger = TensorBoardLogger("tb_logs", name=config["name"])
    # Create Early Stopping callback
    early_stopper = EarlyStopping(
                        monitor="val_acc",
                        patience=100,
                        mode="max",
                        )
    # Create Model Checkpoint Callback
    mdl_chkpt = ModelCheckpoint(
                        monitor="val_acc",
                        mode="max"
                        )
    
    # Create trainer object
    trainer = pl.Trainer(accelerator="gpu", 
                         devices=1, 
                         max_epochs=config["max_epochs"],
                         callbacks=[early_stopper,mdl_chkpt,RichProgressBar()])
                         
    # Train model
    trainer.fit(net, dm)

    # Test best model
    results = trainer.test(net, dm, ckpt_path=mdl_chkpt.best_model_path)

    print("Best accuracy: {:.2f}%".format(results[0]["test_acc"]*100))
    print("Best model path = " + mdl_chkpt.best_model_path)

    if config["prune_and_test"]:
        net = Model.load_from_checkpoint(mdl_chkpt.best_model_path)

        if config["strategy"] == "supermask":
            # With this strategy, no pruning but 10 sampled topologies which accuracies are averaged
            accuracies = []
            for _ in range(10):
                test_acc = trainer.test(net,dm)[0]["test_acc"]
                accuracies.append(test_acc)
            final_test_acc = sum(accuracies)/10
        elif config["strategy"] == "edge-popup":
            # With this strategy, no pruning since it is enforced during the forward pass
            final_test_acc = trainer.test(net,dm)[0]["test_acc"]
        elif config["strategy"] == "ASLP":
            net.strategy.prune_net(net)
            final_test_acc = trainer.test(net,dm)[0]["test_acc"]

        print("Pruned model test accuracy: {:.2f}%".format(final_test_acc*100))