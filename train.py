from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from PhrExt import PhrExtDataLoader, LitPhrExt

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_dummy")

    hyperparameter = {
        'pretrained_ck': 'roberta-base',
        'method_for_layers': 'sum',
        'layers_use_from_last': 4
    }
    lit_phrext = LitPhrExt(**hyperparameter)

    # dataloader
    phrext_dataloader = PhrExtDataLoader(pretrained_ck='roberta-base', max_length=25)
    [train_dataloader, test_dataloader, valid_dataloader] = phrext_dataloader.get_dataloader(batch_size=128, types=["train", "test", "validation"])

    # train model
    trainer = pl.Trainer(max_epochs=2, devices=[0], accelerator="gpu", logger=wandb_logger)#, strategy="ddp")
    trainer.fit(model=lit_phrext, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(dataloaders=test_dataloader)

    # save model & tokenizer
    # lit_phrext.export_model('phrext/v1')
