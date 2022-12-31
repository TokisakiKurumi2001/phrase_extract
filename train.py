from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from PhrExt import PhrExtDataLoader, LitPhrExt

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_phrase_extraction")

    hyperparameter = {
        'pretrained_ck': 'roberta-base',
        'method_for_layers': 'mean',
        'layers_use_from_last': 2,
        'lr': 3e-5
    }
    lit_phrext = LitPhrExt(**hyperparameter)

    # dataloader
    phrext_dataloader = PhrExtDataLoader(pretrained_ck='roberta-base', max_length=180)
    [train_dataloader, test_dataloader, valid_dataloader] = phrext_dataloader.get_dataloader(batch_size=64, types=["train", "test", "validation"])

    # train model
    trainer = pl.Trainer(max_epochs=25, devices=[0], accelerator="gpu", logger=wandb_logger)#, strategy="ddp")
    trainer.fit(model=lit_phrext, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(dataloaders=test_dataloader)

    # save model & tokenizer
    lit_phrext.export_model('phrext_model/v2')
