import pytorch_lightning as pl
import torch
import torchmetrics
from stlFakeDataModule import STLFakeDataModule
from boringModel import BoringModel



class TestModel(BoringModel):
    layer = torch.nn.Linear(6, 1)
    metrics = torchmetrics.MetricCollection([torchmetrics.AUROC(), torchmetrics.Accuracy()])
    test_metrics = torchmetrics.MetricCollection({"AUROC/test":torchmetrics.AUROC(),"Accuracy/test":torchmetrics.Accuracy()})

    def loss(self, prediction, target):
        loss = torch.nn.functional.binary_cross_entropy(prediction, target)
        return loss

    def training_step(self, batch, _):
        values, target = batch
        output = self.layer(values)
        output = torch.sigmoid(output)
        loss = self.loss(output, target)
        target = target.type(torch.int)
        self.metrics.update(output, target)
        return {"loss": loss}

    def validation_step(self, batch, _):
        values, target = batch
        output = self.layer(values)
        output = torch.sigmoid(output)
        loss = self.loss(output, target)
        return {'x': loss}

    def test_step(self, batch, _):
        values, target = batch
        output = self.layer(values)
        output = torch.sigmoid(output)
        loss = self.loss(output, target)
        target = target.type(torch.int)
        self.test_metrics.update(output, target)
        return {'y': loss}

    def training_epoch_end(self, outs):
        self.log_dict(self.metrics.compute())
    
    def test_epoch_end(self, outs):
        self.log_dict(self.test_metrics.compute())


data_module = STLFakeDataModule(batch_size=1000)
model = TestModel()

trainer = pl.Trainer(max_epochs=1000)
trainer.fit(model, data_module)
trainer.test(model, datamodule=data_module)
