import pytorch_lightning as pl
import torch
import torchvision
import wandb
import timm

import src.adversarial
import src.baseline

# MODEL_PATH = "/home/pedro/school/disertation/adversarial/pretrained_cifar/c3t1ovq6_resnet18_epoch=39_val_accuracy_top1=0.95.ckpt"
# DATASETS_ROOT = "/raid/data/pedro/datasets"
DATASETS_ROOT = "/home/pedro/imagenet_val"
EPOCHS = 1
DEVICE = 2
IMAGENET_ITER_STEP = 1  # 100 original
BATCH_SIZE = 512 # different for efficientnet!

cfg = list(
    # [
    #     {
    #         "advnn": {
    #             "cls": src.baseline.Blur,
    #             "kwargs": {
    #                 "kernel_size": sigma * 4
    #                 - 1,  # in contradiction with paper, but kernel size must be odd
    #                 "sigma": sigma,
    #             },
    #         },
    #         "bboxnn": {
    #             "cls": timm.create_model,
    #             "kwargs": {"model_name": model, "pretrained": True},
    #         },
    #         "simnn": {"cls": src.adversarial.SimulationNN, "kwargs": {}},
    #     }
    #     for sigma in range(2, 10)
    # ]
    # + [
    # [
    #     {
    #         "advnn": {
    #             "cls": src.baseline.Noise,
    #             # "kwargs": {"sigma": sigma.round(decimals=2).item()},
    #             "kwargs": {"sigma": sigma},
    #         },
    #         "bboxnn": {
    #             "cls": timm.create_model,
    #             "kwargs": {'model_name': model, 'pretrained': True},
    #         },
    #         "simnn": {"cls": src.adversarial.SimulationNN, "kwargs": {}},
    #     }
    #     # for sigma in range(0, 100, 10)
    #     # for sigma in torch.arange(0, 110 / 255, 10 / 255)
    #     for sigma in [80]
    # ]
    # + [
    # # [
    #     {
    #         "advnn": {
    #             "cls": src.baseline.Contrast,
    #             "kwargs": {"factor": factor.round(decimals=2).item()},
    #         },
    #         "bboxnn": {
    #             "cls": timm.create_model,
    #             "kwargs": {'model_name': model, 'pretrained': True},
    #         },
    #         "simnn": {"cls": src.adversarial.SimulationNN, "kwargs": {}},
    #     }
    #     # for factor in range(0,1.1,0.1)
    #     for factor in torch.arange(0.1, 2.1, 0.1)
    # ]
    # +[
    # [
    #     {
    #         "advnn": {
    #             "cls": src.baseline.JPEG,
    #             "kwargs": {"q": q},
    #         },
    #         "bboxnn": {
    #             "cls": timm.create_model,
    #             "kwargs": {'model_name': model, 'pretrained': True},
    #         },
    #         "simnn": {"cls": src.adversarial.SimulationNN, "kwargs": {}},
    #     }
    #     # for factor in range(0,1.1,0.1)
    #     for q in list(range(2, 22, 2)) + [100]
    # ]
    # + [
    # # # [
    #     {
    #         "advnn": {
    #             "cls": src.baseline.Rescale,
    #             "kwargs": {"factor": factor},
    #         },
    #         "bboxnn": {
    #             "cls": timm.create_model,
    #             "kwargs": {'model_name': model, 'pretrained': True},
    #         },
    #         "simnn": {"cls": src.adversarial.SimulationNN, "kwargs": {}},
    #     }
    #     # for factor in range(0,1.1,0.1)
    #     for factor in [1/100, 5/100, 1/10, 1/5, 3/10, 1/2, 4/5, 1]
    # ]
    # + [
    [
        {
            "advnn": {
                "cls": src.baseline.Grid,
                "kwargs": {"factor": factor, "gridsize": gridsize},
            },
            "bboxnn": {
                "cls": timm.create_model,
                "kwargs": {'model_name': model, 'pretrained': True},
            },
            "simnn": {"cls": src.adversarial.SimulationNN, "kwargs": {}},
        }
        # for factor in range(0,1.1,0.1)
        # for gridsize, factor in [(2, x) for x in torch.arange(0.5, 1.5, 0.2)] # Lastest
        for gridsize, factor in [(2, x) for x in [1.0, 1.5]] # Lastest
        # + [(4, x) for x in torch.arange(0.5, 1.5, 0.2)]
        # + [(8, x) for x in torch.arange(0.5, 1.5, 0.2)]
        # + [(16, x) for x in torch.arange(0.5, 1.5, 0.2)]
    ]
    for model in [
        # "resnet50.a1_in1k",
        # "resnet50.fb_ssl_yfcc100m_ft_in1k",
        # "resnet50.fb_swsl_ig1b_ft_in1k",
        # # compare with timm's effnet
        # 'efficientnet_b0',
        # 'efficientnet_b1',
        # 'efficientnet_b2',
        # 'efficientnet_b3',
        # 'efficientnet_b4',
         #standard experiment mode,
        'resnet18d',
        'resnet26d',
        'resnet34d',
        'resnet101d',
        'resnet152d',
        'resnet200d',
        # 'vit_huge_patch14_224_in21k', # NO PRETRAINED WEIGH',
        'vit_large_patch16_224',
        'vit_base_patch16_224',
        'vit_small_patch16_224',
        'vit_tiny_patch16_224',
        'tf_efficientnet_b0',
        'tf_efficientnet_b1',
        'tf_efficientnet_b2',
        'tf_efficientnet_b3',
        'tf_efficientnet_b4',
        'tf_efficientnet_b5',
        'tf_efficientnet_b6',
        'tf_efficientnet_b7',
        'swin_tiny_patch4_window7_224',
        'swin_small_patch4_window7_224',
        'swin_base_patch4_window7_224',
        'swin_large_patch4_window7_224',
        'convnext_atto',
        'convnext_base',
        'convnext_femto',
        'convnext_large',
        'convnext_nano',
        'convnext_pico',
        'convnext_small',
        'convnext_tiny',
        'convnext_xlarge',
        'mobilenetv3_small_050',
        'mobilenetv3_small_075',
        'mobilenetv3_small_100',
    ]
)


def kwargs2str(kwargs):
    return f"{','.join([f'{k}={v}' for k,v in kwargs.items()])}"


def main():
    m = timm.create_model('resnet18', pretrained=True)
    # format check
    architectures = ["resnet", "efficientnet", "convnext", "mobilenetv3", "swin", "vit"]
    all_architectures_known = []
    for x in cfg:
        for c in x:
            all_architectures_known.append(
                any([a in c["bboxnn"]["kwargs"]["model_name"] for a in architectures])
            )
    assert all(all_architectures_known)

    # computation
    for x in cfg:
        for c in x:
            model_family = [
                x for x in architectures if x in c["bboxnn"]["kwargs"]["model_name"]
            ][0]
            name = "ball"
            advnn_kwargs = kwargs2str(c["advnn"]["kwargs"])
            bboxnn_kwargs = kwargs2str(c["bboxnn"]["kwargs"])
            name = f"{name}_advnn:{c['advnn']['cls'].__name__}({advnn_kwargs})_bboxnn:{c['bboxnn']['cls'].__name__}({bboxnn_kwargs})"
            wandb_exp = wandb.init(
                reinit=True,
                project="adversarial",
                entity="petr_petr",
                name=name,
                group="advgen_baseline2",
                tags=["whole_imagenet", model_family, c["advnn"]["cls"].__name__, "TODO"]
            )

            blackbox_classifier = c["bboxnn"]["cls"](**c["bboxnn"]["kwargs"])
            blackbox_classifier.eval()
            try:
                blackbox_input_size = blackbox_classifier.pretrained_cfg[
                    "test_input_size"
                ][-1]
            except KeyError:
                blackbox_input_size = blackbox_classifier.pretrained_cfg["input_size"][
                    -1
                ]
            adversarial_nn = c["advnn"]["cls"](**c["advnn"]["kwargs"])
            simulation_nn = c["simnn"]["cls"](**c["simnn"]["kwargs"])

            test_imagenet = torchvision.datasets.ImageNet(
                root="/raid/data/imagenet/2012/images",
                split="val",
                transform=src.adversarial.Transform(
                    resize_size=blackbox_input_size,
                    crop_size=blackbox_input_size,
                    interpolation=torchvision.transforms.functional.InterpolationMode(
                        blackbox_classifier.pretrained_cfg["interpolation"]
                    ),
                ),
            )
            imagenet_subset_indices = list(
                range(0, len(test_imagenet), IMAGENET_ITER_STEP)
            )
            imagenet_sampler = src.adversarial.ImagenetSampler(imagenet_subset_indices)
            eval_loader = torch.utils.data.DataLoader(
                test_imagenet,
                batch_size=BATCH_SIZE if not 'efficientnet' in name else 96,
                shuffle=False,
                num_workers=32,
                drop_last=True,
                sampler=imagenet_sampler,
                pin_memory=True,
            )

            wandb_logger = pl.loggers.WandbLogger(experiment=wandb_exp)
            trainer = pl.Trainer(
                max_epochs=EPOCHS,
                precision="16",
                accelerator="gpu",
                devices=[DEVICE],
                logger=wandb_logger,
                callbacks=[
                    pl.callbacks.LearningRateMonitor(logging_interval="step"),
                ],
            )

            trainable_params = sum(
                p.numel() for p in blackbox_classifier.parameters() if p.requires_grad
            )
            adversarial_transform = c["advnn"]["cls"].__name__
            wandb.log(
                {
                    "params": trainable_params,
                    "input_size": blackbox_input_size,
                    "base_architecture": model_family,
                    "adv_transform": adversarial_transform,
                    **c["advnn"]["kwargs"],
                    **c["bboxnn"]["kwargs"],
                    "imagenet_iter_step": IMAGENET_ITER_STEP,
                }
            )

            trainee = src.adversarial.Trainee(
                blackbox_classifier,
                adversarial_nn,
                simulation_nn,
                EPOCHS,
                mean=blackbox_classifier.pretrained_cfg["mean"],
                std=blackbox_classifier.pretrained_cfg["std"],
            )
            trainer.validate(trainee, eval_loader)


if __name__ == "__main__":
    main()
