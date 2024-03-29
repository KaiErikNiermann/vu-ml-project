class config:
    """
    ### configuration 
    This is the main configuration for our testing, its based in the recommended configuration from the kaggle 
    - source : https://www.kaggle.com/code/awsaf49/planttraits2024-kerascv-starter-notebook
    """
    verbose = 1                 # Verbosity
    seed = 42                   # Random seed
    image_size = [224, 224]     # Input image size
    epochs = 12                 # Training epochs
    batch_size = 48             # Batch size
    lr_mode = "step"            # LR scheduler mode from one of "cos", "step", "exp"
    num_classes = 6             # Number of classes in the dataset
    num_folds = 5               # Number of folds to split the dataset
    fold = 0                    # Which fold to set as validation data
    preset = [
        "efficientnetv2_b2_imagenet", 
        "efficientnetv2_s_imagenet",
        "resnet50_imagenet",
    ]

    class_names = [
        "X4_mean",
        "X11_mean",
        "X18_mean",
        "X26_mean",
        "X50_mean",
        "X3112_mean",
    ]
    
    aux_class_names = list(map(lambda x: x.replace("mean", "sd"), class_names))
    num_classes = len(class_names)
    aux_num_classes = len(aux_class_names)
    
    
