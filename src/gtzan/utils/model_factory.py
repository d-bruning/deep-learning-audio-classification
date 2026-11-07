from models.cnn import build_cnn_model
from models.mlp import build_mlp_model
from models.resnet50 import build_resnet50_model
from models.efficientnet import build_efficientnet_model
from models.crnn_v4 import build_crnn_model
from models.mobilenetv2 import build_mobilenetv2_model
from models.residualcnn import build_residualcnn_model
from models.vgg import build_vgg_model
from models.shallowresnet import build_shallowresnet_model
from models.capsnet import build_capsnet_model

def build_model(
    model_name,
    input_shape,
    num_classes,
    learning_rate=1e-3
):

    model_name = model_name.lower()

    if model_name == "cnn":
        return build_cnn_model(
            input_shape,
            num_classes,
            learning_rate
        )

    if model_name == "mlp":
        return build_mlp_model(
            input_shape,
            num_classes,
            learning_rate
        )

    if model_name == "resnet50":
        return build_resnet50_model(
            input_shape,
            num_classes,
            learning_rate
        )

    if model_name == "efficientnet":
        return build_efficientnet_model(
            input_shape,
            num_classes,
            learning_rate
        )
        
    if model_name == "crnn":
        return build_crnn_model(
            input_shape,
            num_classes,
            learning_rate
        )
        
    if model_name == "mobilenetv2":
        return build_mobilenetv2_model(
            input_shape,
            num_classes,
            learning_rate
        )

    if model_name == "residualcnn":
        return build_residualcnn_model(
            input_shape,
            num_classes,
            learning_rate
        )
        
    if model_name == "vgg":
        return build_vgg_model(
            input_shape,
            num_classes,
            learning_rate
        )

    if model_name == "capsnet":
        return build_capsnet_model(
            input_shape,
            num_classes,
            learning_rate
        )

    if model_name == "shallowresnet":
        return build_shallowresnet_model(
            input_shape,
            num_classes,
            learning_rate
        ) 
        
    raise ValueError(
        f"Unknown model: {model_name}"
    )