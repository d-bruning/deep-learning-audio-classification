from models.cnn import (
    build_cnn_model
)

from models.mobilenetv2 import (
    build_mobilenetv2_model
)

from models.crnn import (
    build_crnn_model
)

#from models.ast import (
#    build_ast_model
#)

from models.vit import (
    build_vit_model
)

MODEL_BUILDERS = {

    "cnn":
        build_cnn_model,

    "mobilenetv2":
        build_mobilenetv2_model,

    "crnn":
        build_crnn_model,

    #"ast":
    #    build_ast_model,

    "vit":
        build_vit_model
}


def build_model(
    model_name,
    input_shape,
    num_classes,
    learning_rate
):

    model_name = (
        model_name
        .lower()
        .strip()
    )

    if model_name not in MODEL_BUILDERS:

        available_models = (
            ", ".join(
                MODEL_BUILDERS.keys()
            )
        )

        raise ValueError(
            f"Unknown model: "
            f"{model_name}. "
            f"Available models: "
            f"{available_models}"
        )

    return MODEL_BUILDERS[
        model_name
    ](
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate
    )