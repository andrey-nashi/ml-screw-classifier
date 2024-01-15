import copy
import albumentations as alb


class TransformationFactory:

    # ---------------------------------
    _TEMPLATE_S = \
    {
        "__version__": "1.3.0",
        "transform":
        {
            "__class_fullname__": "Compose",
            "p": 1.0,
            "transforms": []
        }
    }
    # ---------------------------------

    TR_NAME = "__class_fullname__"
    TR_P = "p"
    TR_ALWAYS_APPLY = "always_apply"

    CFG_TRANSFORM_NAME = "name"
    CFG_TRANSFORM_ARGS = "args"

    @staticmethod
    def create_transform(transformation_list: list):
        if len(transformation_list) == 0: return None

        serialized_cfg = copy.deepcopy(TransformationFactory._TEMPLATE_S)
        tl = serialized_cfg["transform"]["transforms"]

        for transform_cfg in transformation_list:
            transform_name = transform_cfg[TransformationFactory.CFG_TRANSFORM_NAME]
            transform_args = transform_cfg[TransformationFactory.CFG_TRANSFORM_ARGS]

            tr = copy.deepcopy(transform_args)
            tr[TransformationFactory.TR_NAME] = transform_name
            if TransformationFactory.TR_P not in tr:
                tr[TransformationFactory.TR_P] = 0.5
            if TransformationFactory.TR_ALWAYS_APPLY not in tr:
                tr[TransformationFactory.TR_ALWAYS_APPLY] = False

            tl.append(tr)

        transform_obj = alb.core.serialization.from_dict(serialized_cfg)
        return transform_obj

