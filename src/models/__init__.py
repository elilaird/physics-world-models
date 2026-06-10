from src.models.visual import VisualWorldModel
from src.models.rich_sid import RichSIDVisualWorldModel

MODEL_REGISTRY = {
    "visual_world_model": VisualWorldModel,
    "visual_world_model_rich_sid": RichSIDVisualWorldModel,
}
