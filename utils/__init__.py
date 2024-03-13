import importlib

from langchain.tools import BaseTool
from pyrootutils import setup_root

ROOT = setup_root(".")


def load_base_tools(directory: str) -> list[BaseTool]:
    """Load all BaseTool instances from directory."""
    all_instance = []
    path = ROOT / directory
    for filename in path.glob("*.py"):
        if filename.stem == "__init__":
            continue
        # 需要被导入的class必须与文件名存在对应关系：image_qa.py -> ImageQaTool(BaseTool)
        class_name = "".join([x.capitalize() for x in filename.stem.split("_")]) + "Tool"
        import_path = ".".join([directory, filename.parts[-1][:-3]])
        module = importlib.import_module(import_path)
        try:
            cls = getattr(module, class_name)
        except AttributeError:
            continue
        if issubclass(cls, BaseTool):
            all_instance.append(cls())
    return all_instance
