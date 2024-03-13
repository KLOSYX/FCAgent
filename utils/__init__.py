import hashlib
import importlib
from io import BytesIO

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


def list_to_markdown(lst):
    markdown = ""
    for item in lst:
        markdown += f"- {item}\n"
    return markdown


def generate_filename_from_image(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    hasher = hashlib.md5()
    hasher.update(img_byte_arr)
    hash_value = hasher.hexdigest()[:8]
    return str(hash_value)
