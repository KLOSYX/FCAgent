import asyncio
import functools
import hashlib
import importlib
from io import BytesIO

from langchain.tools import BaseTool
from loguru import logger
from pyrootutils import setup_root

ROOT = setup_root(".")


def load_base_tools(directory: str, except_classes: None | list[str] = None) -> list[BaseTool]:
    """Load all BaseTool instances from directory."""
    all_instance = []
    path = ROOT / directory
    for filename in path.glob("*.py"):
        if filename.stem == "__init__":
            continue
        # 需要被导入的class必须与文件名存在对应关系：image_qa.py -> ImageQaTool(BaseTool)
        class_name = "".join([x.capitalize() for x in filename.stem.split("_")]) + "Tool"
        if except_classes is not None and class_name in except_classes:
            continue
        import_path = ".".join([directory, filename.parts[-1][:-3]])
        module = importlib.import_module(import_path)
        try:
            cls = getattr(module, class_name)
        except AttributeError:
            continue
        if issubclass(cls, BaseTool):
            instance = cls()
            assert hasattr(instance, "cn_name"), f"{cls} must have a cn_name attribute"
            assert hasattr(instance, "is_multimodal"), f"{cls} must have a is_multimodal attribute"
            all_instance.append(instance)
    return sorted(all_instance, key=lambda x: x.cn_name)


def list_to_markdown(lst):
    markdown = ""
    for item in lst:
        markdown += f"- {item}\n"
    return markdown


def generate_filename_from_image(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    hasher = hashlib.sha256()
    hasher.update(img_byte_arr)
    hash_value = hasher.hexdigest()[:8]
    return str(hash_value)


def tool_exception_catch(tool_name: str = "tool"):
    """Decorator to catch exceptions in both async and sync tool functions."""
    err_message: str = f"Failed to invoke tool: `{tool_name}`."

    def decorator(func):
        # 检查被装饰的函数是否是异步函数
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper_async(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"Error in tool {tool_name}")
                    return err_message

            return wrapper_async
        else:

            @functools.wraps(func)
            def wrapper_sync(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"Error in tool {tool_name}")
                    return err_message

            return wrapper_sync

    return decorator
