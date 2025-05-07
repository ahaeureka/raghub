from typing import Union

import filetype
import numpy as np
from PIL import Image


class FileHelper:
    """
    A class containing utility methods for file operations.
    """

    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Get the file extension from a file path.

        :param file_path: The path of the file.
        :return: The file extension.
        """
        return file_path.split(".")[-1] if "." in file_path else ""

    @staticmethod
    def get_file_name(file_path: str) -> str:
        """
        Get the file name without extension from a file path.

        :param file_path: The path of the file.
        :return: The file name without extension.
        """
        return file_path.split("/")[-1].split(".")[0] if "/" in file_path else ""

    @staticmethod
    def get_file_name_with_suffix(file_path: str) -> str:
        """
        Get the file name with suffix from a file path.

        :param file_path: The path of the file.
        :return: The file name with suffix.
        """
        return file_path.split("/")[-1] if "/" in file_path else ""

    @staticmethod
    def get_file_name_without_suffix(file_path: str) -> str:
        """
        Get the file name without suffix from a file path.

        :param file_path: The path of the file.
        :return: The file name without suffix.
        """
        return file_path.split("/")[-1].split(".")[0] if "/" in file_path else ""

    @staticmethod
    def get_file_name_without_suffix_and_dir(file_path: str) -> str:
        """
        Get the file name without suffix and directory from a file path.

        :param file_path: The path of the file.
        :return: The file name without suffix and directory.
        """
        return file_path.split("/")[-1].split(".")[0] if "/" in file_path else ""

    @staticmethod
    def get_mime_type(file_path: str) -> str:
        """
        Get the MIME type from a file path.

        :param file_path: The path of the file.
        :return: The MIME type.
        """
        mime_type = filetype.guess(file_path)
        if mime_type is None:
            return "application/octet-stream"  # Default MIME type for unknown files
        return mime_type.mime
        # with open(file_path, "rb") as file:
        #     file_content = file.read(2048)
        #     return magic.from_buffer(file_content, mime=True)

    @staticmethod
    def is_pdf(file_path: str) -> bool:
        """
        Check if the file is a PDF.

        :param file_path: The path of the file.
        :return: True if the file is a PDF, False otherwise.
        """
        mime_type = FileHelper.get_mime_type(file_path)
        return mime_type == "application/pdf"

    @staticmethod
    def is_image(file_path: str) -> bool:
        """
        Check if the file is an image.

        :param file_path: The path of the file.
        :return: True if the file is an image, False otherwise.
        """
        mime_type = FileHelper.get_mime_type(file_path)
        return mime_type.startswith("image/")

    @staticmethod
    def get_width_height(file_path: str) -> tuple:
        """
        Get the width and height of an image file.

        :param file_path: The path of the image file.
        :return: A tuple containing the width and height of the image.
        """
        if FileHelper.is_image(file_path):
            with Image.open(file_path) as img:
                return img.size
        elif FileHelper.is_pdf(file_path):
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            page = reader.pages[0]
            return page.mediabox.width, page.mediabox.height
        else:
            raise ValueError("Unsupported file type for width and height extraction.")

    @staticmethod
    def get_image(src: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        Convert the input to a format suitable for OCR processing.

        This method can be overridden by subclasses to customize the image conversion process.

        Args:
            src: The input image, which can be a file path, PIL Image, or OpenCV image.

        Returns:
            np.ndarray: The converted image suitable for OCR processing.
        """
        if isinstance(src, str):
            return Image.open(src)
        elif isinstance(src, Image.Image):
            return src
        elif isinstance(src, np.ndarray):
            return Image.fromarray(src)
        raise ValueError(f"Unsupport src type of {type(src)}")

    @staticmethod
    def get_image_np(src: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Convert the input to a NumPy array suitable for OCR processing.
        This method can be overridden by subclasses to customize the image conversion process.
        Args:
            src: The input image, which can be a file path, PIL Image, or OpenCV image.
        Returns:
            np.ndarray: The converted image suitable for OCR processing.
        """
        if isinstance(src, str):
            return np.array(Image.open(src))
        elif isinstance(src, Image.Image):
            return np.array(src)
        elif isinstance(src, np.ndarray):
            return src
        raise ValueError(f"Unsupported src type of {type(src)}")
