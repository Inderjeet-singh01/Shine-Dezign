import os
import shutil
import uuid

from fastapi import UploadFile

from app.core.config import settings

from app.core.logger import logger



def save_upload_file(file: UploadFile) -> str:
    logger.info(f"📤 Uploading file: {file.filename}")
    if not os.path.exists(settings.UPLOAD_DIR):
        os.makedirs(settings.UPLOAD_DIR)

    original_name = file.filename or "file"
    extension = ""

    if "." in original_name:
        extension = "." + original_name.split(".")[-1]

    unique_filename = f"{uuid.uuid4().hex}{extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"📤 File uploaded successfully: {unique_filename}")
    return unique_filename

