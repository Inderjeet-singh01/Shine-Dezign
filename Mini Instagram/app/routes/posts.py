import os

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.db.database import get_db
from app.dependencies.auth import get_current_user_from_session, get_current_user_from_token
from app.models.post import Post
from app.models.user import User
from app.services.file_service import save_upload_file

router = APIRouter(tags=["Posts"])


@router.get("/posts")
def get_posts(db: Session = Depends(get_db)):
    logger.info("[POSTS] Fetching all posts")
    posts = db.query(Post).all()
    result = []

    for post in posts:
        user = db.query(User).filter(User.id == post.user_id).first()
        if user:
            result.append(
                {
                    "id": post.id,
                    "image": post.image,
                    "content": post.content,
                    "username": user.username,
                    "user_id": user.id,
                }
            )

    logger.info(f"[POSTS] Returned {len(result)} posts")
    return result


@router.post("/posts")
def create_post(
    content: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_from_session),
    db: Session = Depends(get_db),
):
    logger.info(f"[POST CREATE][SESSION] Request received for user_id={current_user.id}")

    saved_filename = save_upload_file(file)
    logger.info(f"[POST CREATE][SESSION] File saved as {saved_filename}")

    new_post = Post(content=content, image=saved_filename, user_id=current_user.id)

    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    logger.info(f"[POST CREATE][SESSION] Success post_id={new_post.id} user_id={current_user.id}")
    return {"message": "Post created successfully"}


@router.post("/posts/token")
def create_post_with_token(
    content: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db),
):
    logger.info(f"[POST CREATE][TOKEN] Request received for user_id={current_user.id}")

    saved_filename = save_upload_file(file)
    logger.info(f"[POST CREATE][TOKEN] File saved as {saved_filename}")

    new_post = Post(content=content, image=saved_filename, user_id=current_user.id)

    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    logger.info(f"[POST CREATE][TOKEN] Success post_id={new_post.id} user_id={current_user.id}")
    return {"message": "Post created successfully", "user_id": current_user.id}


@router.delete("/posts/{post_id}")
def delete_post(
    post_id: int,
    current_user: User = Depends(get_current_user_from_session),
    db: Session = Depends(get_db),
):
    logger.info(f"[POST DELETE] Request received for post_id={post_id} by user_id={current_user.id}")

    post = db.query(Post).filter(Post.id == post_id).first()

    if not post:
        logger.warning(f"[POST DELETE] Failed because post_id={post_id} was not found")
        raise HTTPException(status_code=404, detail="Post not found")

    if post.user_id != current_user.id:
        logger.warning(
            f"[POST DELETE] Denied because user_id={current_user.id} does not own post_id={post_id}"
        )
        raise HTTPException(status_code=403, detail="You can delete only your own posts")

    image_path = os.path.join(settings.UPLOAD_DIR, post.image)
    if post.image and os.path.exists(image_path):
        os.remove(image_path)
        logger.info(f"[POST DELETE] Image removed for post_id={post_id}: {post.image}")

    db.delete(post)
    db.commit()

    logger.info(f"[POST DELETE] Success for post_id={post_id} by user_id={current_user.id}")
    return {"message": "Post deleted successfully"}
