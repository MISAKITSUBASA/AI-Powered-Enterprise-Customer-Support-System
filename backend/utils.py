import functools
import contextlib
from sqlalchemy.orm import Session

@contextlib.contextmanager
def db_transaction(db: Session):
    """
    Context manager for database transactions with proper error handling
    """
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
