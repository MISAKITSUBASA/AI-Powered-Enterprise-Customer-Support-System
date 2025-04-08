import os
import bcrypt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Import database models
from models import Base, User

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./customer_support.db")

# Handle special case for PostgreSQL from Heroku/AWS RDS
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create database directory if it doesn't exist (for SQLite)
if DATABASE_URL.startswith("sqlite"):
    db_path = DATABASE_URL.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

# Connect to database
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

# Create tables
print("Creating database tables...")
Base.metadata.create_all(bind=engine)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

# Check if we need to create a default admin user
DEFAULT_ADMIN_USERNAME = os.getenv("DEFAULT_ADMIN_USERNAME")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD")

if DEFAULT_ADMIN_USERNAME and DEFAULT_ADMIN_PASSWORD:
    # Check if user already exists
    existing_user = session.query(User).filter(User.username == DEFAULT_ADMIN_USERNAME).first()
    
    if not existing_user:
        print(f"Creating default admin user: {DEFAULT_ADMIN_USERNAME}")
        
        # Hash password
        hashed = bcrypt.hashpw(DEFAULT_ADMIN_PASSWORD.encode("utf-8"), bcrypt.gensalt())
        
        # Create admin user
        admin_user = User(
            username=DEFAULT_ADMIN_USERNAME,
            password_hash=hashed.decode("utf-8"),
            is_admin=True
        )
        
        session.add(admin_user)
        session.commit()
        print("Default admin user created successfully")
    else:
        print(f"Admin user {DEFAULT_ADMIN_USERNAME} already exists")

# Close session
session.close()
print("Database initialization complete")
