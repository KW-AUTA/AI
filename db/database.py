from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from db.setting import Settings

settings = Settings()

engine = create_async_engine(settings.database_url, echo=True)
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
