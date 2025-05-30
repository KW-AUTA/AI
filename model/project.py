from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Enum, ForeignKey, Date, DateTime
from sqlalchemy.orm import relationship, Mapped, mapped_column
import enum
import datetime

Base = declarative_base()

class ProjectStatus(str, enum.Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    NOT_STARTED = "NOT_STARTED"

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    figma_url: Mapped[str] = mapped_column("figma_url")
    figma_json: Mapped[str] = mapped_column("figma_json")
    root_figma_page: Mapped[str] = mapped_column("root_figma_page")
    service_url: Mapped[str] = mapped_column("service_url")
    project_name: Mapped[str] = mapped_column("project_name")
    description: Mapped[str] = mapped_column()
    expected_test_execution: Mapped[datetime.date] = mapped_column("expected_test_execution")
    project_created_date: Mapped[datetime.date] = mapped_column("project_created_date")
    project_end: Mapped[datetime.date] = mapped_column("project_end")

    project_status: Mapped[ProjectStatus] = mapped_column(Enum(ProjectStatus), nullable=False)
    test_execute_time: Mapped[datetime.datetime] = mapped_column("test_execute_time")
    test_rate: Mapped[int] = mapped_column("test_rate")
