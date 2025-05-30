from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from model.project import Project
from routes.dto.response.mapping_response import MappingInfo

async def execute_component_mapping_test(current_url: str, current_page: str, project_id: int, db: AsyncSession):
    stmt = select(Project).where(Project.id == project_id)
    result = await db.execute(stmt)
    project = result.scalars().first()
    print(project.figma_json)

    return [
        MappingInfo(
            componentName="LoginButton",
            destinationFigmaPage="HomePage",
            destinationUrl="/home",
            actualUrl="/home",
            failReason=None,
            isSuccess=True,
            isRouting=True
        ),
        MappingInfo(
            componentName="CancelButton",
            destinationFigmaPage=None,
            destinationUrl=None,
            actualUrl=None,
            failReason=None,
            isSuccess=True,
            isRouting=False
        )
    ]