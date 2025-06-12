from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from routes.dto.response.mapping_response import MappingInfo, MappingResponse
from yolo.mapping import mapping
import json
import os

async def execute_component_mapping_test(current_url: str, current_page: str, project_id: int, db: AsyncSession):
    try:
        # ai_db 데이터베이스의 임시 테이블에서 데이터 조회
        stmt = text("SELECT * FROM ai_db.my_temp_table WHERE id = :project_id")
        result = await db.execute(stmt, {"project_id": project_id})
        project = result.first()
        
        if not project:
            print(f"Project not found with id: {project_id}")
            return MappingResponse(mappings=[])
            
        print("Project data:", project)

        # 임시 JSON 파일 생성
        temp_json_path = f"/tmp/figma_json_{project_id}.json"
        with open(temp_json_path, "w") as f:
            json.dump(json.loads(project.mapping_data), f)  # mapping_data 컬럼 사용

        # 매핑 실행
        try:
            print("Calling mapping() function...")
            mapping_info = mapping(
                base_url=current_url,
                json_path=temp_json_path  # figma_json 대신 json_path 사용
            )
            print("Mapping result:", mapping_info)
            return mapping_info
        except Exception as e:
            print(f"Error in mapping() function: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print("Traceback:", traceback.format_exc())
            return MappingResponse(mappings=[])
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_json_path):
                os.remove(temp_json_path)

    except Exception as e:
        print(f"Error in execute_component_mapping_test: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print("Traceback:", traceback.format_exc())
        return MappingResponse(mappings=[])
