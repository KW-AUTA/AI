from yolo.mapping import mapping


async def execute_component_mapping_test(current_url: str, current_page: str, figma_url: str):
    try:
        print(figma_url)
        mapping_infos = mapping(current_url, figma_url)
        return mapping_infos
    except Exception as e:
        print(f"Error in mapping() function: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print("Traceback:", traceback.format_exc())
        return []
