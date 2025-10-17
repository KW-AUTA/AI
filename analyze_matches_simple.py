from service.component_test import execute_component_mapping_test
import asyncio

base_url = 'https://heeeesssuu.github.io/web3studio'
json_url = 'https://api.figma.com/v1/files/PbZUlZHMdFfAWfpKLYNkRd?depth=10'
current_page = '78:2'

print('\n' + '=' * 80)
print('매칭 결과 상세 분석')
print('=' * 80 + '\n')

matches = asyncio.run(execute_component_mapping_test(base_url, current_page, json_url))

print('\n' + '=' * 80)
print(f'매칭 결과: 총 {len(matches)}개')
print('=' * 80 + '\n')

# 각 매칭 결과 출력
for i, match in enumerate(matches, 1):
    print(f'[Match #{i}]')

    # ExtractedElement에서 텍스트 추출
    comp_text = match.componentName.text if hasattr(match.componentName, 'text') else 'N/A'
    comp_box = match.componentName.box if hasattr(match.componentName, 'box') else 'N/A'

    print(f'  Component Text: {comp_text}')
    print(f'  Component Box: {comp_box}')
    print(f'  Destination Page: {match.destinationFigmaPage}')
    print(f'  Destination URL: {match.destinationUrl}')
    print(f'  Success: {match.isSuccess}')
    print(f'  Routing: {match.isRouting}')

    if match.failReason:
        print(f'  Fail Reason: {match.failReason}')
    print()

print('=' * 80)
print('분석 완료')
print('=' * 80)
