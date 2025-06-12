import mysql.connector
import json


def check_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="0419",  # 실제 비밀번호로 변경
        database="ai_db"
    )
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM my_temp_table")
        rows = cursor.fetchall()
        for row in rows:
            print("id:", row[0])
            print("project_id:", row[1])
            print("mapping_data(raw):", row[2])
            # mapping_data 컬럼이 JSON이면 파싱해서 예쁘게 출력
            try:
                parsed = json.loads(row[2])
                print("mapping_data(parsed):", json.dumps(parsed, indent=2, ensure_ascii=False))
            except Exception:
                pass
            print("-"*40)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    check_data() 