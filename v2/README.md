1. install the dependencies
```bash
    uv sync
```
2. ollama model download
```bash
    ollama pull exaone3.5:7.8b
    ollama pull snowflake-arctic-embed2:latest
```

3. run the project
```bash
    uv run fastapi dev
```

4. migrate the database
- 다른 임베딩 모델을 사용한다면 해당 모델에 맞는 마이그레이션을 진행해야 합니다(.env 파일을 수정해주세요.)
```bash
    uv run  alembic upgrade head
```

4. open the browser and go to the following url
```bash
    http://localhost:8000/
```

---

### 간단한 채팅 API 서버
- 간단한 질문에 대한 답변을 하는 AI 채팅
- 질문에 항상 빽다방 정보를 포함하여 빽다방 이외는 답변이 불가능한 단순한 AI 채팅
- V1에서는 List를 통해 Vector를 관리, V2에서는 PSQL DB를 통해 Vector를 관리합니다.


