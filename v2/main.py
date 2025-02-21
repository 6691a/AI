import json

from sqlalchemy import select

from fastapi import FastAPI, Depends, WebSocket
from fastapi.responses import HTMLResponse

from db import get_db_session
from model import Item, PaikdabangMenuDocument, TaxLawDocument
from langchain_core.documents import Document

from ai import PaikdabangAI

app = FastAPI()


@app.get("/item")
async def get(db = Depends(get_db_session)):
    vector = [3, 1, 2]

    async with (db as session):
        query = (
            select(Item, Item.embedding.cosine_distance(vector).label('distance'))
            .order_by(Item.embedding.cosine_distance(vector))
            .limit(4)
        )
        items = await session.execute(query)

    return [
        {
            "id": item.id,
            "distance": distance
        }
        for item, distance in items.all()
    ]


@app.get("/setup")
async def setup(db = Depends(get_db_session)):
    """
    빽다방.txt 파일을 읽어서 데이터베이스에 저장합니다.
    """
    docs: list[Document] = PaikdabangMenuDocument.load("빽다방.txt")
    split_docs = PaikdabangMenuDocument.split(docs)
    page_contents = [doc.page_content for doc in split_docs]

    embeds = PaikdabangMenuDocument.embed(page_contents)

    async with (db as session):
        items = [
            PaikdabangMenuDocument.model_validate({
                "page_content": doc.page_content,
                "meta_data": doc.metadata,
                "embedding": embed

            })
            for doc, embed in zip(split_docs, embeds)
        ]

        def _bulk_op(sync_session, items):
            sync_session.bulk_save_objects(items)

        await session.run_sync(_bulk_op, items)
        await session.commit()
    return {"message": "success"}


@app.get("/query")
async def main(db = Depends(get_db_session)):
    question_embedding = PaikdabangMenuDocument.embeddings("빽다방 고카페인 음료 종류는?")

    async with (db as session):
        query = (
            select(PaikdabangMenuDocument, PaikdabangMenuDocument.embedding.cosine_distance(question_embedding).label('distance'))
            .order_by(PaikdabangMenuDocument.embedding.cosine_distance(question_embedding))
            .limit(4)
        )
        items = await session.execute(query)

    return [
        {
            "id": item.id,
            "distance": distance,
            "page_content": item.page_content
        }
        for item, distance in items.all()
    ]


html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat {text}</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off" placeholder="메시지를 입력하세요..."/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/{endpoint}");
            var messageMap = new Map();

            ws.onmessage = function(event) {{
                const data = JSON.parse(event.data);
                const messageId = data.message_id;
                const messageContent = data.message;
                const messageType = data.message_type;

                if (!messageMap.has(messageId)) {{
                    // 새 메시지 생성
                    const messages = document.getElementById('messages');
                    const messageElement = document.createElement('li');
                    const content = document.createElement('div');

                    content.className = 'message ' + (messageType === 'ai' ? 'ai-message' : 'user-message');
                    content.textContent = messageContent;

                    messageElement.appendChild(content);
                    messages.appendChild(messageElement);
                    messageMap.set(messageId, content);

                    // 자동 스크롤
                    window.scrollTo(0, document.body.scrollHeight);
                }} else {{
                    // 기존 메시지 업데이트 (스트리밍)
                    const existingMessage = messageMap.get(messageId);
                    existingMessage.textContent += messageContent;
                    window.scrollTo(0, document.body.scrollHeight);
                }}
            }};

            function sendMessage(event) {{
                event.preventDefault();
                const input = document.getElementById("messageText");
                const message = input.value.trim();

                if (message) {{
                    // 사용자 메시지 표시
                    const messages = document.getElementById('messages');
                    const messageElement = document.createElement('li');
                    const content = document.createElement('div');

                    content.className = 'message user-message';
                    content.textContent = message;

                    messageElement.appendChild(content);
                    messages.appendChild(messageElement);

                    // 메시지 전송
                    ws.send(message);
                    input.value = '';

                    // 자동 스크롤
                    window.scrollTo(0, document.body.scrollHeight);
                }}
            }}
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html.format(endpoint="ws/stream", text="stream"))


@app.websocket("/ws/stream")
async def websocket_endpoint_stream(websocket: WebSocket, db = Depends(get_db_session), ai = Depends(PaikdabangAI)):
    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        ai: PaikdabangAI
        async with (db as session):
            message = ai.astream(data, session=session)
            async for chunk in message:
                await websocket.send_json(chunk.model_dump())


@app.get("/tax/setup")
async def tax_setup(db = Depends(get_db_session)):
    """
    세금 계산 데이터를 데이터베이스에 저장합니다.
    """
    file_path = "sample-taxlaw-1000.jsonl"
    with open(file_path, "r") as file:
        lines = file.readlines()

    json_docs = [json.loads(line) for line in lines]

    page_contents = [doc["page_content"] for doc in json_docs]

    embeds = TaxLawDocument.embed(page_contents)
    docs = [
        TaxLawDocument(
            embedding=embed,
            page_content=doc["page_content"],
            meta_data=doc["metadata"],
        ) for doc, embed in zip(json_docs, embeds)
    ]

    async with (db as session):
        items = [
            TaxLawDocument.model_validate(doc)
            for doc in docs
        ]

        def _bulk_op(sync_session, items):
            sync_session.bulk_save_objects(items)

        await session.run_sync(_bulk_op, items)
        await session.commit()
    return {"message": "success"}

@app.get("/tax/query")
async def tax_query(db = Depends(get_db_session)):
    question_embedding = TaxLawDocument.embeddings("재화 수출하는 경우 영세율 첨부 서류로 수출실적명세서가 없는 경우 해결 방법")

    async with (db as session):
        query = (
            select(TaxLawDocument, TaxLawDocument.embedding.cosine_distance(question_embedding).label('distance'))
            .order_by(TaxLawDocument.embedding.cosine_distance(question_embedding))
            .limit(4)
        )
        items = await session.execute(query)

    return [
        {
            "id": item.id,
            "distance": distance,
            "page_content": item.page_content
        }
        for item, distance in items.all()
    ]
