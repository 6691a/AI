from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from ai import PaikdabangAI

ai = PaikdabangAI()
app = FastAPI()

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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        ai_response: str = await ai.ainvoke(data)
        await websocket.send_text(ai_response)


@app.websocket("/ws/stream")
async def websocket_endpoint_stream(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        message = ai.astream(data)
        async for chunk in message:
            await websocket.send_json(chunk.model_dump())