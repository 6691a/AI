from langchain_ollama import ChatOllama

from chat.ai.deepseek.r1 import DeepSeekR1, DeepSeekR1Params
from chat.ai.exaone.exaone3_5 import Exaone3_5Params, Exaone3_5
from graph.base import Language

from graph.basic import BasicGraph



def main():
    deepseek = DeepSeekR1(DeepSeekR1Params.MEDIUM, temperature=0)
    exaone = Exaone3_5(Exaone3_5Params.SMALL)
    qwen = ChatOllama(model="qwen2.5:7b")
    models = {
        "default": deepseek(),
        "kor": exaone(),
        "check": qwen,
    }
    graph = BasicGraph(models=models)()

    print(
        graph.invoke({
            'question': '한국에서 저녁 메뉴를 추천해주세요',
            'language': Language.KOR,
        })["answer"]
    )


if __name__ == "__main__":
    main()
