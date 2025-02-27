from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

from graph.base import BaseGraph, BaseState
from graph.base.state import Language


class BasicState(BaseState):
    question: str
    thinking: str
    answer: str


class BasicGraph(BaseGraph):
    def __init__(self, models: dict[str, BaseChatModel], default_model_key: str = "default"):
        super().__init__(models, BasicState, default_model_key)

    def graph_node_setup(self):
        self.graph_builder.add_node("think", self.think_node)
        self.graph_builder.add_node("generate", self.generate_node)

    def graph_edge_setup(self):
        self.graph_builder.add_edge(START, "think")
        self.graph_builder.add_edge("think", "generate")
        self.graph_builder.add_edge("generate", END)

    def think_node(self, state: BasicState):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            IMPORTANT INSTRUCTION: You are a reasoning assistant. You MUST ALWAYS respond in ENGLISH ONLY. 
            DO NOT use Chinese or any other language under any circumstances.
            Your task is to provide clear, logical reasoning for questions.
            """),
            ("human", """
            I need your reasoning in ENGLISH ONLY for this question:

            {question}

            Remember: Respond ONLY in English, even if the question is in another language.
            """)
        ])

        model = self.get_think_model()
        chain = prompt | model

        response = chain.invoke({
            "question": state["question"]
        })

        return {
            "thinking": response.content
        }

    def generate_node(self, state: BasicState):
        language = state["language"]
        model = self.get_language_model(language)
        prompt = self.get_language_prompt(language)

        chain = prompt | model
        response = chain.invoke({
            "question": state["question"],
            "think": state["thinking"]
        })

        return {
            "answer": response.content
        }

    def get_language_prompt(self, language: Language) -> ChatPromptTemplate:
        if language == Language.KOR:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                    IMPORTANT INSTRUCTION: 한국어로만 답변하는 도우미입니다.

                    당신은 다음을 받게 됩니다:
                    1. 사용자의 질문(question)
                    2. 영어로 작성된 생각 과정(think)

                    답변 작성 규칙:
                    - 'think'의 모든 정보와 추론 과정을 답변에 완전히 포함시켜야 합니다
                    - 'think'에서 언급된 모든 사실, 계산, 논리를 자연스럽게 설명하세요
                    - 'think'라는 단어나 외부 추론 과정이 있었다는 사실을 절대 언급하지 마세요
                    - "제공된 사고 과정에서는", "영어 추론에 따르면" 같은 표현을 사용하지 마세요
                    - 마치 처음부터 당신이 직접 추론한 것처럼 자연스럽게 설명하세요
                """),
                ("human", "question {question}\nthink {think}"),
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                    당신은 다음을 받게 됩니다:
                    1. 사용자의 질문(question)
                    2. 영어로 작성된 생각 과정(think)

                    답변 작성 규칙:
                    - 'think'의 모든 정보와 추론 과정을 답변에 완전히 포함시켜야 합니다
                    - 'think'에서 언급된 모든 사실, 계산, 논리를 자연스럽게 설명하세요
                    - 'think'라는 단어나 외부 추론 과정이 있었다는 사실을 절대 언급하지 마세요
                    - "제공된 사고 과정에서는", "영어 추론에 따르면" 같은 표현을 사용하지 마세요
                    - 마치 처음부터 당신이 직접 추론한 것처럼 자연스럽게 설명하세요
                """),
                ("human", "question {question}\nthink {think}"),
        ])
        return prompt

    def get_language_model(self, language: Language, **kwargs) -> BaseChatModel:
        if language == Language.KOR:
            return self.models["kor"]
        return self.default_model