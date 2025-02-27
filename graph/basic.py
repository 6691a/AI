import re

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, END
from langchain_core.prompts import ChatPromptTemplate

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
        # self.graph_builder.add_node("check_think_language", self.check_think_language_node)
        self.graph_builder.add_node("check_helpfulness", self.check_helpfulness_grader_node)

    def graph_edge_setup(self):
        self.graph_builder.add_edge(START, "think")
        # self.graph_builder.add_edge("think", "check_think_language")

        # self.graph_builder.add_conditional_edges(
        #     "check_think_language",
        #     self.check_think_language_route,
        #     {
        #         'is_english': 'generate',
        #         'not_english': 'think',
        #     }
        # )
        self.graph_builder.add_edge("think", "generate")
        self.graph_builder.add_edge("generate", "check_helpfulness")
        self.graph_builder.add_conditional_edges(
            "check_helpfulness",
            self.check_helpfulness_grader_route,
            {
                'helpful': END,
                'unhelpful': 'think',
            }
        )

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

    def check_think_language_route(self, state: BasicState) -> Literal["is_english", "not_english"]:
        """
        Check if the provided text is in English.
        Return only a single digit: 1 if the text is entirely in English, or 0 if it contains any non-English content.

        종종 낮은 모델을 사용하면 영어가 아닌 언어로 답변하는 경우가 있습니다 그럴경우 해당 노드를 통해 검사 후 다시 think 노드로 이동합니다.
        영어가 나온다는걸 확신 할 수 있다면 해당 노드를 사용하지 않아도 됩니다.
        """
        language = state["language"]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are a language detector. Check if the provided text is in English.
                Return only a single digit: 1 if the text is entirely in English, or 0 if it contains any non-English content.
            """),
            ("human", "{thinking}")
        ])

        chain = prompt | self.models["num_predict"]

        response = chain.invoke({
            "thinking": state["thinking"]
        })
        is_english = "1" in response.content

        return "is_english" if is_english else "not_english"

    def check_think_language_node(self, state: BasicState) -> BasicState:
        """
        가독성을 위한 노드
        """
        return state

    def check_helpfulness_grader_route(self, state: BasicState) -> Literal['helpful', 'unhelpful']:
        # TODO: deppseek-r1 모델은 tools를 지원하지 않아 langchain-ai/rag-answer-helpfulness 프롬프트를 복사하여 사용합니다.
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are a teacher grading a quiz. 
                You will be given a QUESTION and a STUDENT ANSWER. 

                Grade criteria:
                (1) The STUDENT ANSWER must be concise and directly relevant to the QUESTION
                (2) The STUDENT ANSWER must be helpful in addressing the QUESTION

                Scoring system (BINARY ONLY):
                - Score = 1: The answer meets ALL criteria above. This is a helpful answer.
                - Score = 0: The answer fails to meet ANY of the criteria. This is not a helpful answer.

                IMPORTANT: You MUST assign ONLY a 0 or 1 score - no other values are permitted.

                First, analyze the answer against each criterion.
                Then provide your reasoning for the final score.
                Finally, output your score in this exact format at the end: "FINAL SCORE: [0 or 1]"
            """),
            ("human", """
            QUESTION: {question}
            STUDENT ANSWER: {answer}
            """)
        ])
        query = state["question"]
        answer = state["answer"]


        # chain = prompt | self.default_model
        chain = prompt | self.models["check"]

        response = chain.invoke({
            "question": query,
            "answer": answer
        })

        cleaned_content = re.sub(
            r"<think>.*?</think>\n?",
            "", response.content,
            flags=re.DOTALL
        )

        score = "1" in cleaned_content
        print(cleaned_content)
        print("helpful" if score else "unhelpful")
        return "helpful" if score else "unhelpful"

    def check_helpfulness_grader_node(self, state: BasicState) -> BasicState:
        """
        가독성을 위한 노드
        """
        return state

    def get_language_prompt(self, language: Language, **kwargs) -> ChatPromptTemplate:
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