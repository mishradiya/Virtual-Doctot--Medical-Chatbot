##########################################################
## Author:  Divya Acharya
## Project: V-Doc
## File: test_rag.py
## Date: Aug 27, 2024
## Purpose: evaluating RAG using deep eval framework.
##########################################################
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM
from typing import List, Tuple

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase


class CustomChatOpenAI(DeepEvalBaseLLM):
    def __init__(
        self, api_key: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    ):
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name, api_key=api_key, base_url="https://api.aimlapi.com"
        )

    def load_model(self):
        return self.llm

    def generate(self, messages: List[Tuple[str, str]]) -> str:
        response = self.llm.invoke(messages)
        return response.content

    async def a_generate(self, messages: List[Tuple[str, str]]) -> str:
        return self.generate(messages)

    def get_model_name(self):
        return self.model_name


# Usage example
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("MLAI_API_KEY")

# Initialize the custom wrapper
custom_llm = CustomChatOpenAI(api_key=openai_api_key)

# Define test cases
test_cases_to_evaluate = [
    LLMTestCase(
        input="What is Mediterranean diet?",
        actual_output="The Mediterranean diet is a dietary approach that originated in the olive-growing areas of the Mediterranean region and can be promoted to improve health and wellbeing. Definitions vary, but it typically includes consuming fruits, vegetables, nuts, legumes, whole grains, fish, and olive oil, while limiting red meat, processed foods, and saturated fats.",
        expected_output="Mediterranean diet consist of plant-based foods, low intake of dairy products and moderate intake of fish and poultry.",
        retrieval_context=[
            """The Mediterranean diet is a well-researched dietary approach that can be promoted to patients to improve 
            their health and wellbeing. The Mediterranean diet originated in the olive-growing areas of the Mediterranean 
            region and still has a strong cultural association with these areas. While definitions vary, the Mediterranean 
            diet is generally characterised by a “high intake of plant-based foods (fruit, vegetables, nuts and cereals) and 
            olive oil; a moderate intake of fish and poultry; a low intake of dairy products (principally yoghurt and cheese), 
            red meat, processed meats and sweets (for which fresh fruit is often substituted); and a moderate wine intake, 
            normally consumed with meals”."""
        ]
    )
]

test_cases_to_evaluate = [
    LLMTestCase(
        input="What is diabetic ketoacidosis?",
        actual_output="Diabetic ketoacidosis (DKA) is a serious complication of diabetes that occurs when your body produces high levels of blood acids called ketones. It develops when your body can't produce enough insulin. Without enough insulin, your body begins to break down fat as fuel, which produces ketones.",
        expected_output="Diabetic ketoacidosis (DKA) is a condition where there is a lack of insulin, leading to ketones forming in the blood due to the breakdown of fats. It often happens in people with diabetes and can cause symptoms such as vomiting, confusion, and difficulty breathing.",
        retrieval_context=[
            """DKA occurs when there is not enough insulin to meet the body’s basic needs. Causes include newly diagnosed 
            diabetes, missed insulin injections, or illness. It leads to symptoms such as upset stomach, vomiting, 
            deep breathing, and fruity-smelling breath."""
        ]
    )
]

test_cases_to_evaluate = [
    LLMTestCase(
        input="What are the symptoms of type 1 diabetes?",
        actual_output="Symptoms of type 1 diabetes include frequent urination, increased thirst, fatigue, blurred vision, and unexplained weight loss.",
        expected_output="Symptoms of type 1 diabetes include frequent urination, increased thirst, increased hunger, weight loss, tiredness, and behavioral changes such as irritability or mood swings.",
        retrieval_context=[
            """Common symptoms of type 1 diabetes include frequent urination, increased thirst, hunger, weight loss, 
            and mood changes. These symptoms are caused by high blood glucose levels due to a lack of insulin."""
        ]
    )
]

test_cases_to_evaluate = [
    LLMTestCase(
        input="How is insulin administered to patients with diabetes?",
        actual_output="Insulin is administered through injections.",
        expected_output="Insulin is administered through injections or insulin pumps. It is injected into the fatty tissue under the skin using a syringe or insulin pen, or continuously delivered via an insulin pump for better blood glucose control.",
        retrieval_context=[
            """Insulin must be injected into the fatty tissue under the skin. It can be administered through syringes, 
            insulin pens, or insulin pumps that deliver insulin continuously throughout the day."""
        ]
    )
]

test_cases_to_evaluate = [
    LLMTestCase(
        input="How does physical activity affect blood glucose levels in diabetes management?",
        actual_output="Physical activity lowers blood glucose levels and helps improve insulin sensitivity, reducing the need for insulin.",
        expected_output="Physical activity lowers blood glucose levels by increasing insulin sensitivity, allowing glucose to enter the cells more efficiently. Regular exercise helps in maintaining a stable blood sugar level and reduces the amount of insulin needed for glucose management.",
        retrieval_context=[
            """Physical activity helps lower blood glucose levels by improving the body’s sensitivity to insulin. Regular 
            exercise can also aid in weight management, which is crucial in managing both type 1 and type 2 diabetes."""
        ]
    )
]

test_cases_to_evaluate = [
    LLMTestCase(
        input="What are the long-term complications of poorly controlled diabetes?",
        actual_output="Poorly controlled diabetes can lead to complications such as kidney failure, heart disease, and nerve damage.",
        expected_output="Long-term complications of poorly controlled diabetes include cardiovascular diseases, kidney failure, blindness, nerve damage, and amputations due to poor circulation and diabetic foot problems.",
        retrieval_context=[
            """Complications of poorly controlled diabetes include cardiovascular disease, kidney failure, blindness, and 
            amputations due to diabetic foot and poor circulation. These complications arise from consistently high blood 
            sugar levels."""
        ]
    )
]


# Define metrics
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=custom_llm)
contextual_precision = ContextualPrecisionMetric(model=custom_llm)
contextual_recall = ContextualRecallMetric(model=custom_llm)
contextual_relevancy = ContextualRelevancyMetric(model=custom_llm)

# Evaluate
evaluation_result = evaluate(
    test_cases_to_evaluate,
    [answer_relevancy, contextual_precision, contextual_recall, contextual_relevancy],
)
print(evaluation_result)