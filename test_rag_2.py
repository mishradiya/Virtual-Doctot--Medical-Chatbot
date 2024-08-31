##########################################################
## Author:  Divya Acharya
## Project: V-Doc
## File: test_rag_2.py
## Date: Aug 27, 2024
## Purpose: evaluating RAG using ragas framework.
##########################################################from datasets import Dataset
from ragas.metrics import context_precision
from ragas import evaluate

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


chat = ChatOpenAI(
    # model="gpt-4o",
    # model="gpt-3.5-turbo-0125",
    model="gpt-3.5-turbo-0125",
    # temperature=0,
    # max_tokens=None,
    # timeout=None,
    max_retries=2,
    api_key="808881a495ba4b65ae883559f7f1d9a3",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="https://api.aimlapi.com",
    # organization="...",
    # other params...
)

vllm = LangchainLLMWrapper(chat)

from ragas.metrics import (
    context_precision,
    faithfulness,
    context_recall,
)

faithfulness.llm = vllm
context_precision.llm = vllm
context_recall.llm = vllm
data_samples = {
    "question": [
        "What is Diabetic foot?",
         "What is Mediterranean diet?"
    ],
    "answer": [
        "Diabetic foot refers to a group of complications that people with diabetes may experience in their feet as a result of neuropathy, peripheral arterial disease, or other health conditions related to diabetes. Risk factors for developing foot ulcers and potential symptoms include peripheral vascular disease, neuropathy, poor glycemic control, smoking, diabetic nephropathy, previous foot ulceration/amputation, pain in the legs or cramping during physical activity, tingling, burning, or loss of ability to feel heat or cold, and changes in shape or temperature of the feet. It's important for individuals with diabetes to take extra care of their feet and consult a healthcare professional if they notice any signs of potential complications. Additionally, amputations are more common in people with diabetes compared to those without the disease.",
        "The Mediterranean diet is a dietary approach that originated in the olive-growing areas of the Mediterranean region and can be promoted to improve health and wellbeing. Definitions vary, but it typically includes consuming fruits, vegetables, nuts, legumes, whole grains, fish, and olive oil, while limiting red meat, processed foods, and saturated fats.",

    ],
    "contexts": [
        [
            "in amputation.\n127 \n Amputations are estimated to be 10 to 20 times more common in people with diabetes than in those without \nthe disease, and it is estimated that every 30 seconds a lower extremity amputation is taking place somewhere \nin the world as a result of diabetes.\n124 \n\uf0b7 Risk factors for developing foot ulcers  Peripheral vascular disease, neuropathy, poor glycaemic \ncontrol, cigarette smoking, diabetic nephropa thy, previous foot ul ceration/amputation. \n\uf0b7 Symptoms  Pain in the legs or cramping in the thighs or calves during physical activity, tingling, \nburning, or pain in the feet, loss of sense of touch or ability to feel heat or cold very well, a change in \nthe shape of feet over time, dry cracked skin on th e feet, a change in the colour and temperature of",
            "Diabetes prevention, screening, and management: A handbook for pharmacists | p35  \n \n \n \n the feet, thickened yellow toenails, fungal infections between the toes, blisters, sores, ulcers, infected \ncorns, ingrow n toenails. \nPharmacists should ensure they are educating all patien ts with diabetes on the importance of proper foot \ncare. Table 11 provides diabetes foot care counselling ti ps pharmacists can use. If regulations allow, they can \nalso provide foot screenings for their patients. For addi tional information on how to conduct these screenings, \nyou can refer to the IDF’s Clinical Practice Recommendations on the Diabetic Foot.128 \n \nTable 11 - WHO diabetes foot care tips39 \nDiabetes foot care counselling tips \nInspect your feet daily. Chec k for cuts, blisters, redness, \nswelling or nail problems. Use a magnifying hand mirror to look at the bottom of your feet. Shake out your shoes and feel the insides before"
        ],
        [
            "who consumed the lowest-GI diets.\n148 This is further supported by a meta-analysis of prospective cohort \nstudies which states that food and nutrition advice that favours low-GI diets has the potential to produce cost savings for healthcare systems.\n149, 150 \n To support patients who wish to integrate more low-GI foods into their diet, there are many online tools and \nlists that include the GI of foods. Examples includ e the Glycaemic Index Search Tool developed by the \nUniversity of Sydney,\n151 and the Glycaemic Index Food Guide developed by Diabetes Canada.152 \n \n7.1.3 Mediterranean diet \nThe Mediterranean diet is a well-researched dietary appr oach that can be promoted to patients to improve \ntheir health and wellbeing. The Mediterranean diet originated in the olive-growing areas of the Mediterranean \nregion and still has a strong cultural association with  these areas. While definitions vary, the Mediterranean",
            "Diabetes prevention, screening, and management: A handbook for pharmacists \n \n \n \n red meat, processed meats and sweets (for which fresh fr uit is often substituted); and a moderate wine intake, \nnormally consumed with meals”.153  \n The Mediterranean diet has been shown to benefit patien ts with type 2 diabetes and has been associated with \nimprovements in glycaemic control, cardiovascular risk factors and body weight in multiple meta-analyses.\n154, \n155 Another network meta-analysis comp ared nine dietary approaches and found that the Mediterranean diet \nwas the most effective in improving glycaemic control in patients with type 2 diabetes.156  \n There are also social and cultural factors associated wi th the Mediterranean diet, in cluding longer mealtimes, \npost-meal siestas, regular physical ac tivity and shared eating practices.\n157 The Mediterranean Diet Foundation \nhas developed 10 recommendations to  support individuals who wish to adopt the Mediterranean diet, that"

        ],
    ],
    "ground_truth": [
        "A diabetic foot disease is any condition that results directly from peripheral artery disease (PAD) or sensory neuropathy affecting the feet of people living with diabetes. Diabetic foot conditions can be acute or chronic complications of diabetes.[1] Presence of several characteristic diabetic foot pathologies such as infection, diabetic foot ulcer and neuropathic osteoarthropathy is called diabetic foot syndrome. The resulting bone deformity is known as Charcot foot.",
        "Mediterranean diet consist of plant-based foods, low intake of dairy products and moderate intake of fish and poultry."
    ],
}

dataset = Dataset.from_dict(data_samples)
score = evaluate(dataset, metrics=[context_precision, context_recall])
# print(score.to_pandas())