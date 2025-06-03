import pandas as pd
from openai import OpenAI
import instructor
from instructor.batch import BatchJob

from pydantic import BaseModel, Field
from typing import List

import os
from dotenv import load_dotenv

# load .env file to environment
load_dotenv()

API_KEY = os.getenv("API_KEY")
# MODEL_NAME = "o3-2025-04-16"
MODEL_NAME = "gpt-4o-mini"

dataset_path = "train_df.csv"
output_path = "data_with_tone_analysis_mini.csv"


# Example response models
class ClassificationResponse(BaseModel):
    us_vs_them_lang: int = Field(
        ...,
        ge=0,
        le=2,
        description="Score 0-2: 0=Neutral, 1=Moderate rivalry, 2=Extreme demonization. How severely does the post frame opponents as evil/threatening?",
    )
    exaggerated_uncertainty: float = Field(
        ...,
        ge=0,
        le=1,
        description="Output a certainty score 0.0-1.0: 0.0=Speculative, 1.0=Absolute certainty. How definitively are claims presented?",
    )
    source_quality: int = Field(
        ...,
        ge=0,
        le=2,
        description="Score 0-2: 0=Specific evidence, 1=Vague sourcing, 2=No evidence. How verifiable are the claims?",
    )
    victim_villain_language: int = Field(
        ...,
        ge=0,
        le=1,
        description="Answer YES(1)/NO(0): Does the post frame an issue as 'good people harmed by evil actors'?",
    )
    black_and_white_language: int = Field(
        ...,
        ge=0,
        le=1,
        description="Answer YES(1)/NO(0): Does the post reduce a complex issue to one cause, two choices, or blame a single group?",
    )
    dehumanization: int = Field(
        ...,
        ge=0,
        le=2,
        description="Score 0-2: 0=Respectful, 1=Negative labeling, 2=Dehumanizing. How are opponents/minorities described?",
    )
    emotionality: float = Field(
        ...,
        ge=-1,
        le=1,
        description="Emotionality of the text from -1 (very negative) to 1 (very positive)",
    )
    reading_difficulty: float = Field(
        ...,
        ge=0,
        le=1,
        description="Reading difficulty of the text from 0 (very easy) to 1 (very difficult)",
    )
    sentiment: int = Field(
        ...,
        ge=-1,
        le=1,
        description="Overall sentiment of the text (positive (1)/negative (-1)/neutral (0))",
    )
    polarization: float = Field(
        ...,
        ge=0,
        le=1,
        description="Polarization of the text from 1 (very polarizing) to 0 (very nuanced)",
    )


def get_prompt(row):
    prompt = f"""Evaluate the following statement using the provided information.
        
        Statement: {row['statement']}
        Date: {row['date']}
        Speaker: {row['speaker']}
        Speaker description: {row['speaker_description']}
    """
    return prompt

if __name__ == "__main__":
    client = instructor.from_openai(
        OpenAI(
            api_key=API_KEY,
        ),
        mode=instructor.Mode.JSON,
    )
    df = pd.read_csv(dataset_path)
    responses_list = []
    for index, row in df.iterrows():
        prompt = get_prompt(row)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class AI that excels at analyzing tone and sentiment of statements. You're about to be given a statement and asked to analyze the tone and sentiment of the statement.",
                },
                {
                    "role": "user",
                    "content": get_prompt(row),
                }
            ],
            max_retries=3,
            response_model=ClassificationResponse,
        )
        print(response)
        responses_list.append(response.model_dump())

    # Convert list of dictionaries to dataframe and merge with original
    responses_df = pd.DataFrame(responses_list)
    print(responses_df)

    df = pd.concat([df, responses_df], axis=1)
    df.to_csv(output_path, index=False)
    print(df)

    # with open("Data_Predicted-o3.csv", "a") as f:
    #     df.to_csv(f, index=False)
