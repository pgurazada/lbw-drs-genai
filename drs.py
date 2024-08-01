import os
import base64

from dotenv import load_dotenv
from openai import OpenAI


def generate_data_uri(png_file_path: str):
    # Read the PNG image file in binary mode
    with open(png_file_path, 'rb') as image_file:
        image_data = image_file.read()
    
    # Encode the binary image data to base64
    base64_encoded_data = base64.b64encode(image_data).decode('utf-8')
    
    # Construct the data URI
    data_uri = f"data:image/png;base64,{base64_encoded_data}"
    
    return data_uri


def decision(image_path: str, client: OpenAI, lmm: str) -> str:

    image_data = generate_data_uri(image_path)

    user_message = """
    You are an expert in cricket tasked to judge whether a batter is out by Leg Before Wicket (LBW). You will be presented with an image of the ball hitting a batter's pads and you will have to make a decision whether the batter is to be judged out or not applying the following rules. 

    There are three important zones to consider  while making your decision. 
    - Pitching Zone: The Pitching Zone is the two-dimensional area that spans the length of the pitch, with the outer edges of the stumps at each end acting as its boundaries. 
    - Impact Zone: The Impact Zone is where the ball hits the batter’s pad for the first time. It is a three-dimensional area spaced between both sets of stumps, from ground level to an indefinite height. The outer edge of the leg and off stump act as its outer boundaries.
    - Wicket Zone: The Wicket Zone is a two-dimensional space, with the stumps, from their base to the top of the bails, and the full width from the outer edge of the off and leg stumps, bounding the area. 

    Given the image, you will need to project its trajectory between the impact zone and wicket zone. If you conclude that the ball will end up in the wicket zone, the batter needs to be adjudged "Out". If you conclude that the ball will end up outside the wicket zone, the batter needs to be adjudged "Not Out".  Specifically, you should check that:
    - The ball pitches in line between the wicket and wicket or on the off side of the batters wicket (i.e., the ball should be in the pitching zone or on the off side of the batter's wicket)
    - The batter intercepts the ball with any part of their body, except their hands, either full-pitch or after pitching (i.e., the impact zone should be within the leg and off stump)
    - The point of impact is between wicket and wicket, even if above the level of the bails (i.e., it should be in the wicket zone)

    Present your final answer as a JSON object with the following structure:
    {{
    "decision": out or not out
    "reason": detailed explanation of which rules were applied and why the decision was made
    }}

    Do not output anything before or after the JSON.
    """
    
    decision_prompt = [
        {
            'role': 'user', 
            'content': [
                {"type": "text", "text": user_message},
                {"type": "image", "image_url": {"url": image_data}}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=lmm,
            messages=decision_prompt,
            temperature=0
        )
        decision = response.choices[0].message.content
    except Exception as e:
        decision = e

    return decision

if __name__ == "__main__":

    load_dotenv()

    image_path = "data/example2.png"

    anyscale_api_key = os.environ['ANYSCALE_API_KEY']
    lmm = "llava-hf/llava-v1.6-mistral-7b-hf"

    client = OpenAI(
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key=anyscale_api_key
    )

    verdict = decision(
        image_path,
        client,
        lmm
    )

    print(verdict)