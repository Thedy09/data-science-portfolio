# gpt_summary.py
import os
from openai import OpenAI

def generate_gpt_summary(text_block: str) -> str:
    # Charger la clé API à chaque appel
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_key:
        raise RuntimeError('OpenAI key not set.')
    
    if not text_block or not text_block.strip():
        raise RuntimeError('No text to summarize.')
    
    # Initialiser le client
    client = OpenAI(api_key=openai_key)
    
    prompt = f"""You are a data scientist assistant. Summarize the following dataset analysis in 6-8 short bullet points, focusing on: dataset shape, missing data, top numeric statistics, potential outliers, and suggested next steps.

    Analysis text:
    {text_block}
    """
    
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role':'user','content':prompt}],
            temperature=0.2,
            max_tokens=400
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        return "No response generated"
        
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}")
