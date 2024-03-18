import openai
from openai import OpenAI

from utils.get_env import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

"""
        !!! WRITE CLEAR AND SPECIFIC INSTRUCTIONS !!!
"""

"""
        BETTER PROMPT #1
"""

numbers = [1, 2, 3, 4, 5]

user_prompt_1 = f'''
  Do some random actions on your mind with the numbers below, delimited by triple backticks.
  Numbers: ```{numbers}```
'''

"""
        BETTER PROMPT #2
"""

user_prompt_2 = f'''
  Configure the NGINX Web Server for hosting.
  
  Following next steps:
  1. Some
  2. Some
  3. Some
'''

"""
        BETTER PROMPT #3
        
                  !!! USE RTF !!!
                ROLE | TASK | FORMAT
                ! USE SYSTEM ROLE !
            ! GOOD AND CLEAR PROMPT (TASK) !
                ! FORMAT RESPONSE !
"""
# Imported asking for clear format response

"""
        BETTER PROMPT #4
        
                   FEW-SHOT
  !!! GIVE TO PROMPT SOME EXAMPLES OF RESPONSE BY ADDING DIALOG SYSTEM WITH ROLES USER AND ASSISTANT (STYLE, FORMAT, ELSE) !!!
"""

"""
        BETTER PROMPT #5
        
            !!! WRITE STEP BY STEP INSTRUCTIONS !!!
"""

"""
        BETTER PROMPT #6
                TIME TO THINK FOR MODEL :)
                    
                    GIVE TO SYSTEM ROLE TASK LIKE:
    !!! DO THIS TASK AND AFTER YOU GET YOUR OWN SOLUTION CHECK THE EQUALS BETWEEN YOUR SOLUTION AND MY SOLUTION !!!
"""

"""
        BETTER PROMPT #7
                
                ADD HINTS FOR MODEL
                - LIKE:
                
                1. write the simple function
                def (OR import)
                
                IT WILL BE GENERATE PYTHON FUNC BECAUSE WE PROVIDE THE PYTHON HINTS
                
"""

"""
        BETTER PROMPT #8
        
                !!! MODEL MUST WRITE ONLY FACTS !!!
                - LIKE:
                
                1. WRITE SOME POPULAR POEM FROM POPULAR AUTHOR
                CHECK ONLY FACTS FROM TRUE PUBLIC SOURCES SUCH AS: WIKI, else.
                IF YOU CAN'T FOUND, THEN JUST SAY I HAVE NOT INFORMATION
"""

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_prompt_2}
    ]
)

print(completion.choices[0].message.content)

"""
                                --- DALL-E MODEL ---
"""
prompt = "Generate a cool street-wearing boy in cap who sitting on coach with laptop and programming python applications." \
         "1. image must be from behind of this boy !"

response = client.images.generate(
    model='dall-e-2',
    prompt=prompt,
    style='vivid',
    # style='natural',
    size='1024x1024',
    quality='standard',
    n=1
)

image = open('static/main.png', 'rb')

image_create_response = client.images.create_variation(
    image=image,
    n=1,
    size='1024x1024'
)

mask = open('static/mask.png', 'rb')

image_edit_response = client.images.edit(
    image=image,
    mask=mask,
    prompt='The device takes a picture and theres "Python" title in the background.',
    n=1,
    size='1024x1024'
)

print(response.data[0].url)

"""
                                --- WHISPER MODEL --- 
"""

file = 'static/sound.mp3'
with open(file, 'rb') as audio_file:
    transcription = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_file
    )
    print(transcription.text)

rus_file = 'static/rus_sound.mp3'

with open('static/rus_sound.mp3', 'rb') as audio_file:
    translation = client.audio.translations.create(
        model='whisper-1',
        file=audio_file
    )
    print(translation.text)

"""
                                --- TTS MODEL --- 
"""

text = 'In your opinion, what kind of language policy should we have in Ukraine? We live in Ukraine. Ukrainian language is an ' \
       'evolutionary process. For me, the question of language has always been a matter of time. I believe that we should not break ' \
       'anyone. Just like they burned Ukrainian language from this territory. '

text_ua = "Якою, на Вашу думку, має бути мовна політика в Україні? Ми живемо в Україні. Українська мова - це еволюційний процес. " \
          "еволюційний процес. Для мене питання мови завжди було питанням часу. Я вважаю, що ми не повинні ламати нікого. нікого. Так " \
          "само, як спалили українську мову з цієї території. "

response = client.audio.speech.create(
    model='tts-1',
    voice='nova',
    input=text_ua
)
response.stream_to_file('tts.mp3')

"""
                                --- EMBEDDINGS MODEL ---          
"""
embedding = openai.embeddings.create(
    model="text-embedding-3-small",
    input="red"
)
print(embedding.data[0].embedding)

"""
                            DIALOG WITH GPT
"""

questions = []
bot_responses = []
messages = []

system_prompt = "Answer as concisely as possible."
messages.append({"role": "system", "content": system_prompt})

while True:
    current_question = input('Me:')

    if current_question in ['exit', 'quit']:
        break

    if current_question.lower() == '':
        continue

    messages.append({"role": "user", "content": current_question})
    questions.append(current_question)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.85
    )

    current_response = completion.choices[0].message.content
    print(f"Response: {current_response}")
    bot_responses.append(current_response)

    messages.append({"role": "assistant", "content": current_response})
