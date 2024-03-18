import os

"""
                        FOR ENV FILE
"""
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')

with open(_env_path, 'r') as file:
    for line in file:
        key, value = line.strip().split('=')

        os.environ[key] = value

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
