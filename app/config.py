import os
from dotenv import load_dotenv
load_dotenv()

class Config(object):
    STORAGE_PATH = os.getenv('STORAGE_PATH')
    SIGNER_SECRET_KEY="haram disebarluaskan gan"    