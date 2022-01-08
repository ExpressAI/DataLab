
import os
import sys
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './plugins/general/')))
# from .plugins import general
# from .plugins.general import *
from .general import *

general_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./plugins/general/",
)

# function from modified xl-augmenter
from .plugins.general.slangificator.transformation import *
from .plugins.general.simple_cipher.transformation import *
from .plugins.general.replace_synonym.transformation import *
from .plugins.general.replace_hyponyms.transformation import *
from .plugins.general.replace_hypernyms.transformation import *
from .plugins.general.replace_greetings.transformation import *
from .plugins.general.replace_acronyms.transformation import *
from .plugins.general.reformat_date.transformation import *
from .plugins.general.insert_abbreviation.transformation import *
# from .plugins.general.factive_verb.transformation import *
from .plugins.general.emojify.transformation import *
from .plugins.general.correct_typo.transformation import *
# from .plugins.general.change_person_name_by_culture.transformation import *
from .plugins.general.change_person_name.transformation import *
from .plugins.general.change_color.transformation import *
from .plugins.general.change_city_name.transformation import *
from .plugins.general.britishize_americanize.transformation import *
from .plugins.general.add_typo.transformation import *
from .plugins.general.add_filler_words.transformation import *
from .plugins.general.abbreviate_weekday_month.transformation import *
from .plugins.general.abbreviate_country_state.transformation import *
from .plugins.general.abbreviate.transformation import *


# def import_from(module, name):
#     module = __import__(module, fromlist=[name])
#     return getattr(module, name)
#
# for file_name in os.listdir(general_dir):
#     if not file_name.endswith(".py") and file_name!="__pycache__":
#         print(f"{file_name}.transformation.{file_name}")
#         # my_module = importlib.import_module("abbreviate.transformation")
#         file_name = import_from(f"{file_name}.transformation", file_name)
#         print(file_name)

