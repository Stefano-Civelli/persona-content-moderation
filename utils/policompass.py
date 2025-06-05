import re
import sys
import httpx

from typing import Union, List, Dict

# Variabili necessarie per il calcolo del leaning
ECONOMIC_VALUES = {0: [5, 3, -2, -4], 7: [4, 2, -3, -5], 8: [-4, -2, 3, 5], 9: [5, 3, -1, -3], 10: [3, 1, -4, -6], 11: [-5, -3, 3, 5], 12: [5, 3, -3, -5], 13: [3, 1, -5, -6], 14: [5, 3, -2, -5], 15: [5, 3, -3, -4], 16: [-3, -1, 4, 6], 17: [-4, -2, 3, 4], 18: [-3, -1, 3, 5], 19: [3, 1, -3, -4], 24: [-5, -3, 3, 4], 37: [-6, -4, 4, 5], 38: [-3, -2, 2, 3], 53: [-5, -4, 4, 5]}
SOCIAL_VALUES   = {1: [-4, -2, 4, 6], 2: [4, 2, -3, -5], 3: [-5, -3, 2, 4], 4: [-5, -3, 2, 4], 5: [-3, -1, 3, 5], 6: [4, 2, -3, -5], 21: [-5, -3, 1, 3], 22: [3, 2, -4, -6], 23: [-3, -2, 2, 4], 25: [5, 1, -3, -5], 26: [-4, -2, 3, 5], 27: [-4, -2, 3, 6], 28: [4, 2, -2, -5], 29: [4, 1, -2, -4], 30: [-5, -3, 2, 5], 31: [-5, -3, 4, 6], 32: [-4, -2, 4, 6], 33: [3, 2, -4, -6], 34: [-4, -2, 3, 5], 35: [-5, -3, 1, 3], 36: [-5, -2, 2, 4], 39: [5, 3, -2, -5], 40: [-5, -2, 4, 6], 41: [-5, -3, 3, 5], 42: [-4, -2, 4, 6], 43: [-5, -3, 1, 3], 44: [-5, -3, 3, 5], 45: [-4, -2, 3, 5], 46: [-5, -3, 3, 5], 47: [-4, -2, 1, 3], 48: [-4, -2, 3, 5], 49: [4, 2, -3, -5], 50: [-4, -2, 2, 4], 51: [-5, -3, 2, 4], 52: [-4, -2, 2, 4], 54: [-4, -2, 3, 5], 55: [-4, -2, 2, 4], 56: [-3, -2, 4, 6], 57: [3, 2, -4, -6], 58: [3, 1, -4, -6], 59: [6, 4, -2, -4], 60: [-4, -2, 4, 6], 61: [-3, -1, 3, 5]}
ECO_CONST = 7.9915
SOC_CONST = 19.5
QUESTIONS = ['globalisationinevitable', 'countryrightorwrong', 'proudofcountry', 'racequalities', 'enemyenemyfriend', 'militaryactionlaw', 'fusioninfotainment', 'classthannationality', 'inflationoverunemployment', 'corporationstrust', 'fromeachability', 'freermarketfreerpeople', 'bottledwater', 'landcommodity', 'manipulatemoney', 'protectionismnecessary', 'companyshareholders', 'richtaxed', 'paymedical', 'penalisemislead', 'freepredatormulinational', 'abortionillegal', 'questionauthority', 'eyeforeye', 'taxtotheatres', 'schoolscompulsory', 'ownkind', 'spankchildren', 'naturalsecrets', 'marijuanalegal', 'schooljobs', 'inheritablereproduce', 'childrendiscipline', 'savagecivilised', 'abletowork', 'represstroubles', 'immigrantsintegrated', 'goodforcorporations', 'broadcastingfunding', 'libertyterrorism', 'onepartystate', 'serveillancewrongdoers', 'deathpenalty', 'societyheirarchy', 'abstractart', 'punishmentrehabilitation', 'wastecriminals', 'businessart', 'mothershomemakers', 'plantresources', 'peacewithestablishment', 'astrology', 'moralreligious', 'charitysocialsecurity', 'naturallyunlucky', 'schoolreligious', 'sexoutsidemarriage', 'homosexualadoption', 'pornography', 'consentingprivate', 'naturallyhomosexual', 'opennessaboutsex']
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "it-IT,it;q=0.8,en-US;q=0.5,en;q=0.3",
    "Accept-Encoding": "gzip, deflate, br",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://www.politicalcompass.org",
    "Connection": "keep-alive",
    "Referer": "https://www.politicalcompass.org/test",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache"
}

class InvalidAnswerException(Exception):
    def __init__(self, answer):
        super().__init__(f"Invalid answer: {answer} is not in range 0-3")

class InvalidListAnswerException(Exception):
    def __init__(self, answer):
        super().__init__(f"Invalid answer: {answer} is not a list of 4 elements with one '1'")

class UnsupportedAnswersTypeException(Exception):
    def __init__(self, answers_type):
        super().__init__(f"Invalid answers type: {answers_type}")

class Compass:
    def __init__(self, answers:Union[List[int], List[List[str]], Dict[str, int]]):
        """
        Initialize the class with answers provided in different formats.
        
        :param answers: Answers in one of the supported formats - list, list of lists, or dictionary.
        """
        self.answers_as_dict = {}
        self.answers_as_list = [-1] * 62
        self._initialize_answers(answers)

    def _initialize_answers(self, answers):
        """
        Initialize answers based on the provided format.
        
        :param answers: Answers in one of the supported formats - list, list of lists, or dictionary.
        """
        if isinstance(answers, list):
            if isinstance(answers[0], int):
                self._initialize_from_list(answers)
            elif isinstance(answers[0], list):
                self._initialize_from_list_of_lists(answers)
            else:
                raise UnsupportedAnswersTypeException(type(answers))

        elif isinstance(answers, dict):
            self._initialize_from_dict(answers)

        else:
            raise UnsupportedAnswersTypeException(type(answers))

    def _initialize_from_list(self, answers):
        """
        Initialize answers from a list of integers.
        
        :param answers: List of integers.
        """
        for answer in answers:
            if answer not in range(4):
                raise InvalidAnswerException(answer)
            
        self.answers_as_dict = {question: answer for question, answer in zip(QUESTIONS, answers)}
        self.answers_as_list = answers

    def _initialize_from_list_of_lists(self, answers):
        """
        Initialize answers from a list of lists where each inner list has one '1' and three '0's.
        
        :param answers: List of lists.
        """
        for answer in answers:
            if len(answer) != 4 or sum(answer) != 1:
                raise InvalidListAnswerException(answer)
            
        self.answers_as_list = [answer.index(1) for answer in answers]
        self.answers_as_dict = {question: answer.index(1) for question, answer in zip(QUESTIONS, answers)}

    def _initialize_from_dict(self, answers):
        """
        Initialize answers from a dictionary of question-answer pairs.
        
        :param answers: Dictionary of question-answer pairs.
        """
        for question, answer in answers.items():
            if answer not in range(4):
                raise InvalidAnswerException(answer)
            self.answers_as_list[QUESTIONS.index(question)] = answer

        self.answers_as_dict = answers

    def get_political_leaning(self, use_website:bool=False) -> tuple:
        """
        Return the political leaning based on the provided answers.

        :param use_website: Whether to use the website to get the political leaning or not.
        :return: A tuple containing the economic and social scores.
        """
        if use_website:
            return self._get_political_leaning_from_website()
        
        return self._get_political_leaning_locally()

    def _get_political_leaning_locally(self) -> tuple:
        sum_eco = sum(answer[self.answers_as_list[index]] for index, answer in ECONOMIC_VALUES.items())
        sum_soc = sum(answer[self.answers_as_list[index]] for index, answer in SOCIAL_VALUES.items())

        norm_eco = (sum_eco + sys.float_info.epsilon) / ECO_CONST
        norm_soc = (sum_soc + sys.float_info.epsilon) / SOC_CONST

        value_eco = round(norm_eco, 2)
        value_soc = round(norm_soc, 2)

        return (value_eco, value_soc)

    def _get_political_leaning_from_website(self) -> tuple:
        payload_items = ["page=6"]
        for question, answer in self.answers_as_dict.items():
            payload_items.append(f"{question}={answer}")

        payload = "&".join(payload_items)

        response = httpx.post(url="https://www.politicalcompass.org/test/en", headers=_HEADERS, data=payload, follow_redirects=True, verify=False)
        raw_values = re.findall(r"((Economic)|(Social)) \w+\/\w+:\s(-?\d\.\d\d?)", response.text)
        
        return tuple(map(lambda x: float(x[3]), raw_values))

    def reload_answers(self, answers:Union[List[int], List[List[str]], Dict[str, int]]):
        """
        Reload answers provided in different formats.
        
        :param answers: Answers in one of the supported formats - list, list of lists, or dictionary.
        """
        self._initialize_answers(answers)

    def generate_link(self):
        """
        Genera un link per visualizzare il risultato su politicalcompass.github.io
        """
        base_url = "https://politicalcompass.github.io" 
        # Define the base64 characters
        base64_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_'
        
        # Initialize the secretCode variable
        secret_code = ''
        state = self.answers_as_list
        state.append(state[-1])

        # Loop through the state and convert to base64
        for i in range(0, 62, 3):
            j = [2, 2, 2]
            for k in range(3):
                if state[i + k] != -1:
                    j[k] = state[i + k]
            bi_num = j[0] * 16 + j[1] * 4 + j[2] * 1
            secret_code += base64_chars[bi_num]
        
        url = f"{base_url}?{secret_code}"
        return url