import random
import re


class ChatBot:
    
    negative_res =("no","nope","nay","not a chance","sorry")
    exit_commands = ("quit","pause","exit","goodbye","bye","farewell")
    
    def __init__(self):
        self.chat_responses = {
            'ask_about_product' : r'.*\s*product.*',
            'technical_support' : r'.*technical.*support.*',
            'about_returns' : r'.*\s*returnpolicy.*',
            'general_query' : r'.*how.*help.*'
        }
        
    def greet(self):
        self.name = input("Hello!Welcome to our customer support.What's your name?\n")
        will_help = input("Hi {self.name}, how can I assist you today?\n")
        if will_help in self.negative_res:
            print("Alright have a great day!")
            return
        self.chat()
        
    def make_exit(self,reply):
        for command in self.exit_command:
            if command in reply:
                print("Thanks fro reaching out, have a great day!")
                return True
            return False
        
        
        def chat(self):
            reply = input("Please tell me your query: ").lower()
            while  not self.make_exit(reply):
                reply = input(self.match_reply(reply))

    def match_reply(self,reply):
        for intent , regex_pattern in self.support_responses.items(): 
            found_match = re.search(regex_pattern, reply)
            if found_match and intent == 'ask_about_product' :
                return self.ask_about_product()
            elif found_match and intent == 'technical_support':
                return self.technical_support()
            elif found_match and intent == 'about_returns':
                return self.about_returns()
            elif found_match and intent == 'general_query':
                return self.general_query()
        return self.no_match_intent()
    
    def ask_about_product(self):
        responses = ("Our product is top-notch and has excellent reviews!\n",
                     "You can find all product details on our website.\n")
        return random.choice(responses)
    
    def technical_support(self):
        responses = ("Please visit our technical support page for detailed assisitance.\n",
                  "You can also call our tech support helpline for immediate help.\n")
        
    def about_returns(self):
        responses = ("We have a 38-day return policy.\n",
                     "Please ensure the product is its original condition when returning.\n")
        
    def  general_query(self):
        responses = ("How can I assist you further?\n",
                     "Is there anything else you'd like to know?\n")
        return random .choice(responses)
    
    def no_match_intent(self):
        responses = ("I'm sorry I didn't quite understand that. Can you please rephrase?\n",
                     "My aplogies, can you provide more details?\n")
        return random.choice(reponses)
    
bot = ChatBot()
bot.greet()
    
          
