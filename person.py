from model import ChatBot

class PersonBot():
    """
    Ideally, this contains two deep networks.
    """
    def __init__(self, chatbot, personality):
        self.chatbot = chatbot
        self.personality = personality

    def respond(self, text):
        """Respond to text using chatbot + personality."""
        response = self.chatbot.respond(text)
        response = response.replace('.',',')
        # print('chatbot:', response)
        history = text + response
        # Use response from chatbot to seed personality response.
        response += self.personality.respond(history.lower().replace('\n', ''))
        # print('final:', response)
        return response


def main():
    marx = ChatBot(["data/marx.txt"], diversity=0.6)
    marx.load('params/marx_model_params_10_epochs')
    chatbot = ChatBot(["data/movie.txt"], diversity=0.5)
    chatbot.load('params/movie_model_params_2_epochs')
    chatty_marx = PersonBot(chatbot, marx)

    # REPL loop for chatting to marx
    text = 'Hi! Nice to meet you, I am Karl. I\'ve been watching some movies to learn modern lingo. Talk to me about philosophy.'.lower()
    chat_history = text
    while (True):
        message = input(text + '\n').lower()
        chat_history += message
        text = chatty_marx.respond(chat_history)

if __name__ == "__main__":
    main()


