from model import MarxBot

class ChatBot():
    """Dummy ChatBot"""
    def respond(self, text):
        return "I'm sorry, I don't understand you.".lower()

class PersonBot():
    """
    Ideally, this contains two deep networks.
    """
    def __init__(self, chatbot, personality):
        self.chatbot = chatbot
        self.personality = personality

    def respond(self, text, history):
        """Respond to text using chatbot + personality."""
        response = self.chatbot.respond(text)
        # Use response from chatbot to seed personality response.
        response = self.personality.respond(history)
        return response


def main():
    marx = MarxBot("datasets/nietzsche.txt", diversity=0.6)
    marx.load('params/model_params.h5')
    chatbot = ChatBot()
    chatty_marx = PersonBot(chatbot, marx)

    # REPL loop for chatting to marx
    history = True
    text = 'Hi! Nice to meet you, I am Karl.'.lower()
    chat_history = ''
    if history:
        chat_history = text
    while (True):
        message = input(text + '\n').lower()
        print('\n')
        if history:
            chat_history += message
        text = chatty_marx.respond(message, chat_history)

if __name__ == "__main__":
    main()


