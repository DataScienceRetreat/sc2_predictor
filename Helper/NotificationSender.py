import ntfy
import os

class NotificationSender:
    def __init__(self, title):
        self.pushover_key = 'INSERT_PUSHOVER_KEY'
        self.title = title

    def notify(self, text):
        # dirrty hack, but ya it works
        cmd = "ntfy -b pushover -o user_key {} -t '{}' send '{}'".format(
            self.pushover_key, self.title, text)
        return os.system(cmd)