from server import app

if __name__ == '__main__':
    # print("REMINDER >> Create src/backend/secret.txt && src/backend/backups/current.json")
    # print("REMINDER >> Current secret is set to -> "+"UNKNOWN, CHECK YOURSELF")
    # TODO FIND A WAY TO GET THESE REMINDERS DISPLAYING AGAIN THAT ISNT THROUGH DOCKER ECHO
    app.run(debug=False)

# surely there is a less stupid way that gunicorn could have done this... right?