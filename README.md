# Deploy Your Flask Project To Heroku

This is an example repository that uses **Flask** and **Gunicorn** to deploy your project to Heroku.

# Problematic

Currently, Flask is not suitable for a production environment and it doesn't scale well (see [flask documentation on deployment](http://flask.pocoo.org/docs/1.0/deploying/)).

If we want to deploy our project to Heroku, we need a **Web Server Gateway Interface** (WSGI) such as **Gunicorn**.

# Solution

To overcome this obstacle we can use **Gunicorn** to aide us deploy our Flask project into a Heroku app.

This guide _assumes_ that you already had gone through the process of installing and authenticating the [Heroku Toolbelt](https://devcenter.heroku.com/articles/heroku-cli).

At this point you should be able to modify the Flask app `app.py`:
```python
"""Flask App Project."""

from flask import Flask, jsonify
app = Flask(__name__)


@app.route('/')
def index():
    """Return homepage."""
    json_data = {'Hello': 'World!'}
    return jsonify(json_data)


if __name__ == '__main__':
    app.run()
```

**WARNING:** If you change the file name (`app.py`) and the Flask **app** (`app = Flask(__name__)`) then remember to change Heroku's Procfile:
```
web: gunicorn <filename>:<app_name>
```

## Create Your Heroku App

You can also leave `your_app_name` empty if you want Heroku to create a randomized name.

```
$ heroku create your_app_name
Creating app... done, ⬢ your_app_name
https://your_app_name.herokuapp.com/ | https://git.heroku.com/your_app_name.git
```

## Deploy Your Project

Your project is going to be deploy using **gunicorn** as a web server using the **Procfile** and it will be detected as a Python project since it is declared in **runtime.txt**

* **Add necessary files and commit them**


That's it, you can visit your app now with `heroku open`.
